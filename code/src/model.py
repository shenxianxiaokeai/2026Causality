# -*- coding: utf-8 -*-
"""
src/model.py

PIHAM-compatible diffusion-spectral recovery model

核心思想：
- 保留 PIHAM 的输入/输出接口，兼容 main_inference.py
- 不再用失败的重构优化器直接学 Q
- 改为：
    1) 对 obs / corrected 两层做归一化图扩散
    2) 构造融合扩散算子
    3) 对融合算子做谱嵌入
    4) 用 k-means 恢复社区分配 Q
    5) 再由 Q 反推出 B 和 Delta

输出仍保留：
- Q
- B
- Delta
- shared_adj

适用场景：
- 无特征图
- 强社区图
- 你现在的 obs/corrected 因果双层输入
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from typing import List, Tuple, Optional, Dict, Any

from sklearn.cluster import KMeans


# =========================================================
# 工具函数
# =========================================================
def _as_device(device_like: Any) -> torch.device:
    if isinstance(device_like, torch.device):
        return device_like
    return torch.device(str(device_like))


def _symmetrize(mat: Tensor) -> Tensor:
    if mat.dim() == 2:
        return 0.5 * (mat + mat.T)
    elif mat.dim() == 3:
        return 0.5 * (mat + mat.transpose(1, 2))
    else:
        raise ValueError(f"Unsupported tensor dim for symmetrization: {mat.dim()}")


def _safe_eye(n: int, device: torch.device, dtype=torch.float32) -> Tensor:
    return torch.eye(n, device=device, dtype=dtype)


def _row_normalize(X: Tensor, eps: float = 1e-12) -> Tensor:
    norm = torch.norm(X, dim=1, keepdim=True) + eps
    return X / norm


# =========================================================
# PIHAM-compatible diffusion-spectral solver
# =========================================================
class PIHAM(nn.Module):
    def __init__(
        self,
        U_mu_prior: Tensor,
        U_std_prior: Tensor,
        V_mu_prior: Tensor,
        V_std_prior: Tensor,
        W_mu_prior: Tensor,
        W_std_prior: Tensor,
        Hcategorical_mu_prior: Tensor,
        Hcategorical_std_prior: Tensor,
        Hpoisson_mu_prior: Tensor,
        Hpoisson_std_prior: Tensor,
        Hgaussian_mu_prior: Tensor,
        Hgaussian_std_prior: Tensor,
        K: int,
        N: int,
        L: int,
        Z_categorical: int,
        P_poisson: int,
        P_gaussian: int,
        configuration: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.K = int(K)
        self.N = int(N)
        self.L = int(L)
        self.Z_categorical = int(Z_categorical)
        self.P_poisson = int(P_poisson)
        self.P_gaussian = int(P_gaussian)

        self.configuration = configuration
        self.device = _as_device(configuration.get("device", "cpu"))

        # 兼容保留
        self.layer_types: List[str] = ["gaussian"] * self.L
        self.loss_history: List[float] = []
        self.best_seed: Optional[int] = None

        # 新求解器配置
        self.diffusion_alpha = float(configuration.get("diffusion_alpha", 0.15))  # PPR alpha
        self.obs_weight = float(configuration.get("obs_weight", 0.5))              # obs/corrected 融合权重
        self.use_ipw_view = bool(configuration.get("use_ipw_view", True))
        self.embedding_norm = bool(configuration.get("embedding_norm", True))
        self.kmeans_n_init = int(configuration.get("kmeans_n_init", 20))

        # 缓存结果
        self.Q_: Optional[Tensor] = None
        self.B_: Optional[Tensor] = None
        self.Delta_: Optional[Tensor] = None
        self.shared_adj_: Optional[Tensor] = None

        self._cached_A: Optional[Tensor] = None

        # 兼容壳子占位
        self.U = torch.empty(0, device=self.device)
        self.V = torch.empty(0, device=self.device)

    # =========================================================
    # 兼容接口
    # =========================================================
    def initialize(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    def get_UVWH(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        empty = torch.empty(0, device=self.device)
        B = self.B_ if self.B_ is not None else torch.empty((self.K, self.K), device=self.device)
        return empty, empty, B, empty, empty, empty

    def set_layer_types(self, A: Tensor, layer_types: Optional[List[str]] = None) -> None:
        if layer_types is None:
            self.layer_types = ["gaussian"] * self.L
        else:
            if len(layer_types) != self.L:
                raise ValueError(f"layer_types length must be L={self.L}, got {len(layer_types)}")
            self.layer_types = list(layer_types)

    # =========================================================
    # 核心：扩散 + 谱恢复
    # =========================================================
    def _normalize_adj(self, A: Tensor) -> Tensor:
        """
        Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}
        """
        A = _symmetrize(A)
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

    def _ppr_diffusion(self, A: Tensor) -> Tensor:
        """
        PPR-style diffusion:
            S = alpha * (I - (1-alpha) * A_norm)^(-1)
        """
        A_norm = self._normalize_adj(A)
        I = _safe_eye(A.size(0), A.device, A.dtype)
        alpha = self.diffusion_alpha
        M = I - (1.0 - alpha) * A_norm
        S = alpha * torch.linalg.inv(M)
        S = _symmetrize(S)
        S = torch.clamp(S, min=0.0)
        return S

    def _build_fused_operator(self, A: Tensor) -> Tensor:
        """
        A: [L, N, N]
        默认：
        - 若 L=1，则直接对该层做扩散
        - 若 L>=2，则对 obs/corrected 两层分别扩散后融合
        """
        if A.dim() != 3:
            raise ValueError(f"Expected A.shape=(L,N,N), got {tuple(A.shape)}")

        if A.size(0) == 1 or not self.use_ipw_view:
            S = self._ppr_diffusion(A[0])
            return S

        A_obs = A[0]
        A_ipw = A[1]

        S_obs = self._ppr_diffusion(A_obs)
        S_ipw = self._ppr_diffusion(A_ipw)

        w = self.obs_weight
        M = w * S_obs + (1.0 - w) * S_ipw
        M = _symmetrize(M)
        return M

    def _spectral_embedding(self, M: Tensor) -> Tensor:
        """
        对融合算子做谱嵌入，取前 K 个特征向量
        """
        # eigh 适合对称矩阵
        eigvals, eigvecs = torch.linalg.eigh(M)
        # 取最大的 K 个特征向量
        U = eigvecs[:, -self.K:]
        if self.embedding_norm:
            U = _row_normalize(U)
        return U

    def _kmeans_assign(self, U: Tensor, random_state: int) -> Tensor:
        U_np = U.detach().cpu().numpy()
        km = KMeans(
            n_clusters=self.K,
            random_state=int(random_state),
            n_init=self.kmeans_n_init,
        )
        labels = km.fit_predict(U_np)

        Q = np.zeros((self.N, self.K), dtype=np.float32)
        Q[np.arange(self.N), labels] = 1.0
        return torch.tensor(Q, dtype=torch.float32, device=self.device)

    def _estimate_B(self, A_mean: Tensor, Q: Tensor) -> Tensor:
        """
        由固定 Q 反推出 block matrix B
        用最小二乘：
            B = (Q^T Q)^(-1) Q^T A Q (Q^T Q)^(-1)
        """
        QtQ = Q.T @ Q
        epsI = 1e-8 * _safe_eye(self.K, self.device, A_mean.dtype)
        QtQ_inv = torch.linalg.inv(QtQ + epsI)
        B = QtQ_inv @ (Q.T @ A_mean @ Q) @ QtQ_inv
        B = _symmetrize(B)
        B = torch.clamp(B, min=0.0)
        return B

    def _estimate_delta(self, A: Tensor, A_shared: Tensor) -> Tensor:
        Delta = A - A_shared.unsqueeze(0)
        Delta = _symmetrize(Delta)
        eye = _safe_eye(self.N, self.device, Delta.dtype).unsqueeze(0)
        Delta = Delta * (1.0 - eye)
        return Delta

    def _compute_modularity_proxy(self, A: Tensor, labels: np.ndarray) -> float:
        """
        仅用于多 seed 选择，不对外暴露。
        用简单 modularity proxy 选更好的谱解。
        """
        A_np = A.detach().cpu().numpy()
        m = A_np.sum() / 2.0
        if m <= 0:
            return -1e9
        deg = A_np.sum(axis=1)
        q = 0.0
        for i in range(A_np.shape[0]):
            for j in range(A_np.shape[0]):
                if labels[i] == labels[j]:
                    q += A_np[i, j] - deg[i] * deg[j] / (2.0 * m)
        q /= (2.0 * m)
        return float(q)

    # =========================================================
    # forward / loss
    # =========================================================
    def forward(self, A: Tensor, X_gaussian: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        if self.Q_ is None or self.B_ is None or self.Delta_ is None or self.shared_adj_ is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.Q_, self.shared_adj_, self.Delta_

    def _get_omega(self) -> Tensor:
        if self.L <= 0:
            raise ValueError("L must be positive.")
        return torch.ones(self.L, device=self.device) / float(self.L)

    def get_loss_components(self, A: Tensor, X_gaussian: Optional[Tensor] = None) -> Dict[str, Tensor]:
        A = A.to(self.device).float()
        Q, A_shared, Delta = self.forward(A, X_gaussian=X_gaussian)
        A_hat = A_shared.unsqueeze(0) + Delta

        # 分层归一化 reconstruction，更稳
        layer_losses = []
        for l in range(self.L):
            num = torch.mean((A[l] - A_hat[l]) ** 2)
            den = torch.mean(A[l] ** 2) + 1e-8
            layer_losses.append(num / den)
        loss_corr = torch.mean(torch.stack(layer_losses))

        loss_sp = torch.mean(torch.abs(Delta))
        omega = self._get_omega().view(self.L, 1, 1)
        Delta_center = torch.sum(omega * Delta, dim=0)
        loss_cent = torch.mean(Delta_center ** 2)

        lambda1 = float(self.configuration.get("lambda1", 1e-3))
        lambda2 = float(self.configuration.get("lambda2", 1e-2))
        loss_total = loss_corr + lambda1 * loss_sp + lambda2 * loss_cent

        return {
            "loss_corr": loss_corr,
            "loss_sp": loss_sp,
            "loss_cent": loss_cent,
            "loss_total": loss_total,
        }

    def get_total_loss(self, A: Tensor, X_gaussian: Optional[Tensor] = None) -> Tensor:
        return self.get_loss_components(A, X_gaussian=X_gaussian)["loss_total"]

    def get_neg_log_posterior(
        self,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        likelihood_weight: float = 1.0,
    ) -> Tensor:
        _ = X_categorical, X_poisson, likelihood_weight
        return self.get_total_loss(A, X_gaussian=X_gaussian)

    # =========================================================
    # fit
    # =========================================================
    def fit(
        self,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        gamma: float = 0.1,
        tolerance: float = 1e-8,
        num_iter: int = 2000,
        likelihood_weight: float = 1.0,
        learning_rate: float = 1e-2,
        verbose: bool = True,
        N_seeds: int = 10,
        print_likelihoods: bool = True,
    ) -> List[float]:
        """
        不再做梯度下降，而是：
        1) 构造扩散融合算子
        2) 谱嵌入
        3) 多 seed k-means
        4) 选择 modularity proxy 最优解
        """
        _ = X_categorical, X_poisson, X_gaussian
        _ = gamma, tolerance, num_iter, likelihood_weight, learning_rate

        A = A.to(self.device).float()
        self._cached_A = A.detach().clone()

        M = self._build_fused_operator(A)
        U_embed = self._spectral_embedding(M)

        best_score = -1e18
        best_Q = None
        best_labels = None
        best_B = None
        best_Delta = None
        best_shared = None

        seeds = np.random.choice(np.arange(1000), size=int(N_seeds), replace=False)

        # 用平均层反推共享骨架
        A_mean = torch.mean(A, dim=0)

        self.loss_history = []

        for s in seeds:
            Q = self._kmeans_assign(U_embed, random_state=int(s))
            labels = torch.argmax(Q, dim=1).detach().cpu().numpy()

            B = self._estimate_B(A_mean, Q)
            A_shared = Q @ B @ Q.T
            A_shared = _symmetrize(A_shared)
            eye = _safe_eye(self.N, self.device, A_shared.dtype)
            A_shared = A_shared * (1.0 - eye)

            Delta = self._estimate_delta(A, A_shared)

            # 用 modularity proxy 选更好的划分
            score = self._compute_modularity_proxy(M, labels)

            self.Q_ = Q
            self.B_ = B
            self.shared_adj_ = A_shared
            self.Delta_ = Delta

            loss_dict = self.get_loss_components(A)
            cur_loss = float(loss_dict["loss_total"].item())
            self.loss_history.append(cur_loss)

            if verbose:
                print(
                    f"[seed {s}] "
                    f"score={score:.6f} | "
                    f"loss={cur_loss:.6f} | "
                    f"active_clusters={len(np.unique(labels))}"
                )

            if score > best_score:
                best_score = score
                best_Q = Q.detach().clone()
                best_labels = labels.copy()
                best_B = B.detach().clone()
                best_Delta = Delta.detach().clone()
                best_shared = A_shared.detach().clone()
                self.best_seed = int(s)

        self.Q_ = best_Q
        self.B_ = best_B
        self.Delta_ = best_Delta
        self.shared_adj_ = best_shared

        if print_likelihoods:
            print(f"best seed = {self.best_seed}, best modularity proxy = {best_score:.6f}")
            uniq, cnt = np.unique(best_labels, return_counts=True)
            print("active clusters:", len(uniq))
            print("cluster counts:", {int(u): int(c) for u, c in zip(uniq, cnt)})

        return self.loss_history

    # =========================================================
    # 保存结果
    # =========================================================
    def save_results(self, folder_name: str, file_name: str) -> None:
        outfile = folder_name + "theta" + file_name

        if self.Q_ is None:
            raise RuntimeError("Model has not been fit yet.")

        payload = {
            "Q": self.Q_.detach().cpu().numpy(),
            "B": self.B_.detach().cpu().numpy(),
            "Delta": self.Delta_.detach().cpu().numpy(),
            "shared_adj": self.shared_adj_.detach().cpu().numpy(),
            "U": np.empty((0,), dtype=np.float32),
            "V": np.empty((0,), dtype=np.float32),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
            "best_seed": np.array([-1 if self.best_seed is None else self.best_seed], dtype=np.int64),
        }

        np.savez_compressed(outfile + ".npz", **payload)
        print(f'Inferred parameters saved in: {outfile + ".npz"}')
        print('To load: theta=np.load(filename), then e.g. theta["Q"], theta["B"], theta["Delta"]')

    # =========================================================
    # Hessian / Covariance（兼容壳子）
    # =========================================================
    def compute_Hessian(
        self,
        A: Tensor,
        X_categorical: Tensor,
        X_poisson: Tensor,
        X_gaussian: Tensor,
        likelihood_weight: float = 1.0,
    ) -> None:
        _ = A, X_categorical, X_poisson, X_gaussian, likelihood_weight
        print("[Warning] Hessian computation is not enabled in the diffusion-spectral PIHAM.")
        self.Hessian = None

    def is_neg_Hessian_pos_def(self, eps: float = 0.0) -> None:
        _ = eps
        print("[Warning] Hessian check is skipped because Hessian is not computed.")

    def compute_Covariance(self, eps: float = 1e-6, make_psd: bool = False) -> None:
        _ = eps, make_psd
        print("[Warning] Covariance computation is skipped because Hessian is not computed.")
        self.Covariance = None

    def get_Covariance(self, diagonal_only: bool = False):
        _ = diagonal_only
        if hasattr(self, "Covariance"):
            return self.Covariance
        return None


# =========================================================
# assign_priors：兼容旧主脚本
# =========================================================
def assign_priors(
    K: int,
    N: int,
    L: int,
    Z_categorical: int,
    P_poisson: int,
    P_gaussian: int,
    configuration: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = _as_device(configuration.get("device", "cpu"))

    def _zeros(shape):
        return torch.zeros(shape, device=device)

    def _ones(shape):
        return torch.ones(shape, device=device)

    U_mu = _zeros((N, K))
    V_mu = _zeros((N, K))
    W_mu = _zeros((L, K, K))

    Hc_mu = _zeros((K, Z_categorical))
    Hp_mu = _zeros((K, P_poisson))
    Hg_mu = _zeros((K, P_gaussian))

    U_std = _ones((N, K))
    V_std = _ones((N, K))
    W_std = _ones((L, K, K))

    Hc_std = _ones((K, Z_categorical))
    Hp_std = _ones((K, P_poisson))
    Hg_std = _ones((K, P_gaussian))

    return (
        U_mu, V_mu, W_mu,
        Hc_mu, Hp_mu, Hg_mu,
        U_std, V_std, W_std,
        Hc_std, Hp_std, Hg_std,
    )
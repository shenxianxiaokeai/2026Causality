import os
import time
import json
import yaml
import numpy as np
import torch
from argparse import ArgumentParser
from src.model import PIHAM, assign_priors

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def _safe_device(device_cfg: str) -> torch.device:
    """把配置里的 device 字符串转成 torch.device，并在 cuda 不可用时自动回退 cpu。"""
    dev = str(device_cfg).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] 配置指定 cuda，但当前环境不可用，自动回退到 cpu")
        return torch.device("cpu")
    return torch.device(dev)


def _summarize_A(A: torch.Tensor) -> list:
    """统计每一层 A 的 min/max/mean，便于快速检查数据是否正常。"""
    stats = []
    for l in range(A.size(0)):
        x = A[l]
        stats.append(
            {
                "layer": int(l),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
                "mean": float(x.mean().item()),
                "dtype": str(x.dtype).replace("torch.", ""),
            }
        )
    return stats


def main():
    # =========================
    # 1) 解析命令行参数
    # =========================
    p = ArgumentParser()
    p.add_argument("-f", "--in_folder", type=str, default="data/input/EmailEU/corrected", help="输入目录")
    p.add_argument("-d", "--data_file", type=str, default="EmailEU_ipw.pt", help="数据文件名")
    p.add_argument("-K", "--K", type=int, default=42, help="社团数 K")
    p.add_argument("--no_hessian", action="store_true", help="当前结构学习模型默认跳过 Hessian")
    p.add_argument(
        "--layer_types",
        type=str,
        default="",
        help='兼容旧接口保留；当前模型不依赖 layer_types',
    )
    p.add_argument("--diffusion_alpha", type=float, default=None, help="扩散系数 alpha，命令行优先")
    p.add_argument("--obs_weight", type=float, default=None, help="观测层权重，命令行优先")
    p.add_argument("--lambda1", type=float, default=None, help="变化项稀疏正则系数，命令行优先")
    p.add_argument("--lambda2", type=float, default=None, help="中心约束系数，命令行优先")
    args = p.parse_args()

    # =========================
    # 2) 输出目录准备
    # =========================
    output_folder = os.path.join("data", "output", "main", "EmailEU", "corrected")
    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # 3) 读取推断配置
    # =========================
    setting_path = os.path.join("src", "setting_inference.yaml")
    with open(setting_path, "r", encoding="utf-8") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    if args.diffusion_alpha is not None:
        configuration["diffusion_alpha"] = args.diffusion_alpha
    if args.obs_weight is not None:
        configuration["obs_weight"] = args.obs_weight
    if args.lambda1 is not None:
        configuration["lambda1"] = args.lambda1
    if args.lambda2 is not None:
        configuration["lambda2"] = args.lambda2

    device = _safe_device(configuration.get("device", "cpu"))
    configuration["device"] = str(device)

    with open(os.path.join(output_folder, "setting_inference.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(configuration, f, allow_unicode=True)

    if "rseed" in configuration:
        torch.manual_seed(int(configuration["rseed"]))
        np.random.seed(int(configuration["rseed"]))

    # =========================
    # 4) 读取数据
    # =========================
    data_path = os.path.join(args.in_folder, args.data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    data = torch.load(data_path, map_location=device)

    if "A" not in data:
        raise KeyError('数据文件中必须包含 key="A"（邻接张量）')

    A = data["A"].to(device)

    if A.dim() != 3 or A.size(1) != A.size(2):
        raise ValueError(f"A 的形状应为 (L,N,N)，当前 A.shape={tuple(A.shape)}")

    N = A.size(1)
    L = A.size(0)
    K = int(args.K)

    X_categorical = data.get("X_categorical", torch.empty((N, 0), device=device)).to(device)
    X_poisson = data.get("X_poisson", torch.empty((N, 0), device=device)).to(device)
    X_gaussian = data.get("X_gaussian", torch.empty((N, 0), device=device)).to(device)

    Z_categorical = int(X_categorical.size(1))
    P_poisson = int(X_poisson.size(1))
    P_gaussian = int(X_gaussian.size(1))

    # =========================
    # 5) 打印输入摘要
    # =========================
    print("\n==================== [Input Summary] ====================")
    print(f"Data file: {data_path}")
    print(f"A shape: {tuple(A.shape)}  (L,N,N)  L={L}, N={N}")
    print(f"Covariates dims: Z_categorical={Z_categorical}, P_poisson={P_poisson}, P_gaussian={P_gaussian}")
    print(f"Device: {device}")
    print("A stats per layer:")
    for s in _summarize_A(A):
        print(f"  layer {s['layer']}: min={s['min']:.6g}, max={s['max']:.6g}, mean={s['mean']:.6g}, dtype={s['dtype']}")
    print("=========================================================\n")

    # =========================
    # 6) 实例化模型并训练
    # =========================
    tic = time.time()

    priors = assign_priors(K, N, L, Z_categorical, P_poisson, P_gaussian, configuration)
    (
        U_mu_prior,
        V_mu_prior,
        W_mu_prior,
        Hcategorical_mu_prior,
        Hpoisson_mu_prior,
        Hgaussian_mu_prior,
        U_std_prior,
        V_std_prior,
        W_std_prior,
        Hcategorical_std_prior,
        Hpoisson_std_prior,
        Hgaussian_std_prior,
    ) = priors

    model = PIHAM(
        U_mu_prior,
        U_std_prior,
        V_mu_prior,
        V_std_prior,
        W_mu_prior,
        W_std_prior,
        Hcategorical_mu_prior,
        Hcategorical_std_prior,
        Hpoisson_mu_prior,
        Hpoisson_std_prior,
        Hgaussian_mu_prior,
        Hgaussian_std_prior,
        K,
        N,
        L,
        Z_categorical,
        P_poisson,
        P_gaussian,
        configuration,
    )

    # 关键：把整个模型迁移到目标设备
    model = model.to(device)

    # 兼容旧接口：当前模型不真正依赖 layer_types
    if args.layer_types.strip():
        layer_types = [x.strip().lower() for x in args.layer_types.split(",") if x.strip()]
        model.set_layer_types(A, layer_types=layer_types)
    else:
        model.set_layer_types(A)

    print("[Info] layer_types =", model.layer_types)

    model.fit(
        A,
        X_categorical,
        X_poisson,
        X_gaussian,
        tolerance=configuration["tolerance"],
        num_iter=configuration["num_iter"],
        learning_rate=configuration["learning_rate"],
        verbose=configuration["verbose"],
        N_seeds=configuration["N_seeds"],
        print_likelihoods=True,
    )

    # =========================
    # 7) 读取最终损失摘要
    # =========================
    with torch.no_grad():
        loss_dict = model.get_loss_components(A, X_gaussian=X_gaussian)
        final_loss = float(loss_dict["loss_total"].item())
        final_corr = float(loss_dict["loss_corr"].item())
        final_sp = float(loss_dict["loss_sp"].item())
        final_cent = float(loss_dict["loss_cent"].item())

    print(f"\n[Summary] Final total loss = {final_loss:.6f}")
    print(f"[Summary] loss_corr = {final_corr:.6f}")
    print(f"[Summary] loss_sp   = {final_sp:.6f}")
    print(f"[Summary] loss_cent = {final_cent:.6f}")

    # =========================
    # 8) 跳过 Hessian / Covariance
    # =========================
    print("\n[Info] Hessian/Covariance is disabled in the current structural learning model.")

    # =========================
    # 9) 保存结果与运行信息
    # =========================
    tag = f"_{args.data_file.replace('.pt', '')}_K{K}"
    model.save_results(folder_name=output_folder + os.sep, file_name=tag)

    run_info = {
        "data_path": data_path,
        "K": K,
        "N": N,
        "L": L,
        "layer_types": model.layer_types,
        "A_stats": _summarize_A(A),
        "device": str(device),
        "configuration": configuration,
        "final_loss": final_loss,
        "loss_corr": final_corr,
        "loss_sp": final_sp,
        "loss_cent": final_cent,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "no_hessian": True,
    }

    with open(os.path.join(output_folder, f"run_info{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    toc = time.time()
    print(f"\n ---- Time elapsed: {np.round(toc - tic, 4)} seconds ----\n")


if __name__ == "__main__":
    main()
import argparse
import os

import torch

# =========================
# Hyper-parameters
# =========================
EPS = 1e-3
CLIP_Q = 0.95
MAX_WEIGHT = 10.0
SCALE_MODE = "mean"  # "mean" / "fro" / "none"
H_MODE = "degree"    # "degree" / "proximity"


def parse_args():
    parser = argparse.ArgumentParser(description="Construct IPW corrected adjacency layer.")
    parser.add_argument(
        "--in_path",
        type=str,
        default=os.path.join("input", "EmailEU", "raw", "EmailEU_raw.pt"),
        help="Input .pt file path.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.path.join("input", "EmailEU", "corrected", "EmailEU_ipw.pt"),
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--h_mode",
        type=str,
        default=H_MODE,
        choices=["degree", "proximity"],
        help="Propensity feature mode when E_true is absent.",
    )
    return parser.parse_args()


def _finalize_propensity(P: torch.Tensor):
    P = torch.clamp(P, min=EPS, max=1.0 - EPS)
    eye = torch.eye(P.size(0), dtype=torch.bool, device=P.device)
    P = P.clone()
    P[eye] = 0.0
    return P


def _normalize_score_matrix(score: torch.Tensor):
    score = score.clone()
    eye = torch.eye(score.size(0), dtype=torch.bool, device=score.device)
    score[eye] = 0.0
    max_val = float(score.max().item())
    if max_val <= 0.0:
        return torch.zeros_like(score)
    return score / (max_val + 1e-12)


def estimate_propensity_degree(A_obs: torch.Tensor):
    """
    Degree-only mode:
    H_ij^deg = {d_i, d_j, d_i*d_j}
    with configuration-model style estimator:
    pi_hat_ij = d_i * d_j / m
    """
    d = A_obs.sum(dim=1) + 1e-8
    m = A_obs.sum() + 1e-8
    P = torch.outer(d, d) / m
    return _finalize_propensity(P)


def estimate_propensity_proximity(A_obs: torch.Tensor):
    """
    Proximity-only mode:
    H_ij^prox = {CN_ij, Jaccard_ij, AA_ij, RA_ij}
    """
    B = (A_obs > 0).float()
    eye = torch.eye(B.size(0), dtype=torch.bool, device=B.device)
    B[eye] = 0.0

    deg = B.sum(dim=1)
    cn = B @ B

    deg_i = deg.unsqueeze(1)
    deg_j = deg.unsqueeze(0)
    union = deg_i + deg_j - cn

    jaccard = torch.zeros_like(cn)
    valid_union = union > 0
    jaccard[valid_union] = cn[valid_union] / union[valid_union]

    inv_log_deg = torch.zeros_like(deg)
    valid_aa = deg > 1.0
    inv_log_deg[valid_aa] = 1.0 / torch.log(deg[valid_aa] + 1e-12)
    aa = (B * inv_log_deg.unsqueeze(0)) @ B.T

    inv_deg = torch.zeros_like(deg)
    valid_ra = deg > 0.0
    inv_deg[valid_ra] = 1.0 / deg[valid_ra]
    ra = (B * inv_deg.unsqueeze(0)) @ B.T

    cn_norm = _normalize_score_matrix(cn)
    jaccard_norm = _normalize_score_matrix(jaccard)
    aa_norm = _normalize_score_matrix(aa)
    ra_norm = _normalize_score_matrix(ra)

    # Proximity-only aggregation (no degree-product term).
    score = 0.25 * (cn_norm + jaccard_norm + aa_norm + ra_norm)
    return _finalize_propensity(score)


def estimate_propensity_from_graph(A_obs: torch.Tensor, h_mode: str):
    if h_mode == "degree":
        return estimate_propensity_degree(A_obs)
    if h_mode == "proximity":
        return estimate_propensity_proximity(A_obs)
    raise ValueError(f"Unsupported h_mode: {h_mode}")


def normalize_ipw_to_obs(A_obs: torch.Tensor, A_ipw: torch.Tensor, mode: str = "mean"):
    if mode == "none":
        return A_ipw, 1.0

    if mode == "mean":
        obs_mean = A_obs.mean()
        ipw_mean = A_ipw.mean()
        scale = obs_mean / (ipw_mean + 1e-12)
        return A_ipw * scale, float(scale.item())

    if mode == "fro":
        obs_norm = torch.norm(A_obs, p="fro")
        ipw_norm = torch.norm(A_ipw, p="fro")
        scale = obs_norm / (ipw_norm + 1e-12)
        return A_ipw * scale, float(scale.item())

    raise ValueError(f"Unsupported SCALE_MODE: {mode}")


def main():
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path
    h_mode = args.h_mode

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = torch.load(in_path, map_location="cpu")
    if "A" not in data:
        raise KeyError(f"Missing key 'A' in input file: {in_path}")

    A = data["A"].float()
    if A.dim() != 3 or A.size(0) < 1:
        raise ValueError(f"Expected A.shape = (L, N, N), got {tuple(A.shape)}")

    # Use the first layer as observed adjacency.
    A_obs = A[0].clone()

    # Safety: symmetrize and remove self loops.
    A_obs = 0.5 * (A_obs + A_obs.T)
    eye = torch.eye(A_obs.size(0), dtype=torch.bool, device=A_obs.device)
    A_obs[eye] = 0.0

    # Choose propensity source.
    if "E_true" in data:
        P = data["E_true"].float()
        ipw_source = "E_true"
    else:
        P = estimate_propensity_from_graph(A_obs, h_mode=h_mode)
        ipw_source = f"estimated_{h_mode}"

    if P.shape != A_obs.shape:
        raise ValueError(f"P shape mismatch: P={tuple(P.shape)}, A_obs={tuple(A_obs.shape)}")

    denom = torch.clamp(P, min=EPS)
    obs_mask = A_obs > 0

    A_ipw = torch.zeros_like(A_obs)
    A_ipw[obs_mask] = A_obs[obs_mask] / denom[obs_mask]
    A_ipw[eye] = 0.0

    # Step 1: quantile clipping.
    if obs_mask.any():
        q_cap = torch.quantile(A_ipw[obs_mask], CLIP_Q)
        cap = min(float(q_cap.item()), MAX_WEIGHT)
        A_ipw = torch.clamp(A_ipw, max=cap)
    else:
        cap = 0.0

    # Step 2: scale align to A_obs.
    A_ipw, scale = normalize_ipw_to_obs(A_obs, A_ipw, mode=SCALE_MODE)

    # Step 3: re-symmetrize.
    A_ipw = 0.5 * (A_ipw + A_ipw.T)
    A_ipw[eye] = 0.0

    # Two-layer view: [observed, corrected]
    A_views = torch.stack([A_obs, A_ipw], dim=0)

    out_data = dict(data)
    out_data["A"] = A_views
    out_data["A_obs"] = A_obs
    out_data["A_ipw"] = A_ipw
    out_data["view_names"] = ["obs", "corrected"]
    out_data["ipw_source"] = ipw_source
    out_data["ipw_h_mode"] = h_mode
    out_data["ipw_clip_q"] = CLIP_Q
    out_data["ipw_max_weight"] = MAX_WEIGHT
    out_data["ipw_scale_mode"] = SCALE_MODE
    out_data["ipw_scale_factor"] = scale

    torch.save(out_data, out_path)

    obs_mean = A_obs.mean().item()
    obs_max = A_obs.max().item()
    ipw_mean = A_ipw.mean().item()
    ipw_max = A_ipw.max().item()
    nonzero_ipw_mean = A_ipw[obs_mask].mean().item() if obs_mask.any() else 0.0

    print("========== IPW Layer Construction ==========")
    print(f"Input file:      {in_path}")
    print(f"Output file:     {out_path}")
    print(f"Propensity src:  {ipw_source}")
    print(f"h_mode:          {h_mode}")
    print(f"A_obs shape:     {tuple(A_obs.shape)}")
    print(f"A_views shape:   {tuple(A_views.shape)}")
    print(f"CLIP_Q:          {CLIP_Q}")
    print(f"MAX_WEIGHT:      {MAX_WEIGHT}")
    print(f"SCALE_MODE:      {SCALE_MODE}")
    print(f"scale factor:    {scale:.6f}")
    print("------ Layer stats ------")
    print(f"A_obs : mean = {obs_mean:.6f}, max = {obs_max:.6f}")
    print(f"A_ipw : mean = {ipw_mean:.6f}, max = {ipw_max:.6f}")
    print(f"nonzero IPW mean = {nonzero_ipw_mean:.6f}")
    print(f"clip cap = {cap:.6f}")
    print("View mapping: A[0] = obs, A[1] = corrected")
    print("===========================================")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()

import json
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def labels_to_communities(labels):
    labels = np.asarray(labels)
    comms = []
    for c in np.unique(labels):
        comms.append(set(np.where(labels == c)[0]))
    return comms


def compute_modularity(A, y_pred):
    A = to_numpy(A)
    G = nx.from_numpy_array(A)
    communities = labels_to_communities(y_pred)
    return modularity(G, communities, weight="weight")


def load_Q_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print("NPZ keys:", data.files)

    # 优先找常见字段
    candidate_keys = ["Q", "theta", "membership", "pi"]
    for key in candidate_keys:
        if key in data.files:
            Q = data[key]
            print(f"Use '{key}' as Q, shape={Q.shape}")
            return Q

    raise KeyError(
        f"在 {npz_path} 中没有找到 Q。当前可用键为: {data.files}。\n"
        f"请确认哪个字段是社区分配矩阵，然后把 candidate_keys 改一下。"
    )


def load_data_and_labels(pt_path):
    data = torch.load(pt_path, map_location="cpu")

    print("PT keys:", list(data.keys()) if isinstance(data, dict) else type(data))

    if not isinstance(data, dict):
        raise TypeError("pt 文件内容不是 dict，请先检查数据保存格式。")

    # 读取真值标签
    y_true = None
    for key in ["y_true", "labels", "y", "community"]:
        if key in data:
            y_true = to_numpy(data[key]).astype(int)
            print(f"Use '{key}' as y_true, shape={y_true.shape}")
            break
    if y_true is None:
        raise KeyError("pt 文件中未找到真实标签字段，如 y_true / labels / y / community")

    # 读取邻接矩阵
    if "A" not in data:
        raise KeyError("pt 文件中未找到 'A' 字段")
    A = to_numpy(data["A"])
    print("A shape:", A.shape)

    return data, A, y_true


def evaluate(npz_path, pt_path, save_json_path=None):
    # 1) 读 Q
    Q = load_Q_from_npz(npz_path)
    Q = to_numpy(Q)

    if Q.ndim != 2:
        raise ValueError(f"Q 应该是二维矩阵 [N, K]，当前 shape={Q.shape}")

    y_pred = np.argmax(Q, axis=1)

    # 2) 读数据和标签
    data_dict, A, y_true = load_data_and_labels(pt_path)


    Q = np.load(r".\data\output\main\EmailEU\corrected\theta_EmailEU_ipw_K42.npz")["Q"]
    y_pred = np.argmax(Q, axis=1)
    y_true = data_dict["y_true"].cpu().numpy() if hasattr(data_dict["y_true"], "cpu") else np.asarray(
        data_dict["y_true"])

    print("y_pred bincount:", np.bincount(y_pred))
    print("y_true bincount:", np.bincount(y_true))

    print("\nPred distribution in each 32-node block:")
    for i in range(4):
        seg = y_pred[i * 32:(i + 1) * 32]
        vals, cnts = np.unique(seg, return_counts=True)
        print(f"block {i}: {dict(zip(vals, cnts))}")

    # A0 = data_dict["A"][0].cpu().numpy() if hasattr(data_dict["A"][0], "cpu") else np.asarray(data_dict["A"][0])

    # plt.figure(figsize=(5, 5))
    # plt.imshow(A0, cmap="gray_r")
    # plt.title("A[0] original order")
    # plt.show()

    # order_pred = np.argsort(y_pred)
    # A0_pred = A0[order_pred][:, order_pred]

    # plt.figure(figsize=(5, 5))
    # plt.imshow(A0_pred, cmap="gray_r")
    # plt.title("A[0] sorted by predicted labels")
    # plt.show()

    # order_true = np.argsort(y_true)
    # A0_true = A0[order_true][:, order_true]

    # plt.figure(figsize=(5, 5))
    # plt.imshow(A0_true, cmap="gray_r")
    # plt.title("A[0] sorted by current y_true")
    # plt.show()

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true 长度 {len(y_true)} 与 y_pred 长度 {len(y_pred)} 不一致")

    # 3) 计算指标
    results = {}
    results["num_nodes"] = int(len(y_true))
    results["num_classes_true"] = int(len(np.unique(y_true)))
    results["num_classes_pred"] = int(len(np.unique(y_pred)))

    results["NMI"] = float(normalized_mutual_info_score(y_true, y_pred))
    results["ARI"] = float(adjusted_rand_score(y_true, y_pred))

    # 模块度：单层或多层都兼容
    if A.ndim == 2:
        results["Modularity"] = float(compute_modularity(A, y_pred))

    elif A.ndim == 3:
        mods = []
        for i in range(A.shape[0]):
            mod_i = compute_modularity(A[i], y_pred)
            results[f"Modularity_layer_{i}"] = float(mod_i)
            mods.append(mod_i)
        results["Modularity_mean"] = float(np.mean(mods))

    else:
        raise ValueError(f"A 维度不支持，当前 shape={A.shape}")

    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    if save_json_path is not None:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {save_json_path}")

    return results


if __name__ == "__main__":
    npz_path = r".\data\output\main\EmailEU\corrected\theta_EmailEU_ipw_K42.npz"
    pt_path = r"D:.\data\input\EmailEU\corrected\EmailEU_ipw.pt"
    save_json_path = r".\data\output\main\EmailEU\corrected\run_info_EmailEU_ipw_K42.json"

    evaluate(npz_path, pt_path, save_json_path)


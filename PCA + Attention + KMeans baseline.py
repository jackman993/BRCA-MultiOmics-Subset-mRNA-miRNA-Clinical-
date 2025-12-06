#!/usr/bin/env python
# PCA + Attention + KMeans baseline
#
# 用途：
#   - 輸入：基因 / miRNA 表達矩陣（CSV）
#           預設：rows = 基因, columns = 樣本（可用 --samples-in-rows 改）
#   - 過程：標準化 → PCA → cosine similarity → softmax attention → KMeans 分群
#   - 輸出：
#       - clusters.csv        每個 sample 的分群
#       - pca_2d.csv          PCA 前兩軸座標 + cluster
#       - pca_scatter.png     PCA 散佈圖（按群著色）
#       - attention_heatmap.png  注意力熱圖
#       - cluster_sizes.png   各群樣本數
#       - （若有給臨床檔）km_survival.png  KM 曲線
#       - summary.json        總結（樣本數、特徵數、silhouette、logrank p 等）
#
# 執行範例：
#   python yangzhou_engine_v1_baseline.py \
#       --expression brca_mirna_matrix.csv \
#       --clinical brca_clinical.csv \
#       --n-clusters 3 \
#       --n-components 10 \
#       --output-dir results_brca
#
#   若 CSV 是 rows = samples, columns = genes，請加：
#       --samples-in-rows
#
# 臨床檔案格式：
#   - 建議有一欄 sample_id（會用來對齊 expression）
#   - 生存欄位名稱可用：
#       time:  OS_time / os_time / time / survival_time
#       event: OS_event / os_event / event / status
#   - 會自動偵測，只畫 KM、做 log-rank test（前兩群之間）。


import argparse
import os
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 生存分析（可選，如果沒有 lifelines 就略過 survival 部分）
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def set_seed(seed: int = 42) -> None:
    """固定隨機種子，讓結果可重現。"""
    random.seed(seed)
    np.random.seed(seed)


def load_expression(path: str, samples_in_rows: bool = False) -> pd.DataFrame:
    """
    讀取表達矩陣 CSV。

    預設：
        rows = features (基因 / miRNA)
        columns = samples
    若 samples_in_rows=True：
        rows = samples
        columns = features
    """
    df = pd.read_csv(path, index_col=0)
    if samples_in_rows:
        # index = sample_id
        return df
    # 否則假設 rows 是基因，columns 是樣本 → 轉置
    return df.T


def run_pca_attention(
    X: np.ndarray,
    n_components: int = 10,
    attention_temperature: float = 1.0,
):
    """
    標準化 → PCA → cosine similarity → softmax attention。
    回傳：
        X_pca : (n_samples, n_components)
        att   : (n_samples, n_samples) 注意力矩陣
        pca, scaler 方便之後重用
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    # cosine similarity
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8
    X_norm = X_pca / norms
    cosine_sim = X_norm @ X_norm.T

    # softmax attention
    att = np.exp(cosine_sim / attention_temperature)
    att = att / att.sum(axis=1, keepdims=True)
    return X_pca, att, pca, scaler


def cluster_kmeans(
    X_pca: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
):
    """
    對 PCA 結果做 KMeans，回傳：
        labels0 : 0~K-1 的 cluster label
        sil      : silhouette score
        km       : KMeans 模型
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels0 = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels0) if n_clusters > 1 else float("nan")
    return labels0, sil, km


def plot_pca_scatter(X_pca, labels0, out_path: str) -> None:
    """PCA 前兩軸散佈圖。"""
    plt.figure(figsize=(6, 5))
    unique = np.unique(labels0)
    for lab in unique:
        idx = labels0 == lab
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=f"Cluster {lab + 1}",
            alpha=0.7,
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (first two components)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_attention_heatmap(att: np.ndarray, out_path: str) -> None:
    """注意力矩陣熱圖。"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(att, cmap="viridis")
    plt.title("Attention (cosine similarity softmax)")
    plt.xlabel("Sample")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cluster_sizes(labels0: np.ndarray, out_path: str) -> None:
    """各群樣本數長條圖。"""
    vals, counts = np.unique(labels0, return_counts=True)
    plt.figure(figsize=(5, 4))
    plt.bar([f"C{v + 1}" for v in vals], counts)
    plt.ylabel("Sample count")
    plt.title("Cluster sizes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_survival(clin_df: pd.DataFrame, labels0: np.ndarray, out_path: str):
    """
    若有 lifelines，且找到 time/event 欄位，就畫 KM + 做簡單 log-rank。
    回傳：
        dict(time_col, event_col, p_value) 或 None
    """
    if not HAS_LIFELINES:
        return None

    time_col_candidates = ["OS_time", "os_time", "time", "survival_time"]
    event_col_candidates = ["OS_event", "os_event", "event", "status"]

    time_col = next((c for c in time_col_candidates if c in clin_df.columns), None)
    event_col = next((c for c in event_col_candidates if c in clin_df.columns), None)

    if time_col is None or event_col is None:
        return None

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6, 5))

    unique = np.unique(labels0)
    for lab in unique:
        mask = labels0 == lab
        t = clin_df.loc[mask, time_col]
        e = clin_df.loc[mask, event_col]
        if len(t) == 0:
            continue
        kmf.fit(t, e, label=f"Cluster {lab + 1}")
        kmf.plot(ci_show=False)

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title("Kaplan–Meier by cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 簡單示範：只做前兩群的 log-rank
    if len(unique) >= 2:
        m1 = labels0 == unique[0]
        m2 = labels0 == unique[1]
        res = logrank_test(
            clin_df.loc[m1, time_col],
            clin_df.loc[m2, time_col],
            event_observed_A=clin_df.loc[m1, event_col],
            event_observed_B=clin_df.loc[m2, event_col],
        )
        return {
            "time_col": time_col,
            "event_col": event_col,
            "p_value": float(res.p_value),
        }
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Yangzhou Engine v1 – PCA + Attention + KMeans baseline",
    )
    parser.add_argument(
        "--expression",
        required=True,
        help="Expression matrix CSV（預設 rows=genes, cols=samples）",
    )
    parser.add_argument(
        "--clinical",
        help="Optional clinical CSV（建議有 sample_id, OS_time/OS_event 等欄位）",
    )
    parser.add_argument(
        "--samples-in-rows",
        action="store_true",
        help="若表達檔案 rows = samples, columns = genes，請加這個 flag",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="PCA components 數（預設 10）",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="KMeans 分群數（預設 3）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="attention temperature（越小越尖銳，預設 1.0）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed（預設 42）",
    )
    parser.add_argument(
        "--output-dir",
        default="yangzhou_results",
        help="輸出資料夾（預設 yangzhou_results）",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 讀 expression
    expr_df = load_expression(
        args.expression,
        samples_in_rows=args.samples_in_rows,
    )
    sample_ids = expr_df.index.to_numpy()
    X = expr_df.to_numpy()

    # PCA + attention
    X_pca, att, pca, scaler = run_pca_attention(
        X,
        n_components=args.n_components,
        attention_temperature=args.temperature,
    )

    # clustering
    labels0, sil, km = cluster_kmeans(
        X_pca,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )
    # 為了好看，把 0..K-1 轉成 1..K
    labels = labels0 + 1

    # 存 cluster 結果
    cluster_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cluster": labels,
        }
    )
    cluster_df.to_csv(
        os.path.join(args.output_dir, "clusters.csv"),
        index=False,
    )

    # 存 PCA 2D 座標
    pca_df = pd.DataFrame(
        X_pca[:, :2],
        columns=["PC1", "PC2"],
    )
    pca_df.insert(0, "sample_id", sample_ids)
    pca_df["cluster"] = labels
    pca_df.to_csv(
        os.path.join(args.output_dir, "pca_2d.csv"),
        index=False,
    )

    # 圖
    plot_pca_scatter(
        X_pca,
        labels0,
        os.path.join(args.output_dir, "pca_scatter.png"),
    )
    plot_attention_heatmap(
        att,
        os.path.join(args.output_dir, "attention_heatmap.png"),
    )
    plot_cluster_sizes(
        labels0,
        os.path.join(args.output_dir, "cluster_sizes.png"),
    )

    # 簡單 summary
    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_components": int(X_pca.shape[1]),
        "n_clusters": int(args.n_clusters),
        "silhouette": float(sil) if sil == sil else None,
    }

    # 生存分析（如果有 clinical）
    if args.clinical is not None:
        clin_df = pd.read_csv(args.clinical)

        # 若有 sample_id 欄位就照 sample_ids 順序對齊
        if "sample_id" in clin_df.columns:
            clin_df = clin_df.set_index("sample_id").reindex(sample_ids)

        surv_result = run_survival(
            clin_df,
            labels0,
            os.path.join(args.output_dir, "km_survival.png"),
        )
        if surv_result is not None:
            summary["survival_logrank_p"] = surv_result["p_value"]
            summary["time_col"] = surv_result["time_col"]
            summary["event_col"] = surv_result["event_col"]

    # 存 summary.json
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Yangzhou Engine v1 baseline finished ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

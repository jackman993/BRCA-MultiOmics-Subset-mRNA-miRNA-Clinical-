All terminology and formatting have been converted into a fully academic style.

Author: TaiScience Research Unit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# 1. Data Generation
# ---------------------------

def generate_synthetic_genomic_data():
    """
    Generate a synthetic genomics dataset representing three cancer subtypes
    with distinct expression patterns and survival characteristics.

    Returns:
        expr_df: (N × G) expression DataFrame
        survival_df: patient-level clinical and survival data
        true_labels: ground-truth subtype index
    """
    np.random.seed(42)

    n_patients = 200
    n_genes = 50

    # Subtype sizes
    s1, s2, s3 = 70, 80, 50

    # Expression profiles
    expr1 = np.random.normal(8, 1.5, (s1, n_genes))
    expr1[:, :5] += np.random.normal(3, 0.5, (s1, 5))

    expr2 = np.random.normal(6, 1.2, (s2, n_genes))
    expr2[:, 5:10] += np.random.normal(2, 0.5, (s2, 5))

    expr3 = np.random.normal(4, 1.0, (s3, n_genes))
    expr3[:, 10:15] += np.random.normal(1.5, 0.3, (s3, 5))

    # Combine
    expression = np.vstack([expr1, expr2, expr3])
    patient_ids = [f"Patient_{i:03d}" for i in range(1, n_patients + 1)]
    true_labels = np.array([0]*s1 + [1]*s2 + [2]*s3)

    # Gene names
    gene_names = (
        ["S100A4", "S100A8", "S100A9", "S100A11", "S100A12"]
        + ["EGFR", "TP53", "MYC", "BRCA1", "KRAS"]
        + [f"GENE_{i:02d}" for i in range(11, n_genes + 1)]
    )

    expr_df = pd.DataFrame(expression, index=patient_ids, columns=gene_names)

    # Survival simulation
    times, events = [], []
    for label in true_labels:
        if label == 0:
            t = np.random.exponential(15) + 5
            e = np.random.choice([0, 1], p=[0.3, 0.7])
        elif label == 1:
            t = np.random.exponential(25) + 10
            e = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            t = np.random.exponential(40) + 15
            e = np.random.choice([0, 1], p=[0.7, 0.3])
        times.append(t)
        events.append(e)

    survival_df = pd.DataFrame({
        "patient_id": patient_ids,
        "time": times,
        "event": events,
        "age": np.random.normal(65, 12, n_patients),
        "stage": np.random.choice(["I", "II", "III", "IV"], n_patients)
    })

    return expr_df, survival_df, true_labels


# ---------------------------
# 2. Core Analysis Pipeline
# ---------------------------

def run_genomic_analysis():
    expr_df, survival_df, true_labels = generate_synthetic_genomic_data()

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expr_df)

    # PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # Attention matrix (cosine similarity + softmax)
    cos_sim = cosine_similarity(X_pca)
    attention = np.exp(cos_sim) / np.exp(cos_sim).sum(axis=1, keepdims=True)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Biomarker ranking via inter-cluster variance
    merged = expr_df.copy()
    merged["cluster"] = cluster_labels
    cluster_means = merged.groupby("cluster").mean()
    inter_var = cluster_means.var(axis=0)
    top_genes = inter_var.sort_values(ascending=False).head(10).index.tolist()

    # Panel figure construction
    plt.figure(figsize=(20, 16))
    gs = plt.GridSpec(3, 4)

    # ---------------------------
    # Panel A: PCA scatter plot
    # ---------------------------
    ax1 = plt.subplot(gs[0, :2])
    palette = ["#D62728", "#1F77B4", "#2CA02C"]

    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    s=60, alpha=0.7, label=f"Cluster {c}",
                    color=palette[c], edgecolor="white")

    ax1.set_title("PCA Projection of Patient Samples", fontsize=14)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # ---------------------------
    # Panel B: Attention heatmap
    # ---------------------------
    ax2 = plt.subplot(gs[0, 2:])
    order = np.argsort(cluster_labels)
    att_sorted = attention[order][:, order]

    sns.heatmap(att_sorted, cmap="viridis", ax=ax2)
    ax2.set_title("Attention Matrix (Cosine–Softmax)", fontsize=14)
    ax2.set_xlabel("Samples (sorted by cluster)")
    ax2.set_ylabel("Samples (sorted by cluster)")

    # ---------------------------
    # Panel C: Kaplan–Meier curves
    # ---------------------------
    ax3 = plt.subplot(gs[1, :2])
    kmf = KaplanMeierFitter()

    surv_df = survival_df.copy()
    surv_df["cluster"] = cluster_labels

    for c in np.unique(cluster_labels):
        subset = surv_df[surv_df["cluster"] == c]
        kmf.fit(subset["time"], subset["event"], label=f"Cluster {c}")
        kmf.plot(ax=ax3, linewidth=2)

    ax3.set_title("Kaplan–Meier Survival Analysis", fontsize=14)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Survival Probability")
    ax3.grid(True, alpha=0.3)

    # ---------------------------
    # Panel D: Biomarker heatmap
    # ---------------------------
    ax4 = plt.subplot(gs[1, 2:])
    expr_top = expr_df[top_genes].iloc[order].T
    sns.heatmap(expr_top, cmap="RdBu_r", ax=ax4,
                cbar_kws={"label": "Expression Level"})
    ax4.set_title("Top Biomarker Gene Expression", fontsize=14)
    ax4.set_ylabel("Genes")
    ax4.set_xlabel("Samples (sorted by cluster)")

    # ---------------------------
    # Panel E: Biomarker ranking
    # ---------------------------
    ax5 = plt.subplot(gs[2, :2])
    inter_top = inter_var.sort_values(ascending=False).head(10)
    ax5.bar(inter_top.index, inter_top.values, color="#1F77B4")
    ax5.set_title("Biomarker Importance (Inter-Cluster Variance)", fontsize=14)
    ax5.set_ylabel("Variance")
    ax5.set_xticklabels(inter_top.index, rotation=45, ha="right")

    # ---------------------------
    # Panel F: Analysis summary
    # ---------------------------
    ax6 = plt.subplot(gs[2, 2:])
    ax6.axis("off")

    summary = (
        "Genomic Subtype Analysis Summary\n"
        "----------------------------------\n"
        f"Total patients: {expr_df.shape[0]}\n"
        f"Number of clusters: {len(np.unique(cluster_labels))}\n"
        f"Top biomarkers: {len(top_genes)}\n"
        f"PCA components: {pca.n_components_}\n"
        f"Explained variance (PC1+PC2): "
        f"{sum(pca.explained_variance_ratio_[:2]):.1%}\n"
    )

    ax6.text(0, 1, summary, va="top", fontsize=12, family="monospace")

    plt.tight_layout()
    plt.show()

    return {
        "expr": expr_df,
        "survival": survival_df,
        "cluster": cluster_labels,
        "attention": attention,
        "top_genes": top_genes,
        "pca": X_pca,
    }


# ---------------------------    
if __name__ == "__main__":
    print("Running academic genomic analysis pipeline…")
    results = run_genomic_analysis()
    print("Analysis complete.")

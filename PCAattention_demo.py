import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非互動模式
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

# ============================================================
#  File paths (與原版相同，放在工作資料夾即可)
# ============================================================
clinical_path = "BRCA.clin.merged.picked.txt"
mrna_path     = "BRCA.medianexp.txt"
mirna_path    = "BRCA-FFPE.miRseq_mature_RPM.txt"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#  Load data
# ============================================================
print("=" * 60)
print("🧬 BRCA mRNA / miRNA PCA + Attention Subtyping Demo")
print("=" * 60)
print("\n📂 Loading data ...")

clinical_df = pd.read_csv(clinical_path, sep="\t", index_col=0, engine="python")
mrna_df = pd.read_csv(mrna_path, sep="\t", index_col=0, engine="python", skiprows=[1])
mirna_df = pd.read_csv(mirna_path, sep="\t", index_col=0, engine="python", skiprows=[1])

mrna_df = mrna_df.apply(pd.to_numeric, errors="coerce")
mirna_df = mirna_df.apply(pd.to_numeric, errors="coerce")

print("✅ Files loaded successfully.")
print(f"   Clinical : {clinical_df.shape} (features x samples)")
print(f"   mRNA     : {mrna_df.shape} (genes x samples)")
print(f"   miRNA    : {mirna_df.shape} (miRNAs x samples)")

# ============================================================
#  PCA utilities
# ============================================================
def pca_plot(df, title, output_name):
    """
    執行 PCA 並繪製 PC1–PC2 散點圖（樣本層級）。
    df: genes/miRNAs x samples
    """
    print(f"\n🔧 Running PCA for: {title} ...")

    df_T = df.T  # samples x features
    df_numeric = df_T.select_dtypes(include=[np.number])
    df_filled = df_numeric.fillna(0)

    # 移除零變異特徵
    var = df_filled.var()
    keep_cols = var[var > 0].index
    df_clean = df_filled[keep_cols]

    if df_clean.shape[0] < 2 or df_clean.shape[1] < 2:
        print("   ⚠️ 數據維度過小，略過 PCA。")
        return None, None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean)

    # 先做 2 維版本給散點圖
    pca2 = PCA(n_components=2)
    pcs_2d = pca2.fit_transform(X)

    # 另外做較多維度，給後面 variance / attention 用
    n_full = min(50, X.shape[0], X.shape[1])
    pca_full = PCA(n_components=n_full)
    pcs_full = pca_full.fit_transform(X)

    print(f"   ✅ PCA completed. PC1: {pca2.explained_variance_ratio_[0]*100:.2f}% "
          f"PC2: {pca2.explained_variance_ratio_[1]*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        pcs_2d[:, 0], pcs_2d[:, 1],
        c=range(len(pcs_2d)), cmap="viridis", s=40,
        alpha=0.8, edgecolors="white"
    )
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)",
                  fontsize=12, fontweight="bold")
    ax.set_title(f"{title} - PCA", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.text(
        0.02, 0.98, f"N = {len(pcs_2d)} samples",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sample index")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 PCA scatter saved: {out_path}")

    return pca_full, pcs_full, df_clean.index  # pca, scores, sample_ids


def plot_variance_explained(pca, title, output_name, n_show=20):
    """繪製單一 PC 及累積變異比例。"""
    if pca is None:
        return

    n = min(n_show, len(pca.explained_variance_ratio_))
    xs = np.arange(1, n + 1)
    var_ratio = pca.explained_variance_ratio_[:n] * 100
    cumsum = np.cumsum(var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.bar(xs, var_ratio, color="#3498db", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=12, fontweight="bold")
    ax1.set_title("(A) Variance by Each PC", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.plot(xs, cumsum, "o-", color="#e74c3c", linewidth=2, markersize=5)
    ax2.fill_between(xs, cumsum, alpha=0.25, color="#e74c3c")
    ax2.axhline(80, color="#2ecc71", linestyle="--", linewidth=2, label="80%")
    ax2.axhline(95, color="#f39c12", linestyle="--", linewidth=2, label="95%")
    ax2.set_xlabel("Number of Components", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, fontweight="bold")
    ax2.set_title("(B) Cumulative Variance", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 Variance plot saved: {out_path}")


# ============================================================
#  Attention Engine (Engine-Level API)
# ============================================================
def softmax(x, axis=1):
    """
    Numerically stable softmax.
    
    Parameters:
    -----------
    x : np.ndarray - input scores
    axis : int - axis to apply softmax
    
    Returns:
    --------
    np.ndarray - softmax probabilities
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_qkv(features, d_k=32, random_state=42):
    """
    Transformer-style QKV Attention with Linear Projections.
    
    Parameters:
    -----------
    features : np.ndarray - input features (n_samples × dim)
    d_k : int - projection dimension for Q, K, V
    random_state : int - random seed for reproducibility
    
    Returns:
    --------
    A : np.ndarray - attention weights matrix (n_samples × n_samples)
    Z : np.ndarray - attention-pooled embedding (n_samples × d_k)
    
    Formula:
    --------
    Q = features @ Wq
    K = features @ Wk
    V = features @ Wv
    scores = Q @ K.T / sqrt(d_k)
    A = softmax(scores)
    Z = A @ V
    """
    np.random.seed(random_state)
    
    n_samples, dim = features.shape
    
    # Linear projections (Xavier initialization)
    Wq = np.random.randn(dim, d_k) / np.sqrt(dim)
    Wk = np.random.randn(dim, d_k) / np.sqrt(dim)
    Wv = np.random.randn(dim, d_k) / np.sqrt(dim)
    
    # Project to Q, K, V
    Q = features @ Wq
    K = features @ Wk
    V = features @ Wv
    
    # Scaled dot-product attention
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Numerically stable softmax
    A = np.exp(scores - scores.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    
    # Attention-pooled embedding
    Z = A @ V
    
    return A, Z


def attention_pooling(X, temperature=1.0):
    """
    Simple Self-Attention Pooling (Q=K=V variant).
    
    Parameters:
    -----------
    X : np.ndarray - PCA後資料 (samples × features)
    temperature : float - softmax temperature (lower = sharper attention)
    
    Returns:
    --------
    X_pooled : np.ndarray - attention-pooled representation (samples × features)
    A : np.ndarray - attention weights matrix (samples × samples)
    """
    # Q = K = V (self-attention without projection)
    d = X.shape[1]
    scores = X @ X.T / (np.sqrt(d) * temperature)
    A = softmax(scores, axis=1)
    X_pooled = A @ X
    
    return X_pooled, A


def distance_weighted_attention(X, k_neighbors=10):
    """
    Distance-Weighted Attention (基於距離的加權 Attention).
    
    使用 k-NN 找近鄰，根據距離給權重（距離越近權重越大）。
    比 one-hot 更平滑，能捕捉局部結構。
    
    Parameters:
    -----------
    X : np.ndarray - input features (n_samples × features)
    k_neighbors : int - 考慮的鄰居數量
    
    Returns:
    --------
    att_matrix : np.ndarray - attention weights (n_samples × n_samples)
    X_pooled : np.ndarray - attention-pooled embedding (n_samples × features)
    
    Formula:
    --------
    weights = 1 / (distances + epsilon)
    weights = weights / weights.sum()  # normalize
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_samples = X.shape[0]
    k = min(k_neighbors, n_samples - 1)  # 確保 k 不超過樣本數
    
    att_matrix = np.zeros((n_samples, n_samples))
    
    # 找 k 近鄰
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, nn_indices = nn.kneighbors(X)
    
    for i in range(n_samples):
        # 根據距離給權重（距離越近權重越大）
        weights = 1 / (distances[i] + 1e-6)
        weights = weights / weights.sum()
        att_matrix[i, nn_indices[i]] = weights
    
    # Attention-pooled embedding
    X_pooled = att_matrix @ X
    
    return att_matrix, X_pooled


def compute_similarity_matrix(X, method='cosine'):
    """
    Compute patient-patient similarity matrix.
    
    Parameters:
    -----------
    X : np.ndarray - feature matrix (samples × features)
    method : str - 'cosine' or 'dot'
    
    Returns:
    --------
    sim : np.ndarray - similarity matrix (samples × samples)
    """
    if method == 'cosine':
        X_norm = normalize(X)  # L2 row-normalization
        sim = np.clip(X_norm @ X_norm.T, -1.0, 1.0)
    else:  # dot product
        sim = X @ X.T
    
    return sim


# ============================================================
#  Attention-enhanced patient similarity & subtyping
# ============================================================
def run_attention_subtyping(
    pcs_full,
    sample_ids,
    data_label,
    output_prefix,
    n_pcs_for_attention=20,
    n_clusters=3,
    temperature=1.0,
):
    """
    使用 PCA 分數 + Attention Pooling 建構病人表徵，並做亞型分群。

    pcs_full : array (N_samples x N_PCs)
    sample_ids : Index / list, 對應樣本 ID
    data_label : str, 例如 "mRNA" / "miRNA"
    output_prefix : str, 檔名 prefix
    """
    if pcs_full is None or pcs_full.shape[0] < 3:
        print(f"   ⚠️ {data_label}: insufficient samples for attention engine.")
        return None

    print(f"\n🧠 Running attention-enhanced subtyping on {data_label} ...")

    n_samples = pcs_full.shape[0]
    d = min(n_pcs_for_attention, pcs_full.shape[1])
    X = pcs_full[:, :d]  # 取前幾個主成分

    # 1) Compute similarity matrix (for visualization)
    sim = compute_similarity_matrix(X, method='cosine')

    # 2) QKV Attention Pooling (Engine-Level API)
    d_k = min(32, d)  # projection dimension
    attn, Z_attn = attention_qkv(X, d_k=d_k, random_state=42)
    
    print(f"   📐 Input: {X.shape} → Attention: {attn.shape}, Pooled: {Z_attn.shape}")

    # 4) simple clustering in the attended space
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    labels = kmeans.fit_predict(Z_attn)
    print(f"   ✅ Attention engine completed. n_samples={n_samples}, n_clusters={n_clusters}")

    # ---- Save subtype label table ----
    subtype_df = pd.DataFrame(
        {"SampleID": list(sample_ids), "Subtype": labels.astype(int)},
        index=sample_ids,
    )
    csv_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_subtypes.csv")
    subtype_df.to_csv(csv_path)
    print(f"   📄 Subtype assignments saved: {csv_path}")

    # ---- Plot similarity heatmap ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="viridis", aspect="auto")
    ax.set_title(f"{data_label} – Patient Similarity (cosine)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Patient")
    ax.set_ylabel("Patient")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_sim = os.path.join(OUTPUT_DIR, f"{output_prefix}_similarity_heatmap.png")
    plt.savefig(out_sim, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 Similarity heatmap saved: {out_sim}")

    # ---- Plot attention heatmap ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap="magma", aspect="auto")
    ax.set_title(f"{data_label} – Attention Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Context patient")
    ax.set_ylabel("Query patient")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_attn = os.path.join(OUTPUT_DIR, f"{output_prefix}_attention_heatmap.png")
    plt.savefig(out_attn, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 Attention heatmap saved: {out_attn}")

    # ---- PCA scatter colored by subtype (using first two PCs of Z_attn via PCA again) ----
    pca_vis = PCA(n_components=2)
    Z_vis = pca_vis.fit_transform(Z_attn)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        Z_vis[:, 0], Z_vis[:, 1],
        c=labels, cmap="tab10",
        s=45, alpha=0.9, edgecolors="white"
    )
    ax.set_xlabel("Attention-Pooled PC1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention-Pooled PC2", fontsize=12, fontweight="bold")
    ax.set_title(f"{data_label} – Attention-Enhanced Subtypes", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(
        *scatter.legend_elements(num=n_clusters),
        title="Subtype", loc="best", frameon=True
    )
    # 兼容新舊版 matplotlib，安全設定大小
    handles = getattr(legend, 'legend_handles', None) or getattr(legend, 'legendHandles', [])
    for lh in handles:
        try:
            lh.set_sizes([50])
        except AttributeError:
            pass  # Line2D 等物件沒有 set_sizes
    plt.tight_layout()

    out_sub = os.path.join(OUTPUT_DIR, f"{output_prefix}_subtype_scatter.png")
    plt.savefig(out_sub, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 Subtype scatter saved: {out_sub}")

    return subtype_df


# ============================================================
#  Run pipeline
# ============================================================
print("\n" + "=" * 60)
print("📊 PCA + variance analysis")
print("=" * 60)

# ---- mRNA ----
pca_mrna_full, pcs_mrna_full, mrna_samples = pca_plot(
    mrna_df, "BRCA mRNA Expression", "pca_mrna.png"
)
if pca_mrna_full is not None:
    plot_variance_explained(pca_mrna_full, "mRNA Expression", "variance_mrna.png")

# ---- miRNA ----
pca_mirna_full, pcs_mirna_full, mirna_samples = pca_plot(
    mirna_df, "BRCA miRNA Expression", "pca_mirna.png"
)
if pca_mirna_full is not None:
    plot_variance_explained(pca_mirna_full, "miRNA Expression", "variance_mirna.png")

print("\n" + "=" * 60)
print("🧠 Attention-enhanced patient similarity & subtyping")
print("=" * 60)

# mRNA attention engine（樣本數多，結果比較穩定）
if pca_mrna_full is not None:
    run_attention_subtyping(
        pcs_mrna_full,
        sample_ids=mrna_samples,
        data_label="mRNA",
        output_prefix="mrna",
        n_pcs_for_attention=20,
        n_clusters=3,
        temperature=0.5,
    )

# miRNA attention engine（樣本較少，只當成示範）
if pca_mirna_full is not None:
    run_attention_subtyping(
        pcs_mirna_full,
        sample_ids=mirna_samples,
        data_label="miRNA",
        output_prefix="mirna",
        n_pcs_for_attention=10,
        n_clusters=3,
        temperature=0.7,
    )

print("\n" + "=" * 60)
print("🎉 Demo finished. All outputs are under:", OUTPUT_DIR)
for f in sorted(os.listdir(OUTPUT_DIR)):
    print("   -", f)
print("=" * 60)

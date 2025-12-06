import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非互動模式，直接存檔
import os

# ====== File Paths ======
clinical_path = "BRCA.clin.merged.picked.txt"
mrna_path = "BRCA.medianexp.txt"
mirna_path = "BRCA-FFPE.miRseq_mature_RPM.txt"

# 輸出目錄
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Load Data ======
print("=" * 60)
print("🧬 BRCA mRNA/miRNA PCA Analysis Pipeline")
print("=" * 60)
print("\n📂 Loading data...")

# 讀取數據（第一列為基因名/特徵名，設為 index）
# 跳過可能的額外 header 行
clinical_df = pd.read_csv(clinical_path, sep="\t", index_col=0, engine="python")
mrna_df = pd.read_csv(mrna_path, sep="\t", index_col=0, engine="python", skiprows=[1])
mirna_df = pd.read_csv(mirna_path, sep="\t", index_col=0, engine="python", skiprows=[1])

# 確保數值轉換
mrna_df = mrna_df.apply(pd.to_numeric, errors='coerce')
mirna_df = mirna_df.apply(pd.to_numeric, errors='coerce')

print("✅ Files loaded successfully!")
print(f"   Clinical: {clinical_df.shape} (features x samples)")
print(f"   mRNA: {mrna_df.shape} (genes x samples)")
print(f"   miRNA: {mirna_df.shape} (miRNAs x samples)")

# 顯示數據預覽
print(f"\n📋 mRNA 前5個基因: {list(mrna_df.index[:5])}")
print(f"📋 mRNA 前5個樣本: {list(mrna_df.columns[:5])}")
print(f"📋 mRNA 數值欄位數: {mrna_df.select_dtypes(include=[np.number]).shape[1]}")

# ====== PCA Function ======
def pca_plot(df, title, output_name):
    """
    執行 PCA 並繪製散點圖
    
    Parameters:
    -----------
    df : DataFrame - 基因表達矩陣 (genes x samples)
    title : str - 圖表標題
    output_name : str - 輸出檔名
    """
    print(f"\n🔧 Running PCA for: {title} ...")
    
    # 轉置：變成 samples x genes
    df_T = df.T
    print(f"   原始維度: {df.shape} → 轉置後: {df_T.shape}")
    
    # 只保留數值欄位
    df_numeric = df_T.select_dtypes(include=[np.number])
    print(f"   數值欄位: {df_numeric.shape}")
    
    # 處理缺失值：填充為 0 或該基因的中位數
    df_filled = df_numeric.fillna(0)
    
    # 移除零變異的基因（常數列）
    variance = df_filled.var()
    non_zero_var = variance[variance > 0].index
    df_clean = df_filled[non_zero_var]
    print(f"   移除零變異後: {df_clean.shape}")
    
    if df_clean.shape[0] < 2 or df_clean.shape[1] < 2:
        print(f"   ⚠️ 數據不足，跳過 PCA")
        return None, None
    
    # 標準化
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # PCA
    n_components = min(2, df_scaled.shape[0], df_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(df_scaled)
    
    print(f"   ✅ PCA 完成!")
    print(f"   PC1 解釋變異: {pca.explained_variance_ratio_[0]*100:.2f}%")
    if n_components > 1:
        print(f"   PC2 解釋變異: {pca.explained_variance_ratio_[1]*100:.2f}%")
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1] if n_components > 1 else np.zeros(len(pcs)), 
                         c=range(len(pcs)), cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='white')
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if n_components > 1 else "PC2", 
                  fontsize=12, fontweight='bold')
    ax.set_title(f"{title} - PCA", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 添加樣本數量標註
    ax.text(0.02, 0.98, f'N = {len(pcs)} samples', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(scatter, label='Sample Index')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   📊 圖表已儲存: {output_path}")
    
    return pca, pcs


def plot_variance_explained(pca, title, output_name, n_show=20):
    """繪製變異解釋圖"""
    if pca is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_components = min(n_show, len(pca.explained_variance_ratio_))
    x = range(1, n_components + 1)
    
    # 個別變異
    ax1 = axes[0]
    ax1.bar(x, pca.explained_variance_ratio_[:n_components] * 100,
            color='#3498db', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Variance by Each PC', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 累積變異
    ax2 = axes[1]
    cumsum = np.cumsum(pca.explained_variance_ratio_[:n_components]) * 100
    ax2.plot(x, cumsum, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax2.fill_between(x, cumsum, alpha=0.3, color='#e74c3c')
    ax2.axhline(80, color='#2ecc71', linestyle='--', linewidth=2, label='80%')
    ax2.axhline(95, color='#f39c12', linestyle='--', linewidth=2, label='95%')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Cumulative Variance', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   📊 變異圖已儲存: {output_path}")


# ====== Run PCA ======
print("\n" + "=" * 60)
print("📊 執行 PCA 分析")
print("=" * 60)

# mRNA PCA
pca_mrna, pcs_mrna = pca_plot(mrna_df, "BRCA mRNA Expression", "pca_mrna.png")
if pca_mrna is not None:
    # 執行更多 PC 的 PCA 來繪製變異圖
    df_T = mrna_df.T.select_dtypes(include=[np.number]).fillna(0)
    variance = df_T.var()
    df_clean = df_T[variance[variance > 0].index]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    pca_full = PCA(n_components=min(50, df_scaled.shape[0], df_scaled.shape[1]))
    pca_full.fit(df_scaled)
    plot_variance_explained(pca_full, "mRNA Expression", "variance_mrna.png")

# miRNA PCA
pca_mirna, pcs_mirna = pca_plot(mirna_df, "BRCA miRNA Expression", "pca_mirna.png")
if pca_mirna is not None:
    df_T = mirna_df.T.select_dtypes(include=[np.number]).fillna(0)
    variance = df_T.var()
    df_clean = df_T[variance[variance > 0].index]
    if df_clean.shape[0] > 2 and df_clean.shape[1] > 2:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_clean)
        pca_full = PCA(n_components=min(20, df_scaled.shape[0], df_scaled.shape[1]))
        pca_full.fit(df_scaled)
        plot_variance_explained(pca_full, "miRNA Expression", "variance_mirna.png")

# ====== Summary ======
print("\n" + "=" * 60)
print("🎉 分析完成!")
print("=" * 60)
print(f"\n📁 所有結果已儲存至 '{OUTPUT_DIR}' 資料夾:")
for f in os.listdir(OUTPUT_DIR):
    print(f"   - {f}")
print("\n✅ Demo completed.")

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ====== 使用者輸入路徑 ======
clinical_path = r"C:\Users\User\Downloads\MRNAPC~1\data\GDACBR~4.0\GDACBR~1.0\BRCA-F~1.TXT"
mrna_path     = r"C:\Users\User\Downloads\mRNA PCA\data\gdac.broadinstitute.org_BRCA.mRNA_Preprocess_Median.Level_3.2016012800.0.0\gdac.broadinstitute.org_BRCA.mRNA_Preprocess_Median.Level_3.2016012800.0.0\BRCA.medianexp.txt"
mirna_path    = r"C:\Users\User\Downloads\mRNA PCA\data\gdac.broadinstitute.org_BRCA-FFPE.miRseq_Mature_Preprocess.Level_3.2016012800.0.0\gdac.broadinstitute.org_BRCA-FFPE.miRseq_Mature_Preprocess.Level_3.2016012800.0.0\BRCA-FFPE.miRseq_mature_RPM.txt"

# ====== 載入資料 ======
def load_expression(path):
    df = pd.read_csv(path, sep="\t")
    df.rename(columns={df.columns[0]: "gene"}, inplace=True)
    df.set_index("gene", inplace=True)
    df.columns = [c.split(".")[0] for c in df.columns]  # 去掉 .01A 尾碼
    return df

clinical = pd.read_csv(clinical_path, sep="\t")
mrna = load_expression(mrna_path)
mirna = load_expression(mirna_path)

# ====== 匹配樣本 ID ======
samples = list(set(mrna.columns) & set(mirna.columns))
mrna = mrna[samples]
mirna = mirna[samples]

# ====== 組合 mRNA + miRNA ======
combined = pd.concat([mrna, mirna], axis=0)

# ====== PCA ======
X = combined.T  # samples x genes
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pc = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pc[:,0],
    "PC2": pc[:,1],
    "sample": X.index
})

# ====== 簡單 KMeans clustering ======
labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
pca_df["cluster"] = labels

# ====== 繪圖 ======
plt.figure(figsize=(6,5))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["cluster"], cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BRCA mRNA + miRNA PCA")
plt.savefig("brca_pca.png", dpi=300)

# ====== 匯出結果 ======
combined.T.to_csv("BRCA_combined_expression.csv")
clinical.to_csv("BRCA_clinical_cleaned.csv", index=False)
pca_df.to_csv("BRCA_pca_clusters.csv", index=False)

print("Done! Files generated:")
print(" - brca_pca.png")
print(" - BRCA_combined_expression.csv")
print(" - BRCA_clinical_cleaned.csv")
print(" - BRCA_pca_clusters.csv")

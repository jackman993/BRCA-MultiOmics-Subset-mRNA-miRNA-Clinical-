# BRCA-Multi: A Mini Multi-Omics Benchmark for Breast Cancer Analysis

This repository contains a compact, high-quality multi-omics benchmark dataset for BRCA (Breast Invasive Carcinoma), integrating:

- **mRNA expression (median normalized)**
- **miRNA mature expression (RPM)**
- **Tier 1 clinical metadata**

The dataset is constructed from publicly released TCGA GDAC files and aligned into a clean, analysis-ready framework suitable for:

- Dimensionality reduction (PCA / UMAP)
- Clustering (KMeans / HDBSCAN)
- Multi-omics integration research
- Model benchmarking
- Methodological prototyping (ML, DL, bioinformatics)

---

## 🔧 Pipeline Overview

The repository includes a complete executable pipeline:
brca_demo_pipeline.py


This script:

1. Loads mRNA, miRNA, and clinical datasets  
2. Cleans sample identifiers and aligns shared samples  
3. Performs feature fusion (mRNA + miRNA)  
4. Applies PCA for dimensionality reduction  
5. Runs KMeans clustering  
6. Saves all processed outputs and visualization plots

---

## 📁 Output Files

Running the pipeline produces:

- `BRCA_combined_expression.csv` — merged mRNA + miRNA feature matrix  
- `BRCA_clinical_cleaned.csv` — cleaned clinical metadata  
- `BRCA_pca_clusters.csv` — PCA embeddings + cluster labels  
- `brca_pca.png` — PCA plot (ready for publication or documentation)

---

## 🧬 Data Source

All input files originate from the TCGA GDAC Firehose public repository (2016-01-28 release).  
Only processed, non-identifiable files were used.

---

## 🚀 Usage

Run:

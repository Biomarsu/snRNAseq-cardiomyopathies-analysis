import argparse
import os
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Define command-line arguments
parser = argparse.ArgumentParser(description='snRNA-seq analysis pipeline')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
parser.add_argument('--prefix_nf', type=str, default="0", help='Prefix for non-failing samples')
parser.add_argument('--prefix_dcm', type=str, default="1", help='Prefix for DCM samples')
args = parser.parse_args()

# Output_dir to local variables
file_path = args.input_file
output_dir = args.output_dir
nf_prefix = args.prefix_nf
dcm_prefix = args.dcm_prefix

# Define output file paths
output_results_path_by_p = os.path.join(output_dir, "results_top200_byP.csv")
output_results_path_by_fc = os.path.join(output_dir, "results_top200_byFC.csv")
plots_dir = os.path.join(output_dir, "plots")

# Create the directories for output and plots
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load the raw data
data = pd.read_csv(file_path, index_col=0)
print(f"Raw data loaded. Shape: {data.shape}")

# Separate NF and DCM sample columns - ADDED VARIABLES
#nf_prefix = "0"  # Prefix for non-failing samples - REMOVED
#dcm_prefix = "1"  # Prefix for DCM samples - REMOVED
nf_samples = [col for col in data.columns if col.startswith(nf_prefix)]
dcm_samples = [col for col in data.columns if col.startswith(dcm_prefix)]

# Ensure both groups have the same number of columns for pairing
assert len(nf_samples) == len(dcm_samples), "Sample lists must be of equal length for paired test."

# Filter out dropouts (genes with zero expression across all samples)
non_zero_genes = (data > 0).sum(axis=1) > 0
data = data.loc[non_zero_genes]
print(f"After removing dropouts: {data.shape[0]} genes remain.")

# Convert to AnnData for Scanpy analysis
adata = sc.AnnData(data.T)
adata.obs['condition'] = ['NF' if col.startswith(nf_prefix) else 'DCM' for col in data.columns]

# Normalize the data (no log transformation)
sc.pp.normalize_total(adata, target_sum=1e4)
print("Normalization completed.")

# PCA, UMAP, and Clustering
# Step 1: PCA
sc.tl.pca(adata, svd_solver='arpack')
plt.figure(figsize=(12, 10))
sc.pl.pca(adata, color='condition', title='PCA Plot', show=False)
plt.savefig(os.path.join(plots_dir, "PCA_Plot.png"), dpi=120)
plt.close()

# Step 2: UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata)
plt.figure(figsize=(12, 10))
sc.pl.umap(adata, color='condition', title='UMAP Plot', show=False)
plt.savefig(os.path.join(plots_dir, "UMAP_Plot.png"), dpi=120)
plt.close()

# Step 3: Clustering
sc.tl.leiden(adata, resolution=0.5)
adata.obs['leiden'] = adata.obs['leiden'].astype('category')
plt.figure(figsize=(12, 10))
sc.pl.umap(adata, color=['leiden'], title='UMAP with Leiden Clusters', show=False)
plt.savefig(os.path.join(plots_dir, "UMAP_with_Leiden_Clusters.png"), dpi=120)
plt.close()

# Differential Expression Analysis
results = []
for gene in data.index:
    nf_values = data.loc[gene, nf_samples].values
    dcm_values = data.loc[gene, dcm_samples].values

    if len(nf_values) == len(dcm_values) and not np.all(nf_values == dcm_values):
        stat, p_value = wilcoxon(nf_values, dcm_values, zero_method='zsplit', alternative='two-sided')
        log2_fold_change = np.log2(np.mean(dcm_values) + 1) - np.log2(np.mean(nf_values) + 1)
    else:
        p_value, log2_fold_change = None, None

    results.append({
        'gene': gene,
        'adjusted_p_value': p_value,
        'log2_fold_change': log2_fold_change
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.dropna(inplace=True)

# FDR adjustment
results_df['adjusted_p_value'] = multipletests(results_df['adjusted_p_value'], method='fdr_bh')[1]

# Add absolute log2 fold change
results_df['abs_log2_fold_change'] = results_df['log2_fold_change'].abs()

# Rank genes by adjusted p-value
results_df['rank'] = results_df['adjusted_p_value'].rank(method='min', ascending=True)

# Save top 200 genes ranked by adjusted p-value
top_200_by_p = results_df.nsmallest(200, 'adjusted_p_value')[['gene', 'adjusted_p_value', 'log2_fold_change', 'rank']]
top_200_by_p['abs_log2_fold_change'] = top_200_by_p['log2_fold_change'].abs()  # Add this column explicitly
top_200_by_p.to_csv(output_results_path_by_p, index=False)
print(f"Top 200 genes by adjusted p-value saved to {output_results_path_by_p}")

# Save top 200 genes ranked by absolute log2 fold change
top_200_by_fc = top_200_by_p.copy()
top_200_by_fc['rank_by_fc'] = top_200_by_fc['abs_log2_fold_change'].rank(method='min', ascending=False)
top_200_by_fc.sort_values('rank_by_fc', inplace=True)  # Sort by rank value
top_200_by_fc = top_200_by_fc[['gene', 'adjusted_p_value', 'log2_fold_change', 'rank_by_fc']]
top_200_by_fc.to_csv(output_results_path_by_fc, index=False)
print(f"Top 200 genes by absolute log2 fold change saved to {output_results_path_by_fc}")

# Visualization - Volcano Plot for Top 200 Genes
plt.figure(figsize=(12, 10))
colors = ['red' if fc > 1 else 'blue' if fc < -1 else 'gray' for fc in top_200_by_p['log2_fold_change']]
plt.scatter(top_200_by_p['log2_fold_change'], -np.log10(top_200_by_p['adjusted_p_value']),
            c=colors, alpha=0.75)
plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='Significance Threshold (p=0.05)')
plt.axvline(1, color='green', linestyle='--', label='Log2 Fold Change > 1')
plt.axvline(-1, color='green', linestyle='--', label='Log2 Fold Change < -1')
plt.xlabel('Log2 Fold Change', fontsize=14)
plt.ylabel('-Log10 Adjusted P-Value', fontsize=14)
plt.title('Volcano Plot', fontsize=16)
plt.legend(fontsize=12)
plt.savefig(os.path.join(plots_dir, "Volcano_Plot.png"), dpi=120)
plt.close()

# Heatmap for Top 20 Genes by Adjusted P-value
top_20_genes = top_200_by_p.nsmallest(20, 'rank')['gene']
heatmap_data = adata[:, top_20_genes].X  # Extract expression data for top genes
heatmap_data_df = pd.DataFrame(heatmap_data, columns=top_20_genes, index=adata.obs['condition'])

# Split data by NF and DCM
nf_data = heatmap_data_df[heatmap_data_df.index == 'NF']
dcm_data = heatmap_data_df[heatmap_data_df.index == 'DCM']

# Combine data for NF and DCM horizontally
combined_heatmap_data = pd.concat([nf_data, dcm_data], axis=0)

# Transpose for vertical gene labels
combined_heatmap_data = combined_heatmap_data.T

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(combined_heatmap_data, cmap="vlag", xticklabels=True, yticklabels=True, cbar=True)
plt.title('Heatmap of Top 20 Genes', fontsize=16)
plt.xlabel('Samples (NF | DCM)', fontsize=14)
plt.ylabel('Genes', fontsize=14)
plt.axvline(len(nf_data), color='black', linestyle='--', lw=1.5, label='Condition Separator')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12)
plt.savefig(os.path.join(plots_dir, "Heatmap_Top_20.png"), dpi=120)
plt.close()
print("All plots saved.")
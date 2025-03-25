# Single-Nucleus RNA Sequencing Analysis of Cardiomyopathies - Amami 2025

This repository contains the Python code used in the thesis project titled: **"Single-Nucleus RNA Sequencing Analysis of Heart Cells in Dilated and Hypertrophic Cardiomyopathies: Insights into Heart Failure"**. This thesis was submitted on February 10, 2025, to the Faculty of Life Sciences at the Rhein-Waal University of Applied Sciences (HSRW). This work was done in collaboration with the bioinformatics department at the Julius-Maximilians-Universität Würzburg (JMUW).

**Author:** Ahmed Belhssan AMAMI

**Supervisors:**

*   Prof. Dr. rer. nat. habil. Mònica Palmada Fenés (HSRW)
*   Prof. Dr. Thomas Dandekar (JMUW)
*   Dr. Aylin Caliskan (JMUW)
*   Samantha A. W. Crouch (JMUW)

## Data Source

The data originally derives from comprehensive snRNA-seq analyses of heart samples, specifically targeting the molecular landscape of cardiomyopathies. The datasets were initially collected as part of a study by:

*   Chaffin et al. (2022). Single-nucleus profiling of human dilated and hypertrophic cardiomyopathy. *Nature, 608*(7921).

A subset of the Chaffin et al. (2022) data was used in this thesis. Specifically, a set of randomized 1000 cells by 1000 cells CSV file versions of the raw snRNA-seq data was provided by Dr. Aylin Caliskan from the Bioinformatics Department at Würzburg University. The preprocessing involved randomizing the data to construct three sets of pre-processed snRNA-seq count matrices representing the following condition comparisons:

*   Non-failing hearts (NF) vs. Dilated Cardiomyopathy (DCM)
*   Non-failing hearts (NF) vs. Hypertrophic Cardiomyopathy (HCM)
*   Dilated Cardiomyopathy (DCM) vs. Hypertrophic Cardiomyopathy (HCM)

The selection of these pairwise comparisons reflects the research objectives of this thesis, namely to identify DEGs and transcriptional pathways relevant to cardiomyopathies. Subsequently, these results were compared to the Principal Feature Analysis (PFA) results (Caliskan et al., 2023) for the top DEGs for each comparison, which were also provided by Dr. Aylin Caliskan.

## Code Description

The Python script(s) in this repository take a CSV file (count matrix) as input and produce the following outputs:

*   Five plots: PCA, UMAP, UMAP with Leiden clustering, Volcano plot, and Heatmap
*   Two CSV files: Top differentially expressed genes (DEGs), ranked by adjusted p-value and fold change.

The workflow implemented in the scripts is as follows:

1.  Filter dropouts (genes with zero expression across all cells)
2.  Normalization (using Scanpy)
3.  Dimensionality reduction: UMAP and PCA
4.  Leiden clustering
5.  Differential expression analysis: Wilcoxon rank-sum test with FDR adjustment of p-values
6.  Log2 fold-change (FC) calculation
7.  Filtering: Select genes with adjusted p-value < 0.05
8.  Saving: Save top 200 DEGs ranked by adjusted p-value and fold change to separate CSV files.
9.  Visualization: Volcano plot and heatmap of top DEGs.

## Aim of the Work (For context, directly from the thesis work)

This thesis aims to investigate the transcriptional changes underlying cardiomyopathies, specifically focusing on dilated cardiomyopathy and hypertrophic cardiomyopathy, using single-nucleus RNA sequencing data. These transcriptional alterations are crucial to understand as they directly reflect the cellular responses and molecular mechanisms that drive the pathogenesis and progression of these complex cardiac diseases. This work focuses on validating findings generated by Principal Feature Analysis (Caliskan et al., 2023) by identifying differentially expressed genes and rigorously comparing transcriptional profiles across three critical conditions: non-failing hearts versus DCM, NF versus HCM, and DCM versus HCM.

The analysis involves a multi-step approach, beginning with the preprocessing of snRNA-seq data, followed by dimensionality reduction, cell clustering, and the identification of differentially expressed genes (DEGs). These DEGs are then subjected to pathway enrichment analysis to elucidate the biological processes driving the different phenotypes.

Subsequently, these identified DEGs are compared against the PFA-derived genes and existing results from the original study by Chaffin et al. (2022) to not only validate the PFA results but also gain deeper insights into the similarities and differences in gene expression between non-failing hearts, DCM, and HCM.

This approach could provide a better understanding of how cell-specific gene expression patterns differ across disease states and identify key molecular signatures that could serve as potential targets for novel therapeutic interventions. It is hypothesized that the three conditions will have significantly different gene expression profiles, reflecting their different molecular mechanisms of disease.

## Dependencies and Usage

### 📌 Required Libraries
Ensure you have the following Python packages installed:

```bash
pip install pandas numpy scanpy scipy statsmodels matplotlib seaborn
```
or using Conda:
```
conda install -c conda-forge scanpy pandas numpy scipy statsmodels matplotlib seaborn
```
📂 Usage
To run the analysis, execute the Python script in your local environment or Jupyter Notebook:

python script.py

## References

*   Caliskan, A., Rasbach, L., Yu, W., et al. (2023). Optimized cell type signatures revealed from single-cell data by combining principal feature analysis, mutual information, and machine learning. *Computational and Structural Biotechnology Journal, 21*, 3293-3314.
*   Chaffin, M., Papangeli, I., Simonson, B., et al. (2022). Single-nucleus profiling of human dilated and hypertrophic cardiomyopathy. *Nature, 608*(7921).

## License

This repository is released under the **MIT License**.  
Feel free to use, modify, and distribute the code with attribution.

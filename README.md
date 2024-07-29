# Integrative, high-resolution analysis of single cells across experimental conditions with PARAFAC2

[![codecov](https://codecov.io/gh/meyer-lab/parafac2/branch/main/graph/badge.svg?token=srqQtzqc6V)](https://codecov.io/gh/meyer-lab/parafac2)

`parafac2` contains the code for the PARAFAC2 (Pf2) python package, a tensor decomposition technique, used in our study for identifying variation patterns in single-cell populations across conditions. In our study, we discovered association patterns to specific cell populations, genes, and experimental conditions in both a drug perturbational study and systemic lupus erythematosus cohort study. 

## Installation
To add it to your Python package, add the following line to requirements.txt and remake venv: `git+https://github.com/meyer-lab/parafac2.git@main`

## Input Requirements
1. Your AnnData object must include an observations `column condition_unique_idxs` that is a 0-indexed array of which condition each cell is derived from along with the cell barcode 
Preprocessing your data
2. Your AnnData object must be preprocessed (removed doublets, normalized, log transformed) before running the algorithm
3. The function `parafac2_nd` is the Pf2 algorithm with various parameters that can be altered such as rank, tolerance, etc. 

## Outputs
The output of `parafac2_nd` is the first AnnData object and the reconstruction error (R2X). The results of `parafac2_nd` are added to the AnnData object. These include: 
1. The weights for each component `X.uns["Pf2_weights"]`
2. The factors with respect to each dimension in the data where `X.uns[“Pf2_A”]` is the condition factors, `X.uns[“Pf2_B”]` is the eigen-state factors, and `X.varm[“Pf2_C”]` is the genes, where the width of the matrix is the rank used for the algorithm
3. Each cell will have the corresponding values for the projections, `X.obsm["projections"]`, where the width of the matrix is the rank used for the algorithm
4. In addition, each cell has the corresponding weighted projections for each cell in the `X.obsm["weighted_projections"]` for all components, to determine how each cell related to each component pattern, where the width of the matrix is the rank used for the algorithm
5. We recommend implementing an embedding algorithm such as PaCMAP or UMAP on the `X.obsm["projections"]` to visualize cell-to-cell heterogeneity, creating a new columns coined `X.obsm["embedding"]` for example

## Examples
You can find example scripts that load single-cell scRNA-seq data across conditions, implement Pf2, and various ways to interpret and plot Pf2 on the scCP repository (Basic familiarity with the python programming languages is recommended to navigate repository).

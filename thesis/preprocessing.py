from anndata import AnnData
import scanpy as sc

def _generic_preprocessing(adata: AnnData) -> AnnData:
    # parameters for the filtering inspired by CPA's strategy for sciplex2
    sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_cells(adata, min_genes=720)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_total(adata) 
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    return adata[:,adata.var.highly_variable]

def _generic_preprocessing_no_filtering(adata: AnnData) -> AnnData:
    # source: https://github.com/BhattacharyaLab/scVIDR/blob/main/vidr/utils.py
    # scvidr use this preprocessing for the TCDD and sciplex3
    sc.pp.normalize_total(adata) 
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    return adata[:,adata.var.highly_variable]


def preprocess_sciplex3(adata: AnnData) -> AnnData:
    return _generic_preprocessing(adata)

def preprocess_sciplex3_no_filtering(adata: AnnData) -> AnnData:
    return _generic_preprocessing_no_filtering(adata)


def preprocess_nault(adata: AnnData) -> AnnData:
    return _generic_preprocessing(adata)


def preprocess_nault_no_filtering(adata: AnnData) -> AnnData:
    return _generic_preprocessing_no_filtering(adata)


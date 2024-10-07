from anndata import AnnData
import scanpy as sc

def preprocess_sciplex3(adata: AnnData) -> AnnData:
    sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_cells(adata, min_genes=720)
    sc.pp.filter_genes(adata, min_cells=100)
    
    # should this be per cell like https://github.com/facebookresearch/CPA/blob/main/preprocessing/sciplex3.ipynb? No, it is deprecated
    sc.pp.normalize_total(adata) 
    
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    return adata[:,adata.var.highly_variable]
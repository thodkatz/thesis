from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import scanpy as sc
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse
import os

@dataclass(frozen=True)
class Evaluation:
    mean: Optional[NDArray[np.float32]]
    mean_expressed: NDArray[np.float32]
    mean_degs: NDArray[np.float32]
    fractions: NDArray[np.float32]
    fractions_degs: NDArray[np.float32]
    genes: List[str]
    degs: List[str]
    
    @classmethod
    def from_adata(cls, adata: AnnData, degs: List[str]) -> Evaluation:
        """
        degs: Expected to be sorted, first the most differential expressed gene
        """
        degs = degs[:100]
        
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        assert not np.isnan(X).any()
        
        deg_indices = [adata.var_names.get_loc(gene) for gene in degs if gene in adata.var_names]
        deg_expression = X[:, deg_indices]
        
        
        fraction, mean, mean_expressed = cls._from_array(X)
        fraction_degs, _, mean_degs = cls._from_array(deg_expression)
        
        return Evaluation(
            mean=mean,
            mean_expressed=mean_expressed,
            mean_degs=mean_degs,
            fractions=fraction,
            fractions_degs=fraction_degs,
            genes=adata.var_names.values.tolist(),
            degs=degs
        )
        
    @staticmethod
    def _from_array(array: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        fraction = (array > 0).mean(axis=0)
        mean = np.mean(array, axis=0)
        mean_expressed = np.where(array > 0, array, np.nan)
        mean_expressed = np.nanmean(mean_expressed, axis=0)
        return fraction, mean, mean_expressed
        

    @staticmethod
    def compare(reference: Evaluation, target: Evaluation) -> Evaluation:
        assert np.all(reference.genes == target.genes)
        assert np.all(reference.degs == target.degs)
        
        def relative(reference, target):
            return np.abs(reference - target) / reference
        
        return Evaluation(
            mean=None, # avoid dividing by zero
            mean_expressed=relative(reference.mean_expressed , target.mean_expressed),
            fractions=np.abs(reference.fractions - target.fractions),
            fractions_degs=np.abs(reference.fractions_degs - target.fractions_degs),
            mean_degs=relative(reference.mean_degs,  target.mean_degs),
            genes=reference.genes,
            degs=reference.degs
        )
        
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        
        df = pd.DataFrame({
            'gene': self.genes,
            'mean': self.mean,
            'mean_expressed': self.mean_expressed,
            'fraction': self.fractions,
        })
        
        df_degs = pd.DataFrame({
            'deg': self.degs,
            'mean_degs': self.mean_degs,
            'fraction_degs': self.fractions_degs
        })
        
        print("Evalution saved", path)
        
        df.to_csv(f"{path}/metrics_per_gene.csv", index=False)
        df_degs.to_csv(f"{path}/metrics_per_deg.csv", index=False)
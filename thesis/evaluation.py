from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import scanpy as sc
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse

@dataclass(frozen=True)
class Evaluation:
    mean: NDArray[np.float32]
    mean_expressed: NDArray[np.float32]
    mean_degs: NDArray[np.float32]
    fractions: NDArray[np.float32]
    fractions_degs: NDArray[np.float32]
    genes: List[str]
    degs: List[str]
    
    @classmethod
    def from_adata(cls, adata: AnnData, degs: List[str]) -> Evaluation:
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
        mean_expressed = np.where(array > 0, array, 0)
        mean_expressed = np.mean(mean_expressed, axis=0)
        return fraction, mean, mean_expressed
        

    @classmethod
    def diff(cls, eval1: Evaluation, eval2: Evaluation) -> Evaluation:
        assert np.all(eval1.genes == eval2.genes)
        assert np.all(eval1.degs == eval2.degs)
        
        return Evaluation(
            mean=np.abs((eval1.mean - eval2.mean)),
            mean_expressed=np.abs(eval1.mean_expressed - eval2.mean_expressed),
            fractions=np.abs(eval1.fractions - eval2.fractions),
            fractions_degs=np.abs(eval1.fractions_degs - eval2.fractions_degs),
            mean_degs=np.abs(eval1.mean_degs - eval2.mean_degs),
            genes=eval1.genes,
            degs=eval1.degs
        )
        
    def save(self, path: str) -> None:
        df = pd.DataFrame({
            'gene': self.genes,
            'mean': self.mean,
            'mean_expressed': self.mean_expressed,
            'fraction': self.fractions,
        })
        
        df_degs = pd.DataFrame({
            'deg': self.degs,
            'mean': self.mean_degs,
            'fraction': self.fractions_degs
        })
        
        print("Evalution saved", path)
        
        df.to_csv(f"{path}/metrics_per_gene.csv", index=False)
        df_degs.to_csv(f"{path}/metrics_per_deg.csv", index=False)
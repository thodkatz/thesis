from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
import scanpy as sc
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse
import os
from torch import Tensor
from zipp import Path

from scButterfly.model_utlis import tensor2adata, draw_reg_plot
from scButterfly.model_utlis import get_pearson2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Metrics:
    mean: Optional[NDArray[np.float32]]
    mean_expressed: NDArray[np.float32]
    mean_degs: NDArray[np.float32]
    fractions: NDArray[np.float32]
    fractions_degs: NDArray[np.float32]
    genes: List[str]
    degs: List[str]

    @classmethod
    def from_adata(cls, adata: AnnData, degs: List[str]) -> Metrics:
        """
        degs: Expected to be sorted, first the most differential expressed gene
        """
        degs = degs[:100]

        X = adata.X.toarray() if issparse(adata.X) else adata.X
        assert not np.isnan(X).any()

        deg_indices = [
            adata.var_names.get_loc(gene) for gene in degs if gene in adata.var_names
        ]
        deg_expression = X[:, deg_indices]

        fraction, mean, mean_expressed = cls._from_array(X)
        fraction_degs, _, mean_degs = cls._from_array(deg_expression)

        return Metrics(
            mean=mean,
            mean_expressed=mean_expressed,
            mean_degs=mean_degs,
            fractions=fraction,
            fractions_degs=fraction_degs,
            genes=adata.var_names.values.tolist(),
            degs=degs,
        )

    @staticmethod
    def _from_array(
        array: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        fraction = (array > 0).mean(axis=0)
        mean = np.mean(array, axis=0)
        mean_expressed = np.where(array > 0, array, np.nan)
        mean_expressed = np.nanmean(mean_expressed, axis=0)
        return fraction, mean, mean_expressed

    @staticmethod
    def compare(reference: Metrics, target: Metrics) -> Metrics:
        assert np.all(reference.genes == target.genes)
        assert np.all(reference.degs == target.degs)

        def relative(reference, target):
            return np.abs(reference - target) / reference

        return Metrics(
            mean=None,  # avoid dividing by zero
            mean_expressed=relative(reference.mean_expressed, target.mean_expressed),
            fractions=np.abs(reference.fractions - target.fractions),
            fractions_degs=np.abs(reference.fractions_degs - target.fractions_degs),
            mean_degs=relative(reference.mean_degs, target.mean_degs),
            genes=reference.genes,
            degs=reference.degs,
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        df = pd.DataFrame(
            {
                "gene": self.genes,
                "mean": self.mean,
                "mean_expressed": self.mean_expressed,
                "fraction": self.fractions,
            }
        )

        df_degs = pd.DataFrame(
            {
                "deg": self.degs,
                "mean_degs": self.mean_degs,
                "fraction_degs": self.fractions_degs,
            }
        )

        print("Evalution saved", path)

        df.to_csv(f"{path}/metrics_per_gene.csv", index=False)
        df_degs.to_csv(f"{path}/metrics_per_deg.csv", index=False)


def evaluation(
    input: AnnData, ground_truth: AnnData, predicted: Union[List[Tensor], AnnData], output_path: Path
):
    if isinstance(predicted, list) and isinstance(predicted[0], Tensor):
        predicted = tensor2adata(predicted)
    elif isinstance(predicted, AnnData):
        pass
    else:
        raise ValueError("predicted must be a list of tensors or anndata")
    assert input.shape == ground_truth.shape == predicted.shape
    
    predicted.obs['condition'] = 'pred'
    input.obs['condition'] = 'control'
    ground_truth.obs['condition'] = 'control'

    eval_adata = sc.AnnData.concatenate(input, ground_truth, predicted)
    
    fig_list = []
    
    df_metrics = pd.DataFrame()
    
    target_type = predicted.obs["celltype"][0]
    df_metrics['cell_type_test'] = target_type
    
    """ PCA """
    sc.tl.pca(eval_adata)
    fig = sc.pl.pca(eval_adata, color="condition", frameon=False, title="PCA of "+target_type+" by Condition", return_fig=True)
    fig_list.append(fig)
    
    """ DEGs """
    sc.tl.rank_genes_groups(eval_adata, groupby="condition", reference="control", method="t-test", show=False)
    degs_sti = eval_adata.uns["rank_genes_groups"]["names"]["stimulated"]
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"]["pred"]

    result = eval_adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    common_degs = list(set(degs_sti[0:100])&set(degs_pred[0:100]))
    common_nums = len(common_degs)

    sc.pl.rank_genes_groups(eval_adata, n_genes=30, sharey=False, show=False, save=True)


    df_metrics['DEGs'] = common_nums
    
    
    """ R2"""
    r2mean, r2mean_top100, fig = draw_reg_plot(eval_adata=eval_adata,
            cell_type="target_type",
            reg_type='mean',
            axis_keys={"x": "pred", "y": "stimulated"},
            condition_key='condition',
            gene_draw=degs_sti[:10],
            top_gene_list=degs_sti[:100],
            save_path=None,
            title=None,
            show=False,
            fontsize=12
            )
    fig_list.append(fig)

    
    df_metrics['r2mean'] = r2mean
    df_metrics['r2mean_top100'] = r2mean_top100
    
    df = get_pearson2(eval_adata, key_dic={'condition_key': 'condition',
        'cell_type_key': 'cell_type',
        'ctrl_key': 'control',
        'stim_key': 'stimulated',
        'pred_key': 'pred',
        }, n_degs=100, sample_ratio=0.8, times=100)
    df.to_csv(output_path / 'sampled_r2.csv', index=False)

    """ dotplot """
    marker_genes = degs_sti[:20]
    sc.pl.dotplot(eval_adata, marker_genes, groupby='condition', save='.pdf', show=False)     
    
    """ save to pdf """
    with PdfPages(output_path / 'evaluation.pdf') as pdf:
        for i in range(len(fig_list)):
            pdf.savefig(figure=fig_list[i], dpi=200, bbox_inches='tight')
            plt.close()

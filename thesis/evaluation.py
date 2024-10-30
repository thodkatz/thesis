from __future__ import annotations
from dataclasses import dataclass
from pyexpat import model
from typing import List, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
import scanpy as sc
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse
import os
from torch import Tensor
from pathlib import Path

from scButterfly.model_utlis import tensor2adata
from scButterfly.model_utlis import get_pearson2
from scButterfly.draw_cluster import draw_reg_plot
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from thesis.utils import FileModelUtils, append_csv
from thesis import METRICS_PATH
import anndata as ad


@dataclass(frozen=True)
class MetricsPerGene:
    mean: Optional[NDArray[np.float32]]
    mean_expressed: NDArray[np.float32]
    fractions: NDArray[np.float32]  # fraction of cells that express a gene
    genes: List[str]
    degs_ordered: List[str]

    @classmethod
    def from_adata(cls, adata: AnnData, degs: List[str]) -> MetricsPerGene:
        """
        degs: Expected to be sorted, first the most differential expressed gene
        """
        genes = adata.var_names.values.tolist()
        assert set(genes) == set(degs)

        X = adata.X.toarray() if issparse(adata.X) else adata.X
        assert not np.isnan(X).any()

        fraction, mean, mean_expressed = cls._from_array(X)

        return MetricsPerGene(
            mean=mean,
            mean_expressed=mean_expressed,
            fractions=fraction,
            genes=genes,
            degs_ordered=degs,
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
    def compare(reference: MetricsPerGene, target: MetricsPerGene) -> MetricsPerGene:
        assert np.all(reference.genes == target.genes)
        assert np.all(reference.degs_ordered == target.degs_ordered)

        def relative(reference, target):
            return np.abs(reference - target) / reference

        return MetricsPerGene(
            mean=None,  # avoid dividing by zero
            mean_expressed=relative(reference.mean_expressed, target.mean_expressed),
            fractions=np.abs(reference.fractions - target.fractions),
            genes=reference.genes,
            degs_ordered=reference.degs_ordered,
        )

    def get_mean_degs(self, num_degs: int):
        degs = self.degs_ordered[:num_degs]
        return list(
            {
                gene_name: value
                for gene_name, value in zip(self.genes, self.mean_expressed)
                if gene_name in degs
            }.values()
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        df = pd.DataFrame(
            {
                "gene": self.genes,
                "deg": [
                    self.degs_ordered.index(gene) for gene in self.genes
                ],  # I know this is slow
                "mean": self.mean,
                "mean_expressed": self.mean_expressed,
                "fraction": self.fractions,
            }
        )

        print("Evalution saved", path)

        df.to_csv(f"{path}/metrics_per_gene.csv", index=False)


def evaluation_out_of_sample(
    model_config: FileModelUtils,
    input: AnnData,
    ground_truth: AnnData,
    predicted: Union[List[Tensor], AnnData],
    output_path: Path,
    append_metrics: bool = True,
    save_plots: bool = True,
):
    os.makedirs(output_path, exist_ok=True)

    if isinstance(predicted, list) and isinstance(predicted[0], Tensor):
        predicted = tensor2adata(predicted, input.var, input.obs)
    elif isinstance(predicted, AnnData):
        pass
    else:
        raise ValueError("predicted must be a list of tensors or anndata")
    assert input.shape == predicted.shape

    assert len(ground_truth.shape) == len(input.shape) == 2
    n_cells_input = input.shape[0]
    n_cells_stimulated = ground_truth.shape[0]
    n_genes = input.shape[1]
    assert n_genes == ground_truth.shape[1]
    assert n_cells_input > 0
    assert n_cells_stimulated > 0

    predicted_cell_types = predicted.obs[model_config.cell_type_key].unique()
    ground_truth_cell_types = ground_truth.obs[model_config.cell_type_key].unique()
    control_cell_types = input.obs[model_config.cell_type_key].unique()
    assert all(predicted_cell_types == ground_truth_cell_types)
    assert all(predicted_cell_types == control_cell_types)
    target_type = predicted.obs[model_config.cell_type_key][0]

    predicted.obs["condition"] = "pred"
    input.obs["condition"] = "control"
    ground_truth.obs["condition"] = "stimulated"

    eval_adata = ad.concat([input, ground_truth, predicted])

    fig_list = []

    # cluster metrics?
    # sc.pp.pca(predicted) # should I use this
    # sc.pp.neighbors(predicted)

    """ PCA """
    if save_plots:
        sc.tl.pca(eval_adata)
        fig_condition = sc.pl.pca(
            eval_adata,
            color="condition",
            frameon=False,
            title=f"PCA of  {target_type} by condition",
            return_fig=True,
            show=False,
        )
        fig_list.append(fig_condition)

    """ DEGs """
    sc.tl.rank_genes_groups(
        eval_adata,
        groupby="condition",
        reference="control",
        method="t-test",
        show=False,
    )
    degs_sti = eval_adata.uns["rank_genes_groups"]["names"]["stimulated"]
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"]["pred"]

    result = eval_adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    common_degs = list(set(degs_sti[0:100]) & set(degs_pred[0:100]))
    common_nums = len(common_degs)

    """ R2"""
    r2mean, r2mean_top20, r2mean_top100, fig = draw_reg_plot(
        eval_adata=eval_adata,
        cell_type="target_type",
        reg_type="mean",
        axis_keys={"x": "pred", "y": "stimulated"},
        condition_key="condition",
        gene_draw=degs_sti[:10],
        top_gene_list=degs_sti,
        title=None,
        fontsize=12,
    )
    fig_list.append(fig)

    key_dict = {
        "condition_key": "condition",
        "cell_type_key": model_config.cell_type_key,
        "ctrl_key": "control",
        "stim_key": "stimulated",
        "pred_key": "pred",
    }

    df_deg_20 = get_pearson2(
        eval_adata, key_dic=key_dict, n_degs=20, sample_ratio=0.8, times=100
    )
    df_deg_100 = get_pearson2(
        eval_adata, key_dic=key_dict, n_degs=100, sample_ratio=0.8, times=100
    )
    df_deg_20.to_csv(output_path / "r2_degs_20.csv", index=False)
    df_deg_100.to_csv(output_path / "r2_degs_100.csv", index=False)

    """ dotplot """
    if save_plots:
        marker_genes = degs_sti[:20]
        dotplot_fig = sc.pl.dotplot(
            eval_adata, marker_genes, groupby="condition", return_fig=True, show=False
        )
        dotplot_path = output_path / "dotplot.pdf"
        print("saving dotplot to", dotplot_path)
        dotplot_fig.savefig(str(dotplot_path))

    """ save to pdf """
    if save_plots:
        with PdfPages(output_path / "evaluation.pdf") as pdf:
            for i in range(len(fig_list)):
                pdf.savefig(figure=fig_list[i], dpi=200, bbox_inches="tight")
                plt.close()
            print("Evaluation figs saved to", output_path / "evaluation.pdf")

    ground_truth_evaluation = MetricsPerGene.from_adata(
        adata=ground_truth, degs=list(degs_sti)
    )
    predicted_evaluation = MetricsPerGene.from_adata(
        adata=predicted, degs=list(degs_sti)
    )
    diff = MetricsPerGene.compare(ground_truth_evaluation, predicted_evaluation)
    ground_truth_evaluation.save(f"{output_path}/ground_truth")
    predicted_evaluation.save(f"{output_path}/predicted")

    data = {
        "model": [model_config.model_name],
        "dataset": [model_config.dataset_name],
        "experiment_name": [model_config.experiment_name],
        "perturbation": [model_config.perturbation],
        "dose": [model_config.dosage],
        "DEGs": [common_nums],
        "r2mean": [r2mean],
        "r2mean_top20": [r2mean_top20],
        "r2mean_top100": [r2mean_top100],
        "r2mean_top20_boostrap_mean": [df_deg_20["r2_degs_mean"].mean()],
        "r2mean_top100_boostrap_mean": [df_deg_100["r2_degs_mean"].mean()],
        "cell_type_test": [target_type],
        "average_mean_expressed_diff": [np.nanmean(diff.mean_expressed)],
        "average_fractions_diff": [np.nanmean(diff.fractions)],
        "average_mean_degs20_diff": [np.nanmean(diff.get_mean_degs(20))],
        "average_mean_degs100_diff": [np.nanmean(diff.get_mean_degs(100))],
    }
    pd_data = pd.DataFrame(data)
    pd_data.to_csv(output_path / "metrics.csv", index=False)
    print("Writing metrics to", output_path / "metrics.csv")

    if append_metrics:
        append_csv(pd_data, METRICS_PATH)

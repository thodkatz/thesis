# %%
from thesis import ROOT, SAVED_RESULTS_PATH
from thesis.datasets import NaultMultiplePipeline, NaultPipeline
from thesis.evaluation import evaluation_out_of_sample
from thesis.multi_task_aae import FilmLayerFactory, MultiTaskAae, MultiTaskAdversarialAutoencoderUtils, NaultDataset
from anndata import AnnData
import scanpy as sc
import pandas as pd
import numpy as np
import os
from typing import List, Optional
from pandas import DataFrame
import matplotlib.pyplot as plt

overwrite = True

BASELINE_METRICS = ['DEGs', 'r2mean_all_boostrap_mean', 'r2mean_top20_boostrap_mean', 'r2mean_top100_boostrap_mean']
DISTANCE_METRICS = ['edistance', 'wasserstein', 'euclidean', 'mean_pairwise', 'mmd']

METRICS = BASELINE_METRICS + DISTANCE_METRICS

FIGURES_PATH = SAVED_RESULTS_PATH / "multi_task_aae_eval" / "figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

# %%
dataset_pipeline = NaultMultiplePipeline(dataset_pipeline=NaultPipeline())
# dataset_pipeline = PbmcSinglePipeline(
#     dataset_pipeline=PbmcPipeline())


# %%
from thesis.utils import setup_seed

dataset = NaultDataset(
    dataset_condition_pipeline=dataset_pipeline,
    target_cell_type="Hepatocytes - portal",
)

experiment_name = "adversarial"

setup_seed()

film_factory = FilmLayerFactory(
    input_dim=dataset.get_condition_len(),
    hidden_layers=[],
)

tensorboard_path = ROOT / "runs" / "multi_task_aae" / experiment_name
saved_path = SAVED_RESULTS_PATH / "multi_task_aae.pt"

hidden_layers_autoencoder = [32, 32]
hidden_layers_discriminator = [32, 32]

if saved_path.exists() and not overwrite:
    print("Loading model")
    model = MultiTaskAae.load(
        num_features=dataset.get_num_features(),
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        film_layer_factory=film_factory,
        load_path=saved_path,
    )
else:
    model = MultiTaskAae(
        num_features=dataset.get_num_features(),
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        film_layer_factory=film_factory,
    )

model_utils = MultiTaskAdversarialAutoencoderUtils(dataset=dataset, model=model)


if saved_path.exists() and not overwrite:
    pass
else:
    model_utils.train(
        save_path=saved_path, tensorboard_path=tensorboard_path, epochs=100
    )

predictions = model_utils.predict()

dfs = []

for idx, dosage in enumerate(dataset.get_dosages_to_test()):
    evaluation_path = SAVED_RESULTS_PATH / "multi_task_aae_eval" / f"dosage{dosage}"

    df, _ = evaluation_out_of_sample(
        control=dataset.get_ctrl_test(),
        ground_truth=dataset.get_stim_test(dose=dosage),
        predicted=predictions[idx],
        output_path=evaluation_path,
        save_plots=False,
        cell_type_key=dataset_pipeline.cell_type_key,
    )
    df["dose"] = dosage
    dfs.append(df)

    print("Finished evaluation for dosage", dosage)

# %%
from thesis.utils import append_csv

load = False
if load:
    all_df = pd.read_csv(ROOT / "analysis" / "multi_task_aae.csv")
else:
    all_df = pd.concat(dfs, axis=0)
    all_df['experiment'] = experiment_name
    append_csv(all_df, ROOT / "analysis" / "multi_task_aae.csv")

# %%
def _plot_2d_metrics(
    dataset: DataFrame,
    title: str,
    x_labels: List[str],
    metrics: List[str] = BASELINE_METRICS,
    file_name_to_save: Optional[str] = None,
):
    x = np.arange(len(x_labels))
    
    nrows = int(np.ceil(len(metrics) / 2))

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        ax.bar(
            x,
            dataset[metric],
            label='MultiTaskAae',
            alpha=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel(metric)
        # ax.set_title(f"Comparison of {metric}")
        
    if len(metrics) % 2 != 0:
        fig.delaxes(axes[-1])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)
    fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at the top for the legend
    if file_name_to_save:
        plt.savefig(
            FIGURES_PATH / f"{file_name_to_save}.pdf", dpi=300, bbox_inches="tight"
        )
    plt.show()


def plot_2d_metrics_per_dosage(
    title: str,
    file_name_to_save: Optional[str] = None,
    metrics=BASELINE_METRICS
):
    dosages = dataset.get_dosages_to_test()
    assert len(dosages) > 1
    _plot_2d_metrics(
        dataset=all_df,
        title=title,
        x_labels=dosages,
        file_name_to_save=file_name_to_save,
        metrics=metrics
    )

# %%
#plot_2d_metrics_per_dosage(title="")

# %%
#plot_2d_metrics_per_dosage(title="", metrics=DISTANCE_METRICS)

# %%
# control_adata = dataset.get_ctrl_test()
# control_tensor = dataset.get_gene_expressions(control_adata).to("cuda")
# latent = AnnData(X=model.get_latent_representation(control_tensor), obs=control_adata.obs.copy())

# sc.pp.neighbors(latent)
# sc.tl.umap(latent)
# sc.pl.umap(latent, color=['celltype'])

# # %%
# train_adata = dataset.get_train()
# train_tensor = dataset.get_gene_expressions(train_adata).to("cuda")
# latent = AnnData(X=model.get_latent_representation(train_tensor), obs=train_adata.obs.copy())

# latent.obs['Dose'] = latent.obs['Dose'].astype('category')

# sc.pp.neighbors(latent)
# sc.tl.umap(latent)

# sc.pl.umap(latent, color=['Dose'])
# sc.pl.umap(latent, color=['celltype'])



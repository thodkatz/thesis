# %%
from thesis import ROOT, SAVED_RESULTS_PATH
from thesis.datasets import NaultMultiplePipeline, NaultPipeline, NaultSinglePipeline
from thesis.evaluation import evaluation_out_of_sample
from thesis.multi_task_aae import (
    FilmLayerFactory,
    MultiTaskAae,
    MultiTaskAdversarialAutoencoderUtils,
    NaultDataset,
)
from anndata import AnnData
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt
from thesis.utils import setup_seed
from thesis.utils import append_csv

overwrite = True

BASELINE_METRICS = [
    "DEGs",
    "r2mean_all_boostrap_mean",
    "r2mean_top20_boostrap_mean",
    "r2mean_top100_boostrap_mean",
]
DISTANCE_METRICS = ["edistance", "wasserstein", "euclidean", "mean_pairwise", "mmd"]

METRICS = BASELINE_METRICS + DISTANCE_METRICS

FIGURES_PATH = SAVED_RESULTS_PATH / "multi_task_aae_eval" / "figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

setup_seed()

is_single = False
# %%

if not is_single:
    dataset_pipeline = NaultMultiplePipeline(dataset_pipeline=NaultPipeline())
else:
    dataset_pipeline = NaultSinglePipeline(
        dataset_pipeline=NaultPipeline(), dosages=30.0
    )

# %%

dataset = NaultDataset(
    dataset_condition_pipeline=dataset_pipeline,
    target_cell_type="Hepatocytes - portal",
)

experiment_name = "dosage_no_adversarial"

film_factory = FilmLayerFactory(
    input_dim=dataset.get_condition_len(),
    hidden_layers=[],
)

tensorboard_path = SAVED_RESULTS_PATH / "runs" / "multi_task_aae" / experiment_name
saved_path = SAVED_RESULTS_PATH / "multi_task_aae.pt"

hidden_layers_autoencoder = [256, 128]
hidden_layers_discriminator = [64, 64]

if saved_path.exists() and not overwrite:
    print("Loading model")
    model = MultiTaskAae.load(
        num_features=dataset.get_num_features(),
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        film_layer_factory=film_factory,
        load_path=saved_path,
    )
    model = model.to("cuda")
else:
    model = MultiTaskAae(
        num_features=dataset.get_num_features(),
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        film_layer_factory=film_factory,
    )


# %%

model_utils = MultiTaskAdversarialAutoencoderUtils(dataset=dataset, model=model)


if saved_path.exists() and not overwrite:
    pass
else:
    model_utils.train(
        save_path=saved_path,
        tensorboard_path=tensorboard_path,
        epochs=100,
        is_adversarial=False,
    )


# %%
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
        skip_distances=True,
    )
    df["dose"] = dosage
    dfs.append(df)

    print("Finished evaluation for dosage", dosage)


all_df = pd.concat(dfs, axis=0)
all_df["experiment"] = experiment_name
append_csv(all_df, ROOT / "analysis" / "multi_task_aae.csv")

overview_df = pd.DataFrame()
overview_df["experiment"] = [experiment_name]
overview_df["cell_type_test"] = dataset.target_cell_type
overview_df["DEGs"] = all_df["DEGs"].mean()
overview_df["r2mean_all_boostrap_mean"] = all_df["r2mean_all_boostrap_mean"].mean()
overview_df["r2mean_top20_boostrap_mean"] = all_df["r2mean_top20_boostrap_mean"].mean()
overview_df["r2mean_top100_boostrap_mean"] = all_df[
    "r2mean_top100_boostrap_mean"
].mean()
append_csv(overview_df, ROOT / "analysis" / "multi_task_aae_overview.csv")


# %%
def umaps(adata, title: str = ""):
    tensor = dataset.get_gene_expressions(adata).to("cuda")
    latent = AnnData(X=model.get_latent_representation(tensor), obs=adata.obs.copy())

    latent.obs["Dose"] = latent.obs["Dose"].astype("category")

    sc.pp.neighbors(latent)
    sc.tl.umap(latent)

    sc.pl.umap(latent, color=["Dose"])
    plt.savefig(
        f"{FIGURES_PATH}/multi_task_aae_umap_dose_{experiment_name}_{title}.pdf",
        dpi=150,
        bbox_inches="tight",
    )

    sc.pl.umap(latent, color=["celltype"])
    plt.savefig(
        f"{FIGURES_PATH}/multi_task_aae_umap_celltype_{experiment_name}_{title}.pdf",
        dpi=150,
        bbox_inches="tight",
    )


# %%
# umaps(dataset.get_train(), title='train')

# %%

# umaps(dataset.get_stim_test(), title='stim')

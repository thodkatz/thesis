# %%
from thesis import ROOT, SAVED_RESULTS_PATH
from thesis.datasets import NaultMultiplePipeline, NaultPipeline, NaultSinglePipeline
from thesis.evaluation import evaluation_out_of_sample
from thesis.multi_task_aae import (
    FilmLayerFactory,
    MultiTaskAae,
    MultiTaskAdversarialAutoencoderTrainer,
    MultiTaskAdversarialAutoencoderUtils,
    DosagesDataset,
)
from anndata import AnnData
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt
from thesis.utils import setup_seed
from thesis.utils import append_csv
import argparse

overwrite = False

parser = argparse.ArgumentParser(description="Run multi-task aae")
parser.add_argument("--batch_size", type=int, required=False, help="Batch size")
parser.add_argument("--lr", type=float, required=False, help="Learning rate")
parser.add_argument(
    "--autoencoder_pretrain_epochs",
    type=int,
    required=False,
    help="Autoencoder pretrain epochs",
)
parser.add_argument(
    "--discriminator_pretrain_epochs",
    type=int,
    required=False,
    help="Discriminator pretrain epochs",
)
parser.add_argument(
    "--adversarial_epochs", type=int, required=False, help="Adversarial epochs"
)
parser.add_argument(
    "--coeff_adversarial", type=float, required=False, help="Adversarial coefficient"
)
parser.add_argument(
    "--hidden_layers_ae",
    type=int,
    nargs="+",
    required=False,
    help="Hidden layers as space-separated integers (e.g., 64 128 256)",
)
parser.add_argument(
    "--hidden_layers_disc",
    type=int,
    nargs="+",
    required=False,
    help="Hidden layers as space-separated integers (e.g., 64 128 256)",
)
parser.add_argument(
    "--hidden_layers_film",
    type=int,
    nargs="+",
    required=False,
    help="Hidden layers as space-separated integers (e.g., 64 128 256)",
)
args = parser.parse_args()


BASELINE_METRICS = [
    "DEGs",
    "r2mean_all_boostrap_mean",
    "r2mean_top20_boostrap_mean",
    "r2mean_top100_boostrap_mean",
]
DISTANCE_METRICS = ["edistance", "wasserstein", "euclidean", "mean_pairwise", "mmd"]

METRICS = BASELINE_METRICS + DISTANCE_METRICS

batch_size = args.batch_size or 256
learning_rate = args.lr or 1e-4
autoencoder_pretrain_epochs = args.autoencoder_pretrain_epochs or 100
discriminator_pretrain_epochs = args.discriminator_pretrain_epochs or 10
adversarial_epochs = args.adversarial_epochs or 100
coeff_adversarial = args.coeff_adversarial or 0.05
hidden_layers_autoencoder = args.hidden_layers_ae or [32, 32]
hidden_layers_discriminator = args.hidden_layers_disc or [32, 32]
hidden_layers_film = args.hidden_layers_film or []

experiment_name = (
    f"layers_ae_{hidden_layers_autoencoder}_disc_{hidden_layers_discriminator}_film_{hidden_layers_film}_"
    f"lr_{learning_rate}_batch_{batch_size}_ae_epochs_{autoencoder_pretrain_epochs}_"
    f"dis_epochs_{discriminator_pretrain_epochs}_adv_epochs_{adversarial_epochs}_"
    f"coef_adv_{coeff_adversarial}"
)
print(experiment_name)

MULTI_TASK_AAE_PATH = SAVED_RESULTS_PATH / "multi_task_aae" / experiment_name

TENSORBOARD_PATH = SAVED_RESULTS_PATH / "runs" / "multi_task_aae" / experiment_name

model_path = MULTI_TASK_AAE_PATH / "model.pt"

FIGURES_PATH = MULTI_TASK_AAE_PATH / "figures"

os.makedirs(MULTI_TASK_AAE_PATH, exist_ok=True)
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

target_cell_type = "Hepatocytes - portal"

condition_len = len(dataset_pipeline.get_dosages_unique())
num_features = dataset_pipeline.get_num_genes()

film_factory = FilmLayerFactory(
    input_dim=condition_len,
    hidden_layers=hidden_layers_film,
)


if model_path.exists() and not overwrite:
    print("Loading model")
    model = MultiTaskAae.load(
        num_features=num_features,
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        film_layer_factory=film_factory,
        load_path=MULTI_TASK_AAE_PATH,
    )
    model = model.to("cuda")
else:
    model = MultiTaskAae(
        num_features=num_features,
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        film_layer_factory=film_factory,
    )


# %%

model_utils = MultiTaskAdversarialAutoencoderUtils(
    split_dataset_pipeline=dataset_pipeline,
    target_cell_type=target_cell_type,
    model=model,
)


if model_path.exists() and not overwrite:
    pass
else:
    trainer = MultiTaskAdversarialAutoencoderTrainer(
        model=model,
        tensorboard_path=TENSORBOARD_PATH,
        split_dataset_pipeline=dataset_pipeline,
        target_cell_type=target_cell_type,
        device="cuda",
        coeff_adversarial=0.05,
        autoencoder_pretrain_epochs=autoencoder_pretrain_epochs,
        discriminator_pretrain_epochs=discriminator_pretrain_epochs,
        adversarial_epochs=adversarial_epochs,
        lr=learning_rate,
        batch_size=batch_size,
    )

    model_utils.train(trainer=trainer, save_path=MULTI_TASK_AAE_PATH)


# %%
predictions = model_utils.predict()

dfs = []

stim_test = dataset_pipeline.get_stim_test(target_cell_type=target_cell_type)
control_test = dataset_pipeline.get_ctrl_test(target_cell_type=target_cell_type)

dosages_to_test = dataset_pipeline.get_dosages_unique(stim_test)

for idx, dosage in enumerate(dosages_to_test):
    evaluation_path = MULTI_TASK_AAE_PATH / f"dosage{dosage}"

    df, _ = evaluation_out_of_sample(
        control=control_test,
        ground_truth=stim_test[stim_test.obs[dataset_pipeline.dosage_key] == dosage],
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
overview_df["cell_type_test"] = target_cell_type
overview_df["DEGs"] = all_df["DEGs"].mean()
overview_df["r2mean_all_boostrap_mean"] = all_df["r2mean_all_boostrap_mean"].mean()
overview_df["r2mean_top20_boostrap_mean"] = all_df["r2mean_top20_boostrap_mean"].mean()
overview_df["r2mean_top100_boostrap_mean"] = all_df[
    "r2mean_top100_boostrap_mean"
].mean()
append_csv(overview_df, ROOT / "analysis" / "multi_task_aae_overview.csv")


# %%
def umaps(adata, title: str = ""):
    tensor = DosagesDataset.get_gene_expressions(adata).to("cuda")
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
train_adata, validation_adata = dataset_pipeline.split_dataset_to_train_validation(
    target_cell_type=target_cell_type
)
umaps(train_adata, title="train")
umaps(validation_adata, title="validation")

# %%

umaps(stim_test, title="stim")

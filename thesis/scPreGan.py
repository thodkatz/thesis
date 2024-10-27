from scPreGAN.reproducibility.scPreGAN_OOD_prediction import train_and_predict
from thesis.datasets import get_control_perturb_pbmc
from typing import Optional
from anndata import AnnData
from thesis import SAVED_RESULTS_PATH
from thesis.utils import ModelConfig
from scPreGAN.model.util import load_anndata
from scPreGAN.model.scPreGAN import is_model_trained
from scPreGAN import Model
from thesis.evaluation import evaluation_out_of_sample

REFRESH = False


def run_pbmc(experiment_name: str, dataset: AnnData, batch: Optional[int] = None):
    model_config = ModelConfig(
        model_name="scPreGan",
        dataset_name="pbmc",
        experiment_name=experiment_name,
        perturbation="ifn-b",
        cell_type_key="cell_type",
        root_path=SAVED_RESULTS_PATH,
    )

    cell_types = dataset.obs["cell_type"].unique().tolist()

    if batch is None:
        batch_list = [idx for idx, _ in enumerate(cell_types)]
    else:
        batch_list = [batch]

    for batch_idx in batch_list:
        _run_batch(model_config=model_config, dataset=dataset, batch=batch_idx)


def _run_batch(model_config: ModelConfig, dataset: AnnData, batch: int):
    condition_key = "condition"
    condition = {"case": "stimulated", "control": "control"}
    cell_type_key = "cell_type"

    cell_types = dataset.obs[cell_type_key].unique().tolist()
    target_cell_type = cell_types[batch]

    adata_split, train_data = load_anndata(
        adata=dataset,
        condition_key=condition_key,
        condition=condition,
        cell_type_key=cell_type_key,
        target_cell_type=target_cell_type,
    )
    control_adata, perturb_adata, case_adata = adata_split
    control_pd, control_celltype_ohe_pd, perturb_pd, perturb_celltype_ohe_pd = (
        train_data
    )

    n_features = control_pd.shape[1]
    n_classes = control_adata.obs[cell_type_key].unique().shape[0]

    output_path = model_config.get_batch_path(batch=batch)
    tensorboard_path = model_config.get_batch_log_path(batch=batch)

    load_model = is_model_trained(output_path=output_path) and not REFRESH

    model = Model(n_features=n_features, n_classes=n_classes, use_cuda=True)
    model.train(
        train_data=train_data,
        output_path=output_path,
        load_model=load_model,
        tensorboard_path=tensorboard_path,
    )

    control_test_adata = control_adata[
        control_adata.obs["cell_type"] == target_cell_type
    ]
    perturb_test_adata = perturb_adata[
        perturb_adata.obs["cell_type"] == target_cell_type
    ]

    pred_perturbed_adata = model.predict(
        control_adata=control_test_adata,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
    )

    evaluation_out_of_sample(
        model_config=model_config,
        input=control_test_adata,
        ground_truth=perturb_test_adata,
        predicted=pred_perturbed_adata,
        output_path=model_config.get_batch_path(batch),
        save_plots=True,
        append_metrics=True,
    )


def run_pbmc_reproducible(dataset: AnnData, batch: Optional[int] = None):
    model_config = ModelConfig(
        model_name="scPreGan",
        dataset_name="pbmc",
        experiment_name="reproducible",
        perturbation="ifn-b",
        cell_type_key="cell_type",
        root_path=SAVED_RESULTS_PATH,
    )

    cell_types = dataset.obs["cell_type"].unique().tolist()

    if batch is None:
        batch_list = [idx for idx, _ in enumerate(cell_types)]
    else:
        batch_list = [batch]

    for batch_idx in batch_list:
        _run_batch_reproducible(
            model_config=model_config, dataset=dataset, batch=batch_idx
        )


def _run_batch_reproducible(model_config: ModelConfig, dataset: AnnData, batch: int):
    output_path_reproducible = model_config.get_batch_path(batch=batch)

    opt = {
        "cuda": True,
        "dataset": dataset,
        "checkpoint_dir": None,
        "condition_key": "condition",
        "condition": {"case": "stimulated", "control": "control"},
        "cell_type_key": "cell_type",
        "prediction_type": "Dendritic",
        "out_sample_prediction": True,
        "manual_seed": 3060,
        "data_name": "pbmc",
        "model_name": "pbmc_OOD",
        "outf": output_path_reproducible,
        "validation": False,
        "valid_dataPath": None,
        "use_sn": True,
        "use_wgan_div": True,
        "gan_loss": "wgan",
    }

    config = {
        "batch_size": 64,
        "lambda_adv": 0.001,
        "lambda_encoding": 0.1,
        "lambda_l1_reg": 0,
        "lambda_recon": 1,
        "lambta_gp": 1,
        "lr_disc": 0.001,
        "lr_e": 0.0001,
        "lr_g": 0.001,
        "min_hidden_size": 256,
        "niter": 20_000,
        "z_dim": 16,
    }

    pred_perturbed_reproducible_adata = train_and_predict(
        opt=opt,
        config=config,
        tensorboard_path=model_config.get_batch_log_path(batch),
    )

    control_adata, perturb_adata = get_control_perturb_pbmc(dataset)

    cell_types = dataset.obs[model_config.cell_type_key].unique().tolist()
    target_cell_type = cell_types[batch]

    control_test_adata = control_adata[
        control_adata.obs["cell_type"] == target_cell_type
    ]
    perturb_test_adata = perturb_adata[
        perturb_adata.obs["cell_type"] == target_cell_type
    ]

    evaluation_out_of_sample(
        model_config=model_config,
        input=control_test_adata,
        ground_truth=perturb_test_adata,
        predicted=pred_perturbed_reproducible_adata,
        output_path=model_config.get_batch_path(batch),
        save_plots=True,
        append_metrics=True,
    )

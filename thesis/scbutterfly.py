from anndata import AnnData
from scButterfly.train_model_perturb import Model
from scButterfly.split_datasets import (
    unpaired_split_dataset_perturb,
    unpaired_split_dataset_perturb_no_reusing,
)
import torch.nn as nn
from typing import List, Callable, Optional
from thesis import SAVED_RESULTS_PATH, METRICS_PATH
import pandas as pd
from thesis.evaluation import Evaluation
import numpy as np
from pathlib import Path
from thesis.utils import append_csv


SAVED_RESULTS_BUTTERFLY_PATH = SAVED_RESULTS_PATH / "butterfly" / "perturb"
REFRESH = False


def is_finished_batch(dataset_name: str, batch: int):
    # test_train is the last metrics file
    return (
        Path(
            f"{get_batch_path(dataset_name, batch)}/test_train/cluster_index.csv"
        ).exists()
        and not REFRESH
    )


def get_metrics_dataset_path(dataset_name: str):
    return f"{get_dataset_path(dataset_name)}/metrics.csv"


def get_batch_metrics_path(dataset_name: str, batch: int):
    return f"{get_batch_path(dataset_name, batch)}/metrics.csv"


def get_batch_path(dataset_name: str, batch: int):
    return f"{get_dataset_path(dataset_name)}/batch{batch}"


def get_dataset_path(name: str):
    return f"{SAVED_RESULTS_BUTTERFLY_PATH}/{name}"


def _run_dataset(
    name: str,
    control: AnnData,
    perturb: AnnData,
    id_list: List,
    batch_list: List[int],
    perturbation_name: str,
    dosage: int,
):
    for idx, batch in enumerate(batch_list):
        print("name", name, "Batch", batch)
        _run_batch(name=name, control=control, perturb=perturb, id_list=id_list, batch=batch, perturbation_name=perturbation_name, dosage=dosage)


def _run_batch(
    name: str, control: AnnData, perturb: AnnData, id_list: List, batch: int, perturbation_name: str, dosage: int
):
    (
        train_id_control,
        train_id_perturb,
        validation_id_control,
        validation_id_perturb,
        test_id_control,
        test_id_perturb,
    ) = id_list[batch]

    RNA_input_dim = control.X.shape[1]
    ATAC_input_dim = perturb.X.shape[1]
    R_kl_div = 1 / RNA_input_dim * 20
    A_kl_div = 1 / ATAC_input_dim * 20
    kl_div = R_kl_div + A_kl_div

    file_path = get_batch_path(name, batch)

    model = Model(
        R_encoder_nlayer=2,
        A_encoder_nlayer=2,
        R_decoder_nlayer=2,
        A_decoder_nlayer=2,
        R_encoder_dim_list=[RNA_input_dim, 256, 128],
        A_encoder_dim_list=[ATAC_input_dim, 128, 128],
        R_decoder_dim_list=[128, 256, RNA_input_dim],
        A_decoder_dim_list=[128, 128, ATAC_input_dim],
        R_encoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        translator_embed_dim=128,
        translator_input_dim_r=128,
        translator_input_dim_a=128,
        translator_embed_act_list=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer=1,
        discriminator_dim_list_R=[128],
        discriminator_dim_list_A=[128],
        discriminator_act_list=[nn.Sigmoid()],
        dropout_rate=0.1,
        R_noise_rate=0.5,
        A_noise_rate=0.5,
        chrom_list=[],
        logging_path=file_path,
        RNA_data=control,
        ATAC_data=perturb,
        name=f"butterfly/{name}/batch{batch}",
    )
    
    if is_finished_batch(name, batch):
        print("Batch already trained", batch)
        print("Loading model")
        load_model = file_path
    else:
        load_model = None

    model.train(
        R_encoder_lr=0.001,
        A_encoder_lr=0.001,
        R_decoder_lr=0.001,
        A_decoder_lr=0.001,
        R_translator_lr=0.001,
        A_translator_lr=0.001,
        translator_lr=0.001,
        discriminator_lr=0.005,
        R2R_pretrain_epoch=100,
        A2A_pretrain_epoch=100,
        lock_encoder_and_decoder=False,
        translator_epoch=200,
        patience=50,
        batch_size=64,
        r_loss=nn.MSELoss(size_average=True),
        a_loss=nn.MSELoss(size_average=True),
        d_loss=nn.BCELoss(size_average=True),
        loss_weight=[1, 1, 1, R_kl_div, A_kl_div, kl_div],
        train_id_r=train_id_control,
        train_id_a=train_id_perturb,
        validation_id_r=validation_id_control,
        validation_id_a=validation_id_perturb,
        output_path=file_path,
        seed=19193,
        kl_mean=True,
        R_pretrain_kl_warmup=50,
        A_pretrain_kl_warmup=50,
        translation_kl_warmup=50,
        load_model=load_model,
        logging_path=file_path,
    )

    perturb_test = perturb[test_id_perturb]

    test_file_path = f"{file_path}/test"
    pd_metrics, pred_adata, degs = model.test(
        test_id_r=test_id_control,
        test_id_a=test_id_perturb,
        model_path=None,
        load_model=False,
        output_path=test_file_path,
        test_pca=True,
        test_DEGs=True,
        test_R2=True,
        test_dotplot=True,
        output_data=False,
        return_predict=False,
    )

    ground_truth_evaluation = Evaluation.from_adata(adata=perturb_test, degs=degs)
    predicted_evaluation = Evaluation.from_adata(adata=pred_adata, degs=degs)
    diff = Evaluation.compare(ground_truth_evaluation, predicted_evaluation)
    ground_truth_evaluation.save(f"{file_path}/ground_truth")
    predicted_evaluation.save(f"{file_path}/predicted")
    diff.save(f"{file_path}/diff")

    test_file_path = f"{file_path}/test_validation"
    model.test(
        test_id_r=validation_id_control,
        test_id_a=validation_id_perturb,
        model_path=None,
        load_model=False,
        output_path=test_file_path,
        test_pca=True,
        test_DEGs=True,
        test_R2=True,
        test_dotplot=True,
        output_data=False,
        return_predict=False,
    )

    test_file_path = f"{file_path}/test_train"
    model.test(
        test_id_r=train_id_control,
        test_id_a=train_id_perturb,
        model_path=None,
        load_model=False,
        output_path=test_file_path,
        test_pca=True,
        test_DEGs=True,
        test_R2=True,
        test_dotplot=True,
        output_data=False,
        return_predict=False,
    )


    data = {
        "model": "butterfly",
        "dataset": name,
        "perturbation": perturbation_name,
        "dose": dosage,
        "DEGs": pd_metrics["DEGs"].values,
        "r2mean": pd_metrics["r2mean"].values,
        "r2mean_top100": pd_metrics["r2mean_top100"].values,
        "cell_type_test": pd_metrics["data_name"].values,
        "average_mean_expressed_diff": np.nanmean(diff.mean_expressed),
        "average_fractions_diff": np.nanmean(diff.fractions),
        "average_fractions_degs_diff": np.nanmean(diff.fractions_degs),
        "average_mean_degs_diff": np.nanmean(diff.mean_degs),
    }
    pd_data = pd.DataFrame(data)
    append_csv(pd_data, METRICS_PATH)


def _run(
    name: str,
    control: AnnData,
    perturb: AnnData,
    split_func: Callable,
    perturbation_name: str,
    dosage: int,
    cell_type_key: str = "celltype",
    batch_idx: Optional[int] = None,
):
    control.obs["cell_type"] = control.obs[cell_type_key]
    perturb.obs["cell_type"] = perturb.obs[cell_type_key]
    control.obs.index = [str(i) for i in range(control.X.shape[0])]
    perturb.obs.index = [str(i) for i in range(perturb.X.shape[0])]

    assert sorted(control.obs[cell_type_key].unique()) == sorted(
        perturb.obs[cell_type_key].unique()
    )
    batch_list = list(range(0, len(control.obs[cell_type_key].cat.categories)))
    # batch_list = [batch_list[0]]
    if batch_idx is not None:
        batch_list = [batch_list[batch_idx]]

    id_list, _ = split_func(control, perturb)
    _run_dataset(
        name=name,
        control=control,
        perturb=perturb,
        id_list=id_list,
        batch_list=batch_list,
        perturbation_name=perturbation_name,
        dosage=dosage,
    )


def _run_sciplex3(name: str, dataset: AnnData, perturbation_name: str, dosage: int, split_func: Callable, batch: Optional[int] = None):
    control, perturb = _get_control_perturb_sciplex3(dataset)
    return _run(
        name=name,
        control=control,
        perturb=perturb,
        cell_type_key="celltype",
        split_func=split_func,
        batch_idx=batch,
        perturbation_name=perturbation_name,
        dosage=dosage
    )


def run_sciplex3(name: str, dataset: AnnData, perturbation_name: str, dosage: int, batch: Optional[int] = None):
    return _run_sciplex3(
        name=name, dataset=dataset, split_func=unpaired_split_dataset_perturb, batch=batch, perturbation_name=perturbation_name, dosage=dosage
    )


def run_sciplex3_no_reusing(name: str, dataset: AnnData, perturbation_name: str, dosage: int, batch: Optional[int] = None):
    return _run_sciplex3(
        name=name, dataset=dataset, split_func=unpaired_split_dataset_perturb_no_reusing, batch=batch, perturbation_name=perturbation_name, dosage=dosage
    )


def _get_control_perturb_sciplex3(dataset: AnnData):
    num_perturbations = dataset.obs.perturbation.unique()
    assert len(num_perturbations) == 2
    dataset.obs["condition"] = dataset.obs["perturbation"].apply(
        lambda x: "control" if x == "control" else "stimulated"
    )
    control = dataset[dataset.obs.condition == "control"]
    perturb = dataset[dataset.obs.condition == "stimulated"]
    return control, perturb


def _get_control_perturb_nault(dataset: AnnData, drug_dosage: int):
    control = dataset[dataset.obs["Dose"] == 0]
    assert drug_dosage in dataset.obs["Dose"].unique()
    perturb = dataset[dataset.obs["Dose"] == drug_dosage]
    control.obs["condition"] = "control"
    perturb.obs["condition"] = "stimulated"
    return control, perturb


def run_nault_all_dosages(dataset: AnnData, name="nault"):
    for drug_dosage in dataset.obs["Dose"].unique():
        if drug_dosage == 0:
            continue
        print("Drug dosage", drug_dosage)
        control, perturb = _get_control_perturb_nault(dataset, drug_dosage)
        _run(
            name=f"{name}_{drug_dosage}",
            control=control,
            perturb=perturb,
            split_func=unpaired_split_dataset_perturb,
            cell_type_key="celltype",
            perturbation_name="tcdd",
            dosage=drug_dosage
        )


def run_nault_dosage(name: str, dataset: AnnData, drug_dosage: int, batch: Optional[int] = None):
    control, perturb = _get_control_perturb_nault(dataset, drug_dosage)
    return _run(
        name=name,
        control=control,
        perturb=perturb,
        split_func=unpaired_split_dataset_perturb,
        cell_type_key="celltype",
        batch_idx=batch,
        perturbation_name="tcdd",
        dosage=drug_dosage
    )


def run_pbmc(name: str, dataset: AnnData, batch: Optional[int] = None):
    control, perturb = _get_control_perturb_pbmc(dataset)
    return _run(
        name=name,
        control=control,
        perturb=perturb,
        split_func=unpaired_split_dataset_perturb,
        cell_type_key="cell_type",
        batch_idx=batch,
        perturbation_name="ifn-b",
        dosage=0
    )


def _get_control_perturb_pbmc(dataset: AnnData):
    control = dataset[dataset.obs["condition"] == "control"]
    perturb = dataset[dataset.obs["condition"] == "stimulated"]
    return control, perturb

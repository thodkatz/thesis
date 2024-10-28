from anndata import AnnData
from matplotlib.dates import MO
from scButterfly.train_model_perturb import Model
from scButterfly.split_datasets import (
    unpaired_split_dataset_perturb,
    unpaired_split_dataset_perturb_no_reusing,
)
import torch.nn as nn
from typing import List, Callable, Optional
from thesis import SAVED_RESULTS_PATH
import pandas as pd
from thesis.evaluation import evaluation_out_of_sample
import numpy as np
from pathlib import Path
from thesis.utils import ModelConfig
from thesis.datasets import (
    get_control_perturb_nault,
    get_control_perturb_pbmc,
    get_control_perturb_sciplex3,
)


REFRESH = False



def _run_dataset(
    model_config: ModelConfig,
    control: AnnData,
    perturb: AnnData,
    id_list: List,
    batch_list: List[int],
):
    for idx, batch in enumerate(batch_list):
        print("config", model_config, "Batch", batch)
        _run_batch(
            model_config=model_config,
            control=control,
            perturb=perturb,
            id_list=id_list,
            batch=batch,
        )


def _run_batch(
    model_config: ModelConfig,
    control: AnnData,
    perturb: AnnData,
    id_list: List,
    batch: int,
):
    if model_config.is_finished_batch(batch, refresh=REFRESH):
        print("Batch already trained", batch)
        return
    
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

    file_path = str(model_config.get_batch_path(batch))
    tensorboard_path = model_config.get_batch_log_path(batch)

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
        model_config_log_path=file_path,
        RNA_data=control,
        ATAC_data=perturb,
        tensorboard_path=tensorboard_path,
    )



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
        load_model=None,
    )

    input_test, ground_truth_test, predicted_test = model.test(
        test_id_r=test_id_control,
        test_id_a=test_id_perturb,
    )
    evaluation_out_of_sample(
        model_config=model_config,
        input=input_test,
        ground_truth=ground_truth_test,
        predicted=predicted_test,
        output_path=Path(file_path),
        append_metrics=True,
        save_plots=False,
    )

    test_file_path = f"{file_path}/test_validation"
    input_test, ground_truth_test, predicted_test = model.test(
        test_id_r=validation_id_control,
        test_id_a=validation_id_perturb,
    )
    evaluation_out_of_sample(
        model_config=model_config,
        input=input_test,
        ground_truth=ground_truth_test,
        predicted=predicted_test,
        output_path=Path(test_file_path),
        append_metrics=False,
        save_plots=False,
    )

    test_file_path = f"{file_path}/test_train"
    input_train, ground_truth_train, predicted_train = model.test(
        test_id_r=train_id_control,
        test_id_a=train_id_perturb,
    )
    evaluation_out_of_sample(
        model_config=model_config,
        input=input_train,
        ground_truth=ground_truth_train,
        predicted=predicted_train,
        output_path=Path(test_file_path),
        append_metrics=False,
        save_plots=False,
    )


def run(
    model_config: ModelConfig,
    control: AnnData,
    perturb: AnnData,
    split_func: Callable,
    batch_idx: Optional[int] = None,
):
    cell_type_key = model_config.cell_type_key
    control.obs["cell_type"] = control.obs[cell_type_key]
    perturb.obs["cell_type"] = perturb.obs[cell_type_key]
    control.obs.index = [str(i) for i in range(control.X.shape[0])]
    perturb.obs.index = [str(i) for i in range(perturb.X.shape[0])]

    assert sorted(control.obs[cell_type_key].unique()) == sorted(
        perturb.obs[cell_type_key].unique()
    )
    # batch_list = [batch_list[0]]
    if batch_idx is not None:
        batch_list = [batch_idx]
    else:
        batch_list = list(range(0, len(control.obs[cell_type_key].cat.categories)))

    id_list, _ = split_func(control, perturb)
    _run_dataset(
        model_config=model_config,
        control=control,
        perturb=perturb,
        id_list=id_list,
        batch_list=batch_list,
    )


def _run_sciplex3(
    name: str,
    dataset: AnnData,
    perturbation_name: str,
    dosage: int,
    split_func: Callable,
    batch: Optional[int] = None,
):
    control, perturb = get_control_perturb_sciplex3(dataset)

    model_config = ModelConfig(
        model_name="scbutterfly",
        dataset_name="sciplex3",
        experiment_name=name,
        perturbation=perturbation_name,
        cell_type_key="celltype",
        dosage=dosage,
        root_path=SAVED_RESULTS_PATH,
    )

    return run(
        model_config=model_config,
        control=control,
        perturb=perturb,
        split_func=split_func,
        batch_idx=batch,
    )


def run_sciplex3(
    dataset: AnnData,
    perturbation_name: str,
    dosage: int,
    batch: Optional[int] = None,
):
    return _run_sciplex3(
        name="",
        dataset=dataset,
        split_func=unpaired_split_dataset_perturb,
        batch=batch,
        perturbation_name=perturbation_name,
        dosage=dosage,
    )


def run_sciplex3_no_reusing(
    dataset: AnnData,
    perturbation_name: str,
    dosage: int,
    batch: Optional[int] = None,
):
    return _run_sciplex3(
        name="no_reusing",
        dataset=dataset,
        split_func=unpaired_split_dataset_perturb_no_reusing,
        batch=batch,
        perturbation_name=perturbation_name,
        dosage=dosage,
    )


def run_nault_dosage(
    experiment_name: str,
    dataset: AnnData,
    dosage: Optional[int] = None,
    batch: Optional[int] = None,
):
    drug_dosages = dataset.obs["Dose"].unique()
    if dosage is None:
        drug_dosage_list = drug_dosages
    else:
        drug_dosage_list = [dosage]

    for dose in drug_dosage_list:
        control, perturb = get_control_perturb_nault(dataset, dose)

        model_config = ModelConfig(
            model_name="scbutterfly",
            dataset_name="nault",
            experiment_name=experiment_name,
            perturbation="tcdd",
            cell_type_key="celltype",
            dosage=dose,
            root_path=SAVED_RESULTS_PATH,
        )

        run(
            model_config=model_config,
            control=control,
            perturb=perturb,
            split_func=unpaired_split_dataset_perturb,
            batch_idx=batch,
        )


def run_pbmc(experiment_name: str, dataset: AnnData, batch: Optional[int] = None):
    control, perturb = get_control_perturb_pbmc(dataset)

    model_config = ModelConfig(
        model_name="scbutterfly",
        dataset_name="pbmc",
        experiment_name=experiment_name,
        perturbation="ifn-b",
        cell_type_key="cell_type",
        root_path=SAVED_RESULTS_PATH,
    )

    run(
        model_config=model_config,
        control=control,
        perturb=perturb,
        split_func=unpaired_split_dataset_perturb,
        batch_idx=batch,
    )

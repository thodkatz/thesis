from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import List, Optional, Tuple, Union
from scPreGAN.model.util import load_anndata
import torch.nn as nn
import scPreGAN as scpregan
from scPreGAN.reproducibility.scPreGAN_OOD_prediction import train_and_predict

import scanpy as sc
from anndata import AnnData
from torch import Tensor

from thesis.evaluation import evaluation_out_of_sample

from thesis.utils import FileModelUtils
from thesis.datasets import (
    DatasetSinglePerturbationPipeline,
    DatasetPipeline,
)
from thesis import DATA_PATH, SAVED_RESULTS_PATH
import scButterfly.train_model_perturb as scbutterfly
from scgen import SCGEN

from scButterfly.split_datasets import (
    unpaired_split_dataset_perturb,
    unpaired_split_dataset_perturb_no_reusing,
)

InputAdata = AnnData
PredictedAdata = AnnData
GroundTruthAdata = AnnData

Predict = Tuple[InputAdata, GroundTruthAdata, PredictedAdata]

DatasetPipelines = Union[DatasetPipeline, DatasetSinglePerturbationPipeline]

class ModelPipeline(ABC):
    def __init__(
        self,
        dataset_pipeline: DatasetPipelines,
        experiment_name: str,
    ) -> None:
        if isinstance(dataset_pipeline, DatasetPipeline):
            dataset_pipeline = DatasetSinglePerturbationPipeline(
                dataset_pipeline
            )
        elif isinstance(dataset_pipeline, DatasetSinglePerturbationPipeline):
            dataset_pipeline = dataset_pipeline
        else:
            raise ValueError()
            
        self._model_config = FileModelUtils(
            model_name=str(self),
            dataset_name=str(dataset_pipeline.dataset_pipeline),
            experiment_name=experiment_name,
            perturbation=dataset_pipeline.perturbation,
            dosage=dataset_pipeline.dosage,
            cell_type_key=dataset_pipeline.cell_type_key,
            root_path=SAVED_RESULTS_PATH,
        )
        self._dataset_pipeline = dataset_pipeline

    @abstractmethod
    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        pass

    def __call__(
        self,
        batch: int,
        append_metrics: bool = True,
        save_plots: bool = True,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ):

        predict = self._run(
            batch=batch,
            refresh_training=refresh_training,
            refresh_evaluation=refresh_evaluation,
        )
        if predict is None:
            return
        else:
            input_adata, ground_truth_adata, predicted_adata = predict
        file_path = self._model_config.get_batch_path(batch=batch)
        self.evaluation(
            input=input_adata,
            ground_truth=ground_truth_adata,
            predicted=predicted_adata,
            append_metrics=append_metrics,
            save_plots=save_plots,
            output_path=file_path,
        )

    def evaluation(
        self,
        input: AnnData,
        ground_truth: AnnData,
        predicted: Union[List[Tensor], AnnData],
        output_path: Path,
        append_metrics: bool = True,
        save_plots: bool = True,
    ):
        evaluation_out_of_sample(
            model_config=self._model_config,
            input=input,
            ground_truth=ground_truth,
            predicted=predicted,
            output_path=output_path,
            append_metrics=append_metrics,
            save_plots=save_plots,
        )

    def __str__(self) -> str:
        return self.__class__.__name__


class ButterflyPipeline(ModelPipeline):
    def __init__(
        self,
        dataset_pipeline: DatasetPipelines,
        experiment_name: str,
        debug=False,
    ):
        self._epoch_pretrain1 = 1 if debug else 100
        self._epoch_pretrain2 = 1 if debug else 100
        self._epoch_intergrative = 1 if debug else 200
        self._num_workers = 0 if debug else 4

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
        )

    def split_func(self):
        return unpaired_split_dataset_perturb(
            self._dataset_pipeline.control, self._dataset_pipeline.perturb
        )[0]

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        cell_type_key = self._model_config.cell_type_key
        control = self._dataset_pipeline.control
        perturb = self._dataset_pipeline.perturb

        control.obs["cell_type"] = control.obs[cell_type_key]
        perturb.obs["cell_type"] = perturb.obs[cell_type_key]
        control.obs.index = [str(i) for i in range(control.X.shape[0])]
        perturb.obs.index = [str(i) for i in range(perturb.X.shape[0])]

        id_list = self.split_func()

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

        file_path = str(self._model_config.get_batch_path(batch))
        tensorboard_path = self._model_config.get_batch_log_path(batch)

        model = scbutterfly.Model(
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
            model_config_log_path=str(file_path),
            RNA_data=control,
            ATAC_data=perturb,
            tensorboard_path=tensorboard_path,
        )

        if self._model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        ):
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
            R2R_pretrain_epoch=self._epoch_pretrain1,
            A2A_pretrain_epoch=self._epoch_pretrain2,
            lock_encoder_and_decoder=False,
            translator_epoch=self._num_workers,
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
        )
        self._model_config.log_training_batch_is_finished(batch=batch)

        if self._model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return

        input_test, ground_truth_test, predicted_test = model.test(
            test_id_r=test_id_control, test_id_a=test_id_perturb
        )
        return input_test, ground_truth_test, predicted_test


class ButterflyPipelineNoReusing(ButterflyPipeline):
    def split_func(self):
        return unpaired_split_dataset_perturb_no_reusing(
            self._dataset_pipeline.control, self._dataset_pipeline.perturb
        )


class ScPreGanPipeline(ModelPipeline):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: DatasetPipelines,
        debug: bool = False,
    ):
        self._epochs = 1 if debug else 20_000

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
        )

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        condition_key = "condition"
        condition = {"case": "stimulated", "control": "control"}
        cell_type_key = self._model_config.cell_type_key

        dataset = self._dataset_pipeline.dataset_perturbation
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

        output_path = self._model_config.get_batch_path(batch=batch)
        tensorboard_path = self._model_config.get_batch_log_path(batch=batch)

        model = scpregan.Model(
            n_features=n_features, n_classes=n_classes, use_cuda=True
        )

        load_model = self._model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        )

        model.train(
            train_data=train_data,
            output_path=output_path,
            tensorboard_path=tensorboard_path,
            niter=self._epochs,
            load_model=load_model,
        )
        self._model_config.log_training_batch_is_finished(batch=batch)

        if self._model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return

        control_test_adata = control_adata[
            control_adata.obs[cell_type_key] == target_cell_type
        ]
        perturb_test_adata = perturb_adata[
            perturb_adata.obs[cell_type_key] == target_cell_type
        ]

        pred_perturbed_adata = model.predict(
            control_adata=control_test_adata,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
        )

        return control_test_adata, perturb_test_adata, pred_perturbed_adata


class ScPreGanReproduciblePipeline(ScPreGanPipeline):
    def _run(
        self,
        batch: int,
    ) -> Optional[Predict]:
        output_path_reproducible = self._model_config.get_batch_path(batch=batch)

        dataset = self._dataset_pipeline.dataset_perturbation

        cell_types = dataset.obs[self._model_config.cell_type_key].unique().tolist()
        target_cell_type = cell_types[batch]

        opt = {
            "cuda": True,
            "dataset": dataset,
            "checkpoint_dir": None,
            "condition_key": "condition",
            "condition": {"case": "stimulated", "control": "control"},
            "cell_type_key": "cell_type",
            "prediction_type": target_cell_type,
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
            "niter": self._epochs,
            "z_dim": 16,
        }

        cell_type_key = self._model_config.cell_type_key

        pred_perturbed_reproducible_adata = train_and_predict(
            opt=opt,
            config=config,
            tensorboard_path=self._model_config.get_batch_log_path(batch),
        )

        control_adata = self._dataset_pipeline.control
        perturb_adata = self._dataset_pipeline.perturb

        control_test_adata = control_adata[
            control_adata.obs[cell_type_key] == target_cell_type
        ]
        perturb_test_adata = perturb_adata[
            perturb_adata.obs[cell_type_key] == target_cell_type
        ]

        return control_test_adata, perturb_test_adata, pred_perturbed_reproducible_adata


class ScGenPipeline(ModelPipeline):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: DatasetPipelines,
        debug: bool = False,
    ):
        self._epochs = 1 if debug else 100

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
        )

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        dataset = self._dataset_pipeline.dataset_perturbation
        cell_type_key = self._model_config.cell_type_key
        cell_types = dataset.obs[cell_type_key].unique().tolist()
        target_cell_type = cell_types[batch]
        train_pbmc = dataset[
            ~(
                (dataset.obs[cell_type_key] == target_cell_type)
                & (dataset.obs["condition"] == "stimulated")
            )
        ]

        output_path = self._model_config.get_batch_path(batch=batch)

        # must be a bug of scvi-tools, needs the copy to setup anndata
        train_pbmc = train_pbmc.copy()
        SCGEN.setup_anndata(train_pbmc, batch_key="condition", labels_key=cell_type_key)
        model = SCGEN(adata=train_pbmc)

        if self._model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        ):
            model = model.load(str(output_path), train_pbmc)
        else:
            model.train(
                max_epochs=self._epochs,
                batch_size=32,
                early_stopping=True,
                early_stopping_patience=25,
            )
            model.save(str(output_path), overwrite=True)

        self._model_config.log_training_batch_is_finished(batch=batch)

        if self._model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return None

        control_test_adata = self._dataset_pipeline.control
        perturb_test_adata = self._dataset_pipeline.perturb

        perturb_test_adata = perturb_test_adata[
            perturb_test_adata.obs[cell_type_key] == target_cell_type
        ]

        control_test_adata = control_test_adata[
            control_test_adata.obs[cell_type_key] == target_cell_type
        ]

        pred, delta = model.predict(
            ctrl_key="control",
            stim_key="stimulated",
            celltype_to_predict=target_cell_type,
        )

        return control_test_adata, perturb_test_adata, pred

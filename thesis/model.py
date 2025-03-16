from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Union, cast
from typing_extensions import TypeVar
from scPreGAN.model.util import load_anndata
import torch
import torch.nn as nn
import scPreGAN as scpregan
from scPreGAN.reproducibility.scPreGAN_OOD_prediction import train_and_predict

from anndata import AnnData
from torch import Tensor
import pandas as pd
from vidr import vidr

from thesis.evaluation import evaluation_out_of_sample

from thesis.multi_task_aae import (
    FilmLayerFactory,
    MultiTaskAae,
    MultiTaskAaeAdversarialAndOptimalTransportTrainer,
    MultiTaskAdversarialGaussianAutoencoderTrainer,
    MultiTaskAdversarialTrainer,
    MultiTaskAutoencoderDosagesTrainer,
    MultiTaskAutoencoderOptimalTransportTrainer,
    MultiTaskAutoencoderUtils,
    MultiTaskVae,
    MultiTaskVaeAdversarialAndOptimalTransportTrainer,
    MultiTaskVaeDosagesTrainer,
    MultiTaskVaeOptimalTransportTrainer,
)
from thesis.utils import FileModelUtils, SeedSingleton, append_csv
from thesis.datasets import (
    MultipleConditionDatasetPipeline,
    SingleConditionDatasetPipeline,
    SplitDatasetPipeline,
)
from thesis import METRICS_PATH, SAVED_RESULTS_PATH
import scButterfly.train_model_perturb as scbutterfly
from scgen import SCGEN

from scButterfly.split_datasets import (
    unpaired_split_dataset_perturb,
    unpaired_split_dataset_perturb_no_reusing,
)

InputAdata = AnnData
PredictedAdata = AnnData
GroundTruthAdata = AnnData

Predict = Tuple[InputAdata, List[GroundTruthAdata], List[PredictedAdata]]

DatasetPipelines = Union[
    SingleConditionDatasetPipeline,
    MultipleConditionDatasetPipeline,
]

T = TypeVar("T", SingleConditionDatasetPipeline, MultipleConditionDatasetPipeline)


class ModelPipeline(ABC, Generic[T]):
    def __init__(
        self,
        dataset_pipeline: T,
        experiment_name: str,
        seed: int = 19193,
    ) -> None:
        self.dataset_pipeline = dataset_pipeline

        self.model_config = FileModelUtils(
            model_name=str(self),
            dataset_name=str(object=dataset_pipeline),
            experiment_name=experiment_name,
            perturbation=dataset_pipeline.perturbation,
            dosages=(
                [dataset_pipeline.dosages]
                if self.is_single()
                else dataset_pipeline.dosages
            ),
            cell_type_key=dataset_pipeline.cell_type_key,
            root_path=SAVED_RESULTS_PATH,
        )
        print("CUDA is available", torch.cuda.is_available())
        print("Model config", self.model_config)

        if not SeedSingleton.is_set():
            SeedSingleton(seed=seed)
        else:
            print("seed already set", SeedSingleton.get_value())
            print("ModelPipeline: Torch initial seed", torch.random.initial_seed())

    def is_single(self):
        return self.dataset_pipeline.is_single()

    @abstractmethod
    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        """
        Batch is considered the index of the cell type to be held out for out of distribution prediction

        Return None if the evaluation is already finished
        """
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

        file_path = self.model_config.get_batch_path(batch=batch)

        print("ModelPipeline: Torch initial seed", torch.initial_seed())
        assert (
            torch.random.initial_seed() == SeedSingleton.get_value()
        ), f"make sure models are using the same seed {torch.random.initial_seed()} != {SeedSingleton.get_value()}"

        assert (
            len(ground_truth_adata)
            == len(predicted_adata)
            == len(self.model_config.dosages)
        )
        for idx, (ground_truth, predicted) in enumerate(
            zip(ground_truth_adata, predicted_adata)
        ):
            dose = self.model_config.dosages[idx]
            dose_file_path = (
                self.model_config.get_dose_path_multi(batch, dose)
                if not self.is_single()
                else file_path
            )

            if not self.is_single() and self.model_config.is_finished_evaluation_multi(
                batch=batch, dosage=dose, refresh=refresh_evaluation
            ):
                continue

            self.evaluation(
                control=input_adata,
                ground_truth=ground_truth,
                predicted=predicted,
                output_path=dose_file_path,
                dose=dose,
                append_metrics=append_metrics,
                save_plots=save_plots,
            )

    def evaluation(
        self,
        control: AnnData,
        ground_truth: AnnData,
        predicted: Union[List[Tensor], AnnData],
        output_path: Path,
        dose: float,
        append_metrics: bool = True,
        save_plots: bool = True,
    ):
        df, eval_adata = evaluation_out_of_sample(
            control=control,
            ground_truth=ground_truth,
            predicted=predicted,
            output_path=output_path,
            save_plots=save_plots,
            cell_type_key=self.model_config.cell_type_key,
        )

        experiment_df = pd.DataFrame()
        experiment_df["model"] = [self.model_config.model_name]
        experiment_df["dataset"] = [self.model_config.dataset_name]
        experiment_df["experiment_name"] = [self.model_config.experiment_name]
        experiment_df["perturbation"] = [self.model_config.perturbation]
        experiment_df["dose"] = [dose]
        all_df = pd.concat([experiment_df, df], axis=1)

        if append_metrics:
            append_csv(all_df, METRICS_PATH)
        return eval_adata

    def __str__(self) -> str:
        return self.__class__.__name__


class ButterflyPipeline(ModelPipeline[SingleConditionDatasetPipeline]):
    def __init__(
        self,
        dataset_pipeline: SingleConditionDatasetPipeline,
        experiment_name: str,
        seed: int = 19193,
        debug=False,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )
        
        self._epoch_pretrain1 = 1 if debug else 100
        self._epoch_pretrain2 = 1 if debug else 100
        self._epoch_intergrative = 1 if debug else 200
        self._num_workers = 0 if debug else 4
        
        self._control, self._perturb = self._get_control_perturb()


    def _split_func(self):
        return unpaired_split_dataset_perturb(
            self.dataset_pipeline.control,
            self.dataset_pipeline.perturb,
        )[0]
        
    def _get_control_perturb(self):
        cell_type_key = self.model_config.cell_type_key        
        control = self.dataset_pipeline.control
        perturb = self.dataset_pipeline.perturb

        control.obs["cell_type"] = control.obs[cell_type_key]
        perturb.obs["cell_type"] = perturb.obs[cell_type_key]
        control.obs.index = [str(i) for i in range(control.X.shape[0])]
        perturb.obs.index = [str(i) for i in range(perturb.X.shape[0])]
        return control, perturb   

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        control, perturb = self._control, self._perturb

        id_list = self._split_func()

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

        file_path = str(self.model_config.get_batch_path(batch))
        tensorboard_path = self.model_config.get_batch_log_path(batch)

        model = self.get_model(
            file_path=file_path,
            tensorboard_path=tensorboard_path)

        if self.model_config.is_finished_batch_training(
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
            translator_epoch=self._epoch_intergrative,
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
            seed=SeedSingleton.get_value(),
            kl_mean=True,
            R_pretrain_kl_warmup=50,
            A_pretrain_kl_warmup=50,
            translation_kl_warmup=50,
            load_model=load_model,
        )
        self.model_config.log_training_batch_is_finished(batch=batch)

        if self.model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return

        input_test, ground_truth_test, predicted_test = model.test(
            test_id_r=test_id_control, test_id_a=test_id_perturb
        )
        return input_test, [ground_truth_test], [predicted_test]
    
    
    def get_model(self, file_path: str, tensorboard_path: Path):
        RNA_input_dim = self._control.X.shape[1]
        ATAC_input_dim = self._perturb.X.shape[1]
        print("RNA_input_dim", RNA_input_dim)
        print("ATAC_input_dim", ATAC_input_dim)     
        return scbutterfly.Model(
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
            RNA_data=self._control,
            ATAC_data=self._perturb,
            tensorboard_path=tensorboard_path,
            num_workers=self._num_workers,
        )        


class ButterflyPipelineNoReusing(ButterflyPipeline):
    def _split_func(self):
        return unpaired_split_dataset_perturb_no_reusing(
            self.dataset_pipeline.control,
            self.dataset_pipeline.perturb,
        )


class ScPreGanPipeline(ModelPipeline[SingleConditionDatasetPipeline]):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SingleConditionDatasetPipeline,
        debug: bool = False,
        seed: int = 19193,
    ):
        self._epochs = 1 if debug else 20_000

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        condition_key = "condition"
        condition = {"case": "stimulated", "control": "control"}
        cell_type_key = self.model_config.cell_type_key

        dataset = self.dataset_pipeline.dataset
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

        output_path = self.model_config.get_batch_path(batch=batch)
        tensorboard_path = self.model_config.get_batch_log_path(batch=batch)

        model = scpregan.Model(
            n_features=n_features,
            n_classes=n_classes,
            use_cuda=torch.cuda.is_available(),
            manual_seed=SeedSingleton.get_value(),
        )

        load_model = self.model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        )

        model.train(
            train_data=train_data,
            output_path=output_path,
            tensorboard_path=tensorboard_path,
            niter=self._epochs,
            load_model=load_model,
        )
        self.model_config.log_training_batch_is_finished(batch=batch)

        if self.model_config.is_finished_evaluation(
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

        return control_test_adata, [perturb_test_adata], [pred_perturbed_adata]


class ScPreGanReproduciblePipeline(ScPreGanPipeline):
    """
    ScPreGan had two different repos:
    - https://github.com/JaneJiayiDong/scPreGAN
    - https://github.com/XiajieWei/scPreGAN-reproducibility/tree/master
    
    This model uses the reproducible one, just for sanity check.
    It is aborted in favor of the main one (`ScPreGanPipeline`).
    """
    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        output_path_reproducible = self.model_config.get_batch_path(batch=batch)

        dataset = self.dataset_pipeline.dataset

        cell_types = dataset.obs[self.model_config.cell_type_key].unique().tolist()
        target_cell_type = cell_types[batch]

        opt = {
            "cuda": torch.cuda.is_available(),
            "dataset": dataset,
            "checkpoint_dir": None,
            "condition_key": "condition",
            "condition": {"case": "stimulated", "control": "control"},
            "cell_type_key": "cell_type",
            "prediction_type": target_cell_type,
            "out_sample_prediction": True,
            "manual_seed": SeedSingleton.get_value(),
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

        cell_type_key = self.model_config.cell_type_key

        pred_perturbed_reproducible_adata = train_and_predict(
            opt=opt,
            config=config,
            tensorboard_path=self.model_config.get_batch_log_path(batch),
        )

        control_adata = self.dataset_pipeline.control
        perturb_adata = self.dataset_pipeline.perturb

        control_test_adata = control_adata[
            control_adata.obs[cell_type_key] == target_cell_type
        ]
        perturb_test_adata = perturb_adata[
            perturb_adata.obs[cell_type_key] == target_cell_type
        ]

        return (
            control_test_adata,
            [perturb_test_adata],
            [pred_perturbed_reproducible_adata],
        )


class ScGenPipeline(ModelPipeline[SingleConditionDatasetPipeline]):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SingleConditionDatasetPipeline,
        debug: bool = False,
        seed: int = 19193,
    ):
        self._epochs = 1 if debug else 100

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        dataset = self.dataset_pipeline.dataset
        cell_type_key = self.model_config.cell_type_key
        cell_types = dataset.obs[cell_type_key].unique().tolist()
        target_cell_type = cell_types[batch]
        train_adata = dataset[
            ~(
                (dataset.obs[cell_type_key] == target_cell_type)
                & (dataset.obs["condition"] == "stimulated")
            )
        ]

        output_path = self.model_config.get_batch_path(batch=batch)

        # must be a bug of scvi-tools, needs the copy to setup anndata
        train_adata = train_adata.copy()
        SCGEN.setup_anndata(
            train_adata, batch_key="condition", labels_key=cell_type_key
        )
        model = SCGEN(adata=train_adata)

        if self.model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        ):
            model = model.load(str(output_path), train_adata)
        else:
            model.train(
                max_epochs=self._epochs,
                batch_size=32,
                early_stopping=True,
                early_stopping_patience=25,
            )
            model.save(str(output_path), overwrite=True)

        self.model_config.log_training_batch_is_finished(batch=batch)

        if self.model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return None

        control_test_adata = self.dataset_pipeline.control
        perturb_test_adata = self.dataset_pipeline.perturb

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

        return control_test_adata, [perturb_test_adata], [pred]


class VidrPipeline(ModelPipeline[T]):
    """
    sources:
    - https://github.com/BhattacharyaLab/scVIDR/blob/main/bin/scvidr_train.py
    """
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: T,
        debug: bool = False,
        is_scgen_variant: bool = False,
        seed: int = 19193,
    ):
        self._epochs = 1 if debug else 100

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )

        self._is_scgen_variant = is_scgen_variant

    @abstractmethod
    def predict(self, model, target_cell_type: str) -> Tuple:
        pass

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        cell_type_key = self.model_config.cell_type_key
        dose_key = self.model_config.dose_key
        cell_types = self.dataset_pipeline.get_cell_types()
        target_cell_type = cell_types[batch]

        train_adata, _ = self.dataset_pipeline.split_dataset_to_train_validation(
            target_cell_type, validation_split=1.0  # vidr uses its own validation
        )
        perturb_test_adata = self.dataset_pipeline.get_stim_test(target_cell_type)
        control_test_adata = self.dataset_pipeline.get_ctrl_test(target_cell_type)

        output_path = self.model_config.get_batch_path(batch=batch)

        train_adata = train_adata.copy()
        vidr.VIDR.setup_anndata(
            train_adata, batch_key=dose_key, labels_key=cell_type_key
        )
        model = vidr.VIDR(adata=train_adata)

        if self.model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        ):
            model = model.load(str(output_path), train_adata)
        else:
            model.train(
                max_epochs=self._epochs,
                batch_size=128,
                early_stopping=True,
                early_stopping_patience=25,
            )
            model.save(str(output_path), overwrite=True)

        self.model_config.log_training_batch_is_finished(batch=batch)

        dosages_to_test = self.dataset_pipeline.get_dosages_unique(perturb_test_adata)
        # todo: just use a file as a flag to log if evaluation is done or not, similar with training
        # todo: this is the same logic with the MultiTaskAaeAutoencoderPipeline
        if self.is_single() and self.model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return None

        multi_finished = True
        if not self.is_single():
            for dose in dosages_to_test:
                multi_finished = (
                    multi_finished
                    and self.model_config.is_finished_evaluation_multi(
                        batch=batch, dosage=dose, refresh=refresh_evaluation
                    )
                )
        if multi_finished:
            return None

        pred, delta, reg = self.predict(model, target_cell_type)

        perturb_test_adata_per_dose = []
        predictions = []
        if not self.is_single():
            pred = cast(dict, pred)
            doses = sorted(pred.keys())
            for dose in doses:
                stim_per_dose = perturb_test_adata[
                    perturb_test_adata.obs[dose_key] == dose
                ]
                predictions.append(pred[dose])
                perturb_test_adata_per_dose.append(stim_per_dose)
        else:
            perturb_test_adata_per_dose.append(perturb_test_adata)
            predictions.append(pred)

        return control_test_adata, perturb_test_adata_per_dose, predictions


class VidrSinglePipeline(VidrPipeline[SingleConditionDatasetPipeline]):
    """
    sources: 
    - https://github.com/BhattacharyaLab/scVIDR/blob/main/bin/scvidr_predict.py
    """
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SingleConditionDatasetPipeline,
        debug: bool = False,
        is_scgen_variant: bool = False,
        seed: int = 19193,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            debug=debug,
            is_scgen_variant=is_scgen_variant,
            seed=seed,
        )

    def predict(self, model: vidr.VIDR, target_cell_type: str):
        pred, delta, reg = model.predict(
            ctrl_key=0.0,
            treat_key=self.dataset_pipeline.dosages,
            cell_type_to_predict=target_cell_type,
            regression=not self._is_scgen_variant,
            continuous=False,
            doses=None,
        )
        return pred, delta, reg


class VidrMultiplePipeline(VidrPipeline[MultipleConditionDatasetPipeline]):
    """
    sources: 
    - https://github.        model = MultiTaskAae(
com/BhattacharyaLab/scVIDR/blob/main/bin/scvidr_predict.py
    """
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: MultipleConditionDatasetPipeline,
        debug: bool = False,
        is_scgen_variant: bool = False,
        seed: int = 19193,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            debug=debug,
            is_scgen_variant=is_scgen_variant,
            seed=seed,
        )

    def get_max_dosage(self):
        return max(self.dataset_pipeline.dosages)

    def predict(self, model: vidr.VIDR, target_cell_type: str):
        pred, delta, reg = model.predict(
            ctrl_key=0.0,
            treat_key=self.get_max_dosage(),
            cell_type_to_predict=target_cell_type,
            regression=not self._is_scgen_variant,
            continuous=True,
            doses=self.dataset_pipeline.dosages, # excluding control dose
        )
        return pred, delta, reg


class MultiTaskAaeAutoencoderPipeline(ModelPipeline):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SplitDatasetPipeline,
        debug: bool = False,
        seed: int = 19193,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )

        self._epochs = 1 if debug else 400
        self._dropout_rate = 0.5
        self._mask_rate = 0.1
        self._hidden_layers_film = []
        self._hidden_layers_autoencoder = [512, 256, 128]
        self._hidden_layers_discriminator = []  # not used
        self._lr = 2.4590236785521603e-05
        self._batch_size = 64

    def _load_model(
        self,
        output_path: Path,
    ):
        num_features = self.dataset_pipeline.get_num_genes()
        condition_len = len(self.dataset_pipeline.get_dosages_unique())        
        film_factory = FilmLayerFactory(
            input_dim=condition_len,
            hidden_layers=self._hidden_layers_film,
            dropout_rate=self._dropout_rate,
        )          
        model = MultiTaskAae.load(
            num_features=num_features,
            hidden_layers_autoencoder=self._hidden_layers_autoencoder,
            hidden_layers_discriminator=self._hidden_layers_discriminator,
            film_layer_factory=film_factory,
            load_path=output_path,
            mask_rate=self._mask_rate,
            dropout_rate=self._dropout_rate,
        )

        return model

    def get_model(
        self,
    ):
        num_features = self.dataset_pipeline.get_num_genes()
        condition_len = len(self.dataset_pipeline.get_dosages_unique())        
        film_factory = FilmLayerFactory(
            input_dim=condition_len,
            hidden_layers=self._hidden_layers_film,
            dropout_rate=self._dropout_rate,
        )        
        model = MultiTaskAae(
            num_features=num_features,
            hidden_layers_autoencoder=self._hidden_layers_autoencoder,
            hidden_layers_discriminator=self._hidden_layers_discriminator,
            film_layer_factory=film_factory,
            mask_rate=self._mask_rate,
            dropout_rate=self._dropout_rate,
        )
        return model

    def _init_trainer(
        self,
        model: MultiTaskAae,
        tensorboard_path: Path,
        target_cell_type: str,
    ):
        trainer = MultiTaskAutoencoderDosagesTrainer.from_pipeline(
            model=model,
            train_tensorboard=tensorboard_path / "train",
            val_tensorboard=tensorboard_path / "val",
            split_dataset_pipeline=self.dataset_pipeline,
            target_cell_type=target_cell_type,
            epochs=self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            seed=SeedSingleton.get_value(),
        )
        return trainer

    def _run(
        self,
        batch: int,
        refresh_training: bool = False,
        refresh_evaluation: bool = False,
    ) -> Optional[Predict]:
        dose_key = self.model_config.dose_key
        cell_types = self.dataset_pipeline.get_cell_types()
        target_cell_type = cell_types[batch]
        output_path = self.model_config.get_batch_path(batch=batch)
        tensorboard_path = self.model_config.get_batch_log_path(batch=batch)

        if self.model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        ):
            model = self._load_model(output_path=output_path)
            if torch.cuda.is_available():
                model = model.to("cuda")
        else:
            model = self.get_model()

        model_utils = MultiTaskAutoencoderUtils(
            split_dataset_pipeline=self.dataset_pipeline,
            target_cell_type=target_cell_type,
            model=model,
        )

        if not self.model_config.is_finished_batch_training(
            batch=batch, refresh=refresh_training
        ):
            trainer = self._init_trainer(
                model=model,
                tensorboard_path=tensorboard_path,
                target_cell_type=target_cell_type,
            )

            model_utils.train(trainer=trainer, save_path=output_path)

        self.model_config.log_training_batch_is_finished(batch=batch)

        stim_test_adata = self.dataset_pipeline.get_stim_test(target_cell_type)
        control_test_adata = self.dataset_pipeline.get_ctrl_test(target_cell_type)
        dosages_to_test = self.dataset_pipeline.get_dosages_unique(stim_test_adata)
        print("dosages_to_test", dosages_to_test)

        # todo: just use a file as a flag to log if evaluation is done or not, similar with training
        if self.is_single() and self.model_config.is_finished_evaluation(
            batch=batch, refresh=refresh_evaluation
        ):
            return None

        multi_finished = True
        if not self.is_single():
            for dose in dosages_to_test:
                multi_finished = (
                    multi_finished
                    and self.model_config.is_finished_evaluation_multi(
                        batch=batch, dosage=dose, refresh=refresh_evaluation
                    )
                )
                print("multi_finished", multi_finished)
        if multi_finished:
            return None

        pred = model_utils.predict()

        perturb_test_adata_per_dose = []
        predictions = []
        for id, dose in enumerate(dosages_to_test):
            stim_per_dose = stim_test_adata[stim_test_adata.obs[dose_key] == dose]
            predictions.append(pred[dose])
            perturb_test_adata_per_dose.append(stim_per_dose)

        return control_test_adata, perturb_test_adata_per_dose, predictions


class MultiTaskAaeAutoencoderOptimalTransportPipeline(MultiTaskAaeAutoencoderPipeline):
    def _init_trainer(
        self,
        model: MultiTaskAae,
        tensorboard_path: Path,
        target_cell_type: str,
    ) -> MultiTaskAutoencoderOptimalTransportTrainer:

        return MultiTaskAutoencoderOptimalTransportTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            train_tensorboard=tensorboard_path / "train",
            val_tensorboard=tensorboard_path / "val",
            epochs=self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            target_cell_type=target_cell_type,
            seed=SeedSingleton.get_value(),
        )


class MultiTaskAaeAdversarialPipeline(MultiTaskAaeAutoencoderPipeline):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SplitDatasetPipeline,
        debug: bool = False,
        seed: int = 19193,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )

        self._autoencoder_pretrain_epochs = 1 if debug else 400
        self._adversarial_epochs = 1 if debug else 100
        self._discriminator_pretrain_epochs = 1 if debug else 400
        self._hidden_layers_discriminator = [32]
        self._coeff_adversarial = 0.01

    def _init_trainer(
        self,
        model: MultiTaskAae,
        tensorboard_path: Path,
        target_cell_type: str,
    ) -> MultiTaskAdversarialTrainer:

        return MultiTaskAdversarialTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            tensorboard_path=tensorboard_path,
            autoencoder_pretrain_epochs=self._autoencoder_pretrain_epochs,
            discriminator_pretrain_epochs=self._discriminator_pretrain_epochs,
            adversarial_epochs=self._adversarial_epochs,
            coeff_adversarial=self._coeff_adversarial,
            lr=self._lr,
            batch_size=self._batch_size,
            target_cell_type=target_cell_type,
            seed=SeedSingleton.get_value(),
        )


class MultiTaskAaeAdversarialGaussianPipeline(MultiTaskAaeAdversarialPipeline):
    def _init_trainer(
        self,
        model: MultiTaskAae,
        tensorboard_path: Path,
        target_cell_type: str,
    ) -> MultiTaskAdversarialGaussianAutoencoderTrainer:

        return MultiTaskAdversarialGaussianAutoencoderTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            tensorboard_path=tensorboard_path,
            autoencoder_pretrain_epochs=self._autoencoder_pretrain_epochs,
            discriminator_pretrain_epochs=self._discriminator_pretrain_epochs,
            adversarial_epochs=self._adversarial_epochs,
            coeff_adversarial=self._coeff_adversarial,
            lr=self._lr,
            batch_size=self._batch_size,
            target_cell_type=target_cell_type,
            seed=SeedSingleton.get_value(),
        )


class MultiTaskAaeAutoencoderAndOptimalTransportPipeline(
    MultiTaskAaeAdversarialPipeline
):
    def _init_trainer(
        self,
        model: MultiTaskAae,
        tensorboard_path: Path,
        target_cell_type: str,
    ) -> MultiTaskAaeAdversarialAndOptimalTransportTrainer:

        return MultiTaskAaeAdversarialAndOptimalTransportTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            tensorboard_path=tensorboard_path,
            autoencoder_pretrain_epochs=self._autoencoder_pretrain_epochs,
            discriminator_pretrain_epochs=self._discriminator_pretrain_epochs,
            adversarial_epochs=self._adversarial_epochs,
            coeff_adversarial=self._coeff_adversarial,
            lr=self._lr,
            batch_size=self._batch_size,
            target_cell_type=target_cell_type,
            seed=SeedSingleton.get_value(),
        )


class MultiTaskVaeAutoencoderPipeline(MultiTaskAaeAutoencoderPipeline):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SplitDatasetPipeline,
        debug: bool = False,
        seed: int = 19193,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
            debug=debug,
        )

        self._beta = 0.004
        self._lr = 6.62135564829619e-06
        self._batch_size = 32

    def get_model(
        self,
    ):
        num_features = self.dataset_pipeline.get_num_genes()
        condition_len = len(self.dataset_pipeline.get_dosages_unique())        
        film_factory = FilmLayerFactory(
            input_dim=condition_len,
            hidden_layers=self._hidden_layers_film,
            dropout_rate=self._dropout_rate,
        )           
        return MultiTaskVae(
            num_features=num_features,
            hidden_layers_autoencoder=self._hidden_layers_autoencoder,
            hidden_layers_discriminator=self._hidden_layers_discriminator,
            film_layer_factory=film_factory,
            mask_rate=self._mask_rate,
            dropout_rate=self._dropout_rate,
            beta=self._beta,
        )

    def _load_model(
        self,
        output_path: Path,
    ):
        num_features = self.dataset_pipeline.get_num_genes()
        condition_len = len(self.dataset_pipeline.get_dosages_unique())        
        film_factory = FilmLayerFactory(
            input_dim=condition_len,
            hidden_layers=self._hidden_layers_film,
            dropout_rate=self._dropout_rate,
        )          
        return MultiTaskVae.load(
            num_features=num_features,
            hidden_layers_autoencoder=self._hidden_layers_autoencoder,
            hidden_layers_discriminator=self._hidden_layers_discriminator,
            film_layer_factory=film_factory,
            mask_rate=self._mask_rate,
            dropout_rate=self._dropout_rate,
            load_path=output_path,
            beta=self._beta,
        )

    def _init_trainer(
        self, model: MultiTaskVae, tensorboard_path: Path, target_cell_type: str
    ):
        return MultiTaskVaeDosagesTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            target_cell_type=target_cell_type,
            train_tensorboard=tensorboard_path / "train",
            val_tensorboard=tensorboard_path / "val",
            epochs=self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            seed=SeedSingleton.get_value(),
        )


class MultiTaskVaeAutoencoderOptimalTransportPipeline(MultiTaskVaeAutoencoderPipeline):
    def _init_trainer(
        self,
        model: MultiTaskVae,
        tensorboard_path: Path,
        target_cell_type: str,
    ) -> MultiTaskVaeOptimalTransportTrainer:

        return MultiTaskVaeOptimalTransportTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            train_tensorboard=tensorboard_path / "train",
            val_tensorboard=tensorboard_path / "val",
            epochs=self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            target_cell_type=target_cell_type,
            seed=SeedSingleton.get_value(),
        )


class MultiTaskVaeAutoencoderAndOptimalTransportPipeline(
    MultiTaskVaeAutoencoderPipeline
):
    def __init__(
        self,
        experiment_name: str,
        dataset_pipeline: SplitDatasetPipeline,
        debug: bool = False,
        seed: int = 19193,
    ):
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            experiment_name=experiment_name,
            seed=seed,
        )

        self._autoencoder_pretrain_epochs = 1 if debug else 400

        # not used
        self._adversarial_epochs = 1
        self._discriminator_pretrain_epochs = 1
        self._coeff_adversarial = 0

    def _init_trainer(
        self,
        model: MultiTaskVae,
        tensorboard_path: Path,
        target_cell_type: str,
    ) -> MultiTaskVaeAdversarialAndOptimalTransportTrainer:

        return MultiTaskVaeAdversarialAndOptimalTransportTrainer.from_pipeline(
            model=model,
            split_dataset_pipeline=self.dataset_pipeline,
            tensorboard_path=tensorboard_path,
            autoencoder_pretrain_epochs=self._autoencoder_pretrain_epochs,
            discriminator_pretrain_epochs=self._discriminator_pretrain_epochs,
            adversarial_epochs=self._adversarial_epochs,
            coeff_adversarial=self._coeff_adversarial,
            lr=self._lr,
            batch_size=self._batch_size,
            target_cell_type=target_cell_type,
            seed=SeedSingleton.get_value(),
        )

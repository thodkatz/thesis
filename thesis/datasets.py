from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
from thesis import DATA_PATH
import scanpy as sc
from thesis.preprocessing import (
    PreprocessingGenericPipeline,
    PreprocessingPipeline,
)
from anndata import AnnData


class DatasetPipeline(ABC):
    def __init__(
        self,
        data_path: Path,
        cell_type_key: str,
        preprocessing_pipeline: Optional[PreprocessingPipeline],
    ):
        dataset = sc.read_h5ad(data_path)
        self.cell_type_key = cell_type_key

        if preprocessing_pipeline is None:
            self.dataset = dataset
        else:
            self.dataset = preprocessing_pipeline(dataset)

    @abstractmethod
    def get_control_perturb(
        self, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class DatasetSinglePerturbationPipeline:
    def __init__(
        self,
        dataset_pipeline: DatasetPipeline,
        perturbation: str = "",
        dosage: float = 0.0,
    ):
        self.dataset_pipeline = dataset_pipeline
        self.perturbation = perturbation
        self.dosage = dosage
        self.control, self.perturb = self.dataset_pipeline.get_control_perturb(
            perturbation=perturbation, dosage=dosage
        )
        self.dataset_perturbation = self.control.concatenate(self.perturb)
        
    @property
    def cell_type_key(self) -> str:
        return self.dataset_pipeline.cell_type_key
    


class PbmcPipeline(DatasetPipeline):
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        pbmc_data = DATA_PATH / "pbmc/pbmc.h5ad"
        cell_type_key = "cell_type"
        super().__init__(
            data_path=pbmc_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
        )

    def get_control_perturb(self, perturbation: str, dosage: float) -> Tuple[AnnData, AnnData]:
        control = self.dataset[self.dataset.obs["condition"] == "control"]
        perturb = self.dataset[self.dataset.obs["condition"] == "stimulated"]
        return control, perturb


class NaultPipeline(DatasetPipeline):
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        nault_data = DATA_PATH / "scvidr/nault2021_multiDose.h5ad"
        cell_type_key = "celltype"
        super().__init__(
            data_path=nault_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=preprocessing_pipeline,
        )

    def get_control_perturb(
        self, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        control = self.dataset[self.dataset.obs["Dose"] == 0]
        assert dosage in self.dataset.obs["Dose"].unique()
        perturb = self.dataset[self.dataset.obs["Dose"] == dosage]
        control.obs["condition"] = "control"
        perturb.obs["condition"] = "stimulated"
        return control, perturb


class Sciplex3Pipeline(DatasetPipeline):
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        sciplex_data = DATA_PATH / "srivatsan_2020_sciplex3.h5ad"
        cell_type_key = "cell_type"
        super().__init__(
            data_path=sciplex_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=preprocessing_pipeline,
        )

    def get_control_perturb(
        self, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        dataset = self.dataset
        dataset = dataset[
            (dataset.obs["perturbation"] == perturbation)
            | (dataset.obs["perturbation"] == "control")
        ]
        dataset = dataset[
            (dataset.obs["dose_value"] == dosage)
            | (dataset.obs["perturbation"] == "control")
        ]
        num_perturbations = dataset.obs.perturbation.unique()
        assert len(num_perturbations) == 2
        dataset.obs["condition"] = dataset.obs["perturbation"].apply(
            lambda x: "control" if x == "control" else "stimulated"
        )
        control = dataset[dataset.obs.condition == "control"]
        perturb = dataset[dataset.obs.condition == "stimulated"]
        return control, perturb

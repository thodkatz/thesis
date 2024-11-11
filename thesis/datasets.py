from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from thesis import DATA_PATH
import scanpy as sc
from thesis.preprocessing import (
    PreprocessingGenericPipeline,
    PreprocessingPipeline,
)
from anndata import AnnData
import anndata as ad


class DatasetPipeline(ABC):
    def __init__(
        self,
        data_path: Path,
        cell_type_key: str,
        dosage_key: str,
        preprocessing_pipeline: Optional[PreprocessingPipeline],
    ):
        dataset = sc.read_h5ad(data_path)
        self.cell_type_key = cell_type_key
        self.dosage_key = dosage_key

        if preprocessing_pipeline is None:
            self.dataset = dataset
        else:
            print("Preprocessing started")
            self.dataset = preprocessing_pipeline(dataset)
            print("Preprocessing finished")

    def __str__(self) -> str:
        return self.__class__.__name__
    
    def get_dosages(self):
        return sorted(self.dataset.obs[self.dosage_key].unique().tolist())


class PbmcPipeline(DatasetPipeline):
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        pbmc_data = DATA_PATH / "pbmc/pbmc.h5ad"
        cell_type_key = "cell_type"
        dose_key = "Dose"
        super().__init__(
            data_path=pbmc_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            dosage_key=dose_key,
        )
        self.dataset.obs[dose_key] = 0


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
            dosage_key="Dose",
        )
        



class NaultLiverTissuePipeline(NaultPipeline):
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        super().__init__(
            preprocessing_pipeline=preprocessing_pipeline,
        )
        liver_cell_types = [
            "Hepatocytes - central",
            "Hepatocytes - portal",
            "Cholangiocytes",
            "Stellate Cells",
            "Portal Fibroblasts",
            "Endothelial Cells",
        ]
        self.dataset = self.dataset[
            self.dataset.obs[self.cell_type_key].isin(liver_cell_types)
        ]


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
            dosage_key="Dose",
        )


class ConditionDatasetPipeline(ABC):
    def __init__(self, dataset_pipeline: DatasetPipeline) -> None:
        self._dataset_pipeline_name = str(dataset_pipeline)
        self.dataset = dataset_pipeline.dataset
        self.cell_type_key = dataset_pipeline.cell_type_key
        self.dosage_key = dataset_pipeline.dosage_key

    @abstractmethod
    def get_train(self, target_cell_type: str) -> AnnData:
        pass

    @abstractmethod
    def get_ctrl_test(self, target_cell_type: str) -> AnnData:
        pass

    @abstractmethod
    def get_stim_test(self, target_cell_type: str) -> AnnData:
        pass

    def is_single(self):
        return isinstance(self, SingleConditionDatasetPipeline)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self._dataset_pipeline_name}"


class SingleConditionDatasetPipeline(ConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: DatasetPipeline,
        perturbation: str,
        dosages: float,
        control: AnnData,
        perturb: AnnData,
    ) -> None:
        super().__init__(dataset_pipeline)
        self.control = control
        self.perturb = perturb
        self.dataset = ad.concat(
            [self.control, self.perturb],
            join="outer",
            label="condition",
            keys=["control", "stimulated"],
            index_unique=None,
        )
        self.perturbation = perturbation
        self.dosages = dosages

    def get_train(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            ~(
                (self.dataset.obs[self.cell_type_key] == target_cell_type)
                & (self.dataset.obs["condition"] == "stimulated")
            )
        ]

    def get_ctrl_test(self, target_cell_type: str) -> AnnData:
        return self.control[(self.control.obs[self.cell_type_key] == target_cell_type)]

    def get_stim_test(self, target_cell_type: str) -> AnnData:
        return self.perturb[(self.perturb.obs[self.cell_type_key] == target_cell_type)]


class MultipleConditionDatasetPipeline(ConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: DatasetPipeline,
        perturbation: str,
        dosages: Optional[List[float]] = None,
    ) -> None:
        super().__init__(dataset_pipeline)
        dose_key = dataset_pipeline.dosage_key
        if dosages is None:
            dosages = dataset_pipeline.get_dosages()
            dosages.remove(0)
        else:
            dosages = dosages
        self.dosages = dosages
        self.perturbation = perturbation


NaultPipelines = Union[NaultPipeline, NaultLiverTissuePipeline]


class NaultSinglePipeline(SingleConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: NaultPipelines,
        dosages: float,
        perturbation: str = "tcdd",
    ) -> None:
        self._control_dose = 0.0
        dose_key = dataset_pipeline.dosage_key

        dataset = dataset_pipeline.dataset
        control = dataset[dataset.obs[dose_key] == 0]
        assert dosages in dataset.obs[dose_key].unique()
        perturb = dataset[dataset.obs[dose_key] == dosages]
        control.obs["condition"] = "control"
        perturb.obs["condition"] = "stimulated"
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            control=control,
            perturb=perturb,
            perturbation=perturbation,
            dosages=dosages,
        )


class NaultMultiplePipeline(MultipleConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: NaultPipelines,
        perturbation: str = "tcdd",
        dosages: Optional[List[float]] = None,
    ) -> None:

        super().__init__(dataset_pipeline, perturbation=perturbation, dosages=dosages)
        self._control_dose = 0.0

    def get_train(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            ~(
                (self.dataset.obs[self.cell_type_key] == target_cell_type)
                & (self.dataset.obs["Dose"] > self._control_dose)
            )
        ]

    def get_ctrl_test(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            (self.dataset.obs[self.cell_type_key] == target_cell_type)
            & (self.dataset.obs["Dose"] == self._control_dose)
        ]

    def get_stim_test(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            (self.dataset.obs[self.cell_type_key] == target_cell_type)
            & (self.dataset.obs["Dose"] > self._control_dose)
        ]


class PbmcSinglePipeline(SingleConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: PbmcPipeline,
        perturbation: str = "ifn-b",
        dosages: float = -1.0,
    ) -> None:
        dataset = dataset_pipeline.dataset
        control = dataset[dataset.obs["condition"] == "control"]
        perturb = dataset[dataset.obs["condition"] == "stimulated"]
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            control=control,
            perturb=perturb,
            dosages=-1.0,
            perturbation=perturbation,
        )

        # hack to make vidr work with non dosages datasets for single condition experiments
        dose_key = dataset_pipeline.dosage_key
        self.control.obs[dose_key] = 0.0
        self.perturb.obs[dose_key] = -1.0
        self.dataset = ad.concat(
            [self.control, self.perturb],
            join="outer",
            label="batch",
            keys=["control", "stimulated"],
            index_unique=None,
        )


class Sciplex3SinglePipeline(SingleConditionDatasetPipeline):
    def __init__(
        self, dataset_pipeline: Sciplex3Pipeline, perturbation: str, dosages: float
    ) -> None:
        dataset = dataset_pipeline.dataset
        dataset = dataset[
            (dataset.obs["perturbation"] == perturbation)
            | (dataset.obs["perturbation"] == "control")
        ]
        dataset = dataset[
            (dataset.obs["dose_value"] == dosages)
            | (dataset.obs["perturbation"] == "control")
        ]
        num_perturbations = dataset.obs.perturbation.unique()
        assert len(num_perturbations) == 2
        dataset.obs["condition"] = dataset.obs["perturbation"].apply(
            lambda x: "control" if x == "control" else "stimulated"
        )
        control = dataset[dataset.obs.condition == "control"]
        perturb = dataset[dataset.obs.condition == "stimulated"]
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            control=control,
            perturb=perturb,
            perturbation=perturbation,
            dosages=dosages,
        )

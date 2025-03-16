from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

from thesis import DATA_PATH
import scanpy as sc
from thesis.preprocessing import (
    PreprocessingGenericPipeline,
    PreprocessingPipeline,
)
from anndata import AnnData
import anndata as ad

Train = AnnData
Validation = AnnData


class DatasetPipeline(ABC):
    def __init__(
        self,
        cell_type_key: str,
        dosage_key: str,
        data_path: Union[Path, AnnData],
        preprocessing_pipeline: Optional[PreprocessingPipeline],
    ):
        if isinstance(data_path, Path):
            dataset = sc.read_h5ad(data_path)
        else:
            dataset = data_path
        self.cell_type_key = cell_type_key
        self.dosage_key = dosage_key
        self.control_dose = 0.0

        if preprocessing_pipeline is None:
            self.dataset = dataset
        else:
            print("Preprocessing started")
            self.dataset = preprocessing_pipeline(dataset)
            print("Preprocessing finished")

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_dosages_unique(self, adata: Optional[AnnData] = None):
        if adata is None:
            return sorted(self.dataset.obs[self.dosage_key].unique().tolist())
        else:
            return sorted(adata.obs[self.dosage_key].unique().tolist())

    def get_cell_types(self):
        return sorted(self.dataset.obs[self.cell_type_key].unique().tolist())

    def get_num_genes(self):
        return self.dataset.shape[1]


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


class SplitDatasetPipeline(ABC):
    def __init__(self, dataset_pipeline: DatasetPipeline) -> None:
        self._dataset_pipeline_name = str(dataset_pipeline)
        self.dataset_pipeline = dataset_pipeline
        self.cell_type_key = dataset_pipeline.cell_type_key
        self.dosage_key = dataset_pipeline.dosage_key
        self.control_dose = dataset_pipeline.control_dose

    @property
    def dataset(self) -> AnnData:
        return self.dataset_pipeline.dataset

    def split_dataset_to_train_validation(
        self, target_cell_type: str, validation_split: float = 0.8
    ) -> Tuple[AnnData, Optional[AnnData]]:
        """
        Splits the dataset into training and validation subsets based on cell types and dosages.

        Args:
            target_cell_type (str): Target cell type to split.
            validation_split (float): Proportion of data to use for training (default: 0.8).

        Returns:
            Tuple[AnnData, AnnData]: Training and validation AnnData objects.
        """
        adata = self._get_dataset_to_split(target_cell_type=target_cell_type)
        train_ilocs = []
        valid_ilocs = []
        if validation_split == 1.0:
            return adata, None

        for cell_type in self.get_cell_types():
            dataset_cell_type = adata[adata.obs[self.cell_type_key] == cell_type]
            for dose in self.get_dosages_unique(dataset_cell_type):
                dataset_dose = dataset_cell_type[
                    dataset_cell_type.obs[self.dosage_key] == dose
                ]
                split_idx = int(len(dataset_dose) * validation_split)
                train_ilocs.extend(dataset_dose.obs.index[:split_idx])
                valid_ilocs.extend(dataset_dose.obs.index[split_idx:])
        return adata[train_ilocs], adata[valid_ilocs]

    @abstractmethod
    def _get_dataset_to_split(self, target_cell_type: str) -> AnnData:
        pass

    @abstractmethod
    def get_ctrl_test(self, target_cell_type: str) -> AnnData:
        pass

    @abstractmethod
    def get_stim_test(self, target_cell_type: str) -> AnnData:
        pass

    def is_single(self):
        return isinstance(self, SingleConditionDatasetPipeline)

    def get_dosages_unique(self, adata: Optional[AnnData] = None):
        return self.dataset_pipeline.get_dosages_unique(adata=adata)

    def get_cell_types(self):
        return self.dataset_pipeline.get_cell_types()

    def get_num_genes(self):
        return self.dataset_pipeline.get_num_genes()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self._dataset_pipeline_name}"


class SingleConditionDatasetPipeline(SplitDatasetPipeline):
    """
    This class serves the need to have an interface of data being split to control and perturb used by scbutterfly, scgen, scpregan.
    """

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
        self.dataset_pipeline.dataset = ad.concat(
            [self.control, self.perturb],
            join="outer",
            label="condition",
            keys=["control", "stimulated"],
            index_unique=None,
        )
        self.dataset_pipeline.dataset.var = self.control.var.copy()
        self.perturbation = perturbation
        self.dosages = dosages

    def _get_dataset_to_split(self, target_cell_type: str) -> AnnData:
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


class MultipleConditionDatasetPipeline(SplitDatasetPipeline):
    """
    Utility to split dataset based on a subset of dosages
    """

    def __init__(
        self,
        dataset_pipeline: DatasetPipeline,
        perturbation: str,
        dosages: Optional[List[float]] = None,
    ) -> None:
        super().__init__(dataset_pipeline)
        if dosages is None:
            dosages = dataset_pipeline.get_dosages_unique()
            dosages.remove(0)
        else:
            dosages = sorted(dosages)
            dosages_to_filter = deepcopy(dosages)
            dosages_to_filter.append(self.control_dose)
            self.dataset_pipeline.dataset = self.dataset_pipeline.dataset[
                self.dataset_pipeline.dataset.obs[self.dosage_key].isin(
                    dosages_to_filter
                )
            ]
        self.dosages = dosages
        self.perturbation = perturbation

    def _get_dataset_to_split(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            ~(
                (self.dataset.obs[self.cell_type_key] == target_cell_type)
                & (self.dataset.obs[self.dosage_key] != self.control_dose)
            )
        ]

    def get_ctrl_test(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            (self.dataset.obs[self.cell_type_key] == target_cell_type)
            & (self.dataset.obs[self.dosage_key] == self.control_dose)
        ]

    def get_stim_test(self, target_cell_type: str) -> AnnData:
        return self.dataset[
            (self.dataset.obs[self.cell_type_key] == target_cell_type)
            & (self.dataset.obs[self.dosage_key] != self.control_dose)
        ]


NaultPipelines = Union[NaultPipeline, NaultLiverTissuePipeline]


class NaultSinglePipeline(SingleConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: NaultPipelines,
        dosages: float,
        perturbation: str = "tcdd",
    ) -> None:
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


class PbmcSinglePipeline(SingleConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: PbmcPipeline,
        perturbation: str = "ifn-b",
        dosages: float = -1.0,  # fix: not used, just to have a consistent interface with other pipelines
    ) -> None:
        dataset = dataset_pipeline.dataset
        control = dataset[dataset.obs["condition"] == "control"]
        perturb = dataset[dataset.obs["condition"] == "stimulated"]

        # hack to make dosages based models (e.g. vidr) work with non dosages datasets for single condition experiments
        dose_key = dataset_pipeline.dosage_key
        control.obs[dose_key] = 0.0
        perturb.obs[dose_key] = -1.0

        super().__init__(
            dataset_pipeline=dataset_pipeline,
            control=control,
            perturb=perturb,
            dosages=-1.0,
            perturbation=perturbation,
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

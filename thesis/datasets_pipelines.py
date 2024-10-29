from abc import ABC, abstractmethod
from typing import Optional, Tuple
from thesis import DATA_PATH
import scanpy as sc
from thesis.preprocessing_pipelines import (
    PreprocessingGenericPipeline,
    PreprocessingNoFilteringPipeline,
    PreprocessingPipeline,
)
from anndata import AnnData


class SplitControlPerturbPipeline(ABC):
    @abstractmethod
    def __call__(
        self, dataset: AnnData, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        pass


class PbmcSplitControlPerturbPipeline(SplitControlPerturbPipeline):
    def __call__(
        self, dataset: AnnData, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        control = dataset[dataset.obs["condition"] == "control"]
        perturb = dataset[dataset.obs["condition"] == "stimulated"]
        return control, perturb


class Sciplex3SplitControlPerturbPipeline(SplitControlPerturbPipeline):
    def __call__(
        self, dataset: AnnData, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        num_perturbations = dataset.obs.perturbation.unique()
        assert len(num_perturbations) == 2
        dataset.obs["condition"] = dataset.obs["perturbation"].apply(
            lambda x: "control" if x == "control" else "stimulated"
        )
        control = dataset[dataset.obs.condition == "control"]
        perturb = dataset[dataset.obs.condition == "stimulated"]
        return control, perturb


class NaultSplitControlPerturbPipeline(SplitControlPerturbPipeline):
    def __call__(
        self, dataset: AnnData, perturbation: str, dosage: float
    ) -> Tuple[AnnData, AnnData]:
        control = dataset[dataset.obs["Dose"] == 0]
        assert dosage in dataset.obs["Dose"].unique()
        perturb = dataset[dataset.obs["Dose"] == dosage]
        control.obs["condition"] = "control"
        perturb.obs["condition"] = "stimulated"
        return control, perturb


class DatasetPipeline(ABC):
    def __init__(
        self,
        dataset: AnnData,
        perturbation: str,
        dosage: float,
        cell_type_key: str,
        preprocessing_pipeline: Optional[PreprocessingPipeline],
        split_control_perturb_pipeline: SplitControlPerturbPipeline,
    ):
        self.perturbation = perturbation
        self.dosage = dosage
        self.cell_type_key = cell_type_key
        
        if preprocessing_pipeline is None:
            dataset = dataset
        else:
            dataset = preprocessing_pipeline(dataset)
        
        self.control, self.perturb = split_control_perturb_pipeline(
            dataset, self.perturbation, self.dosage
        )
        
        self.dataset = self.control.concatenate(self.perturb)



    def __str__(self) -> str:
        return self.__class__.__name__


class PbmcPipeline(DatasetPipeline):
    def __init__(self):
        pbmc_data = DATA_PATH / "pbmc/pbmc.h5ad"
        dataset = sc.read_h5ad(pbmc_data)
        perturbation = "ifn-b"
        dosage = 0
        cell_type_key = "cell_type"
        super().__init__(
            dataset=dataset,
            perturbation=perturbation,
            dosage=dosage,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            split_control_perturb_pipeline=PbmcSplitControlPerturbPipeline(),
        )


class NaultPipeline(DatasetPipeline):
    def __init__(
        self,
        dosage: float,
        preprocessing: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        nault_data = DATA_PATH / "scvidr/nault2021_multiDose.h5ad"
        dataset = sc.read_h5ad(nault_data)
        cell_type_key = "celltype"
        super().__init__(
            dataset=dataset,
            perturbation="tcdd",
            dosage=dosage,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=preprocessing,
            split_control_perturb_pipeline=NaultSplitControlPerturbPipeline(),
        )


class NaultNoFilteringPipeline(NaultPipeline):
    def __init__(self, dosage: float):
        super().__init__(
            dosage=dosage,
            preprocessing=PreprocessingNoFilteringPipeline(),
        )


class Sciplex3Pipeline(DatasetPipeline):
    def __init__(self, perturbation: str, dosage: float):
        sciplex_data = DATA_PATH / "srivatsan_2020_sciplex3.h5ad"
        dataset = sc.read_h5ad(sciplex_data)
        dataset = dataset[
            (dataset.obs["perturbation"] == perturbation)
            | (dataset.obs["perturbation"] == "control")
        ]
        dataset = dataset[
            (dataset.obs["dose_value"] == dosage)
            | (dataset.obs["perturbation"] == "control")
        ]

        cell_type_key = "cell_type"
        super().__init__(
            dataset=dataset,
            perturbation=perturbation,
            dosage=dosage,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            split_control_perturb_pipeline=Sciplex3SplitControlPerturbPipeline(),
        )

from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from thesis import DATA_PATH
import scanpy as sc
from thesis.preprocessing import (
    PreprocessingGenericPipeline,
    PreprocessingPipeline,
    PreprocessingFilteringPipeline,
    PreprocessingNoFilteringPipeline,
)
from anndata import AnnData
import anndata as ad

Train = AnnData
Validation = AnnData

"""
TODO:
- Refactor the classes of the datasets. Classes such as PbmcPipeline, NaultPipeline, and Sciplex3Pipeline should be attained via classmethods of DatasetPipeline.
"""


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
    """
    preprocessed data from https://github.com/theislab/scgen-re producibility/blob/master/code/DataDownloader.py (Lotfollahi et al., 2019b)
    """

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
    """    
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184506
    """

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

class NaultGSE148339Pipeline(DatasetPipeline):
    """    
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148339
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        nault_data = DATA_PATH / "nault2_single"
        veh62 = sc.read_10x_mtx(
            nault_data, var_names="gene_symbols", make_unique=True, prefix="GSM4460588_VEH62_"
        )
                
        veh64 = sc.read_10x_mtx(
            nault_data, var_names="gene_symbols", make_unique=True, prefix="GSM4460589_VEH64_"
        )
          
        tcdd51 = sc.read_10x_mtx(
            nault_data, var_names="gene_symbols", make_unique=True, prefix="GSM4460590_TCDD51_"
        )
          
        tcdd59 = sc.read_10x_mtx(
            nault_data, var_names="gene_symbols", make_unique=True, prefix="GSM4460591_TCDD59_"
        )
              
        cell_type_key = "celltype"
        
        super().__init__(
            data_path=nault_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=preprocessing_pipeline,
            dosage_key="Dose",
        )


class Nault10xPipeline(DatasetPipeline):
    """
    Single-cell transcriptomics shows dose-dependent disruption of hepatic zonation by TCDD in mice

    Notes:
    - Already log transformed.

    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184506
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        nault_data = DATA_PATH / "nault2"
        cell_type_key = "celltype"
        dataset = sc.read_10x_mtx(
            nault_data, var_names="gene_symbols", make_unique=True
        )
        dataset.raw = dataset
        meta = pd.read_csv(
            f"{nault_data}/DR-metadata_updated.tsv", sep="\t", index_col=0
        )
        dataset.obs = meta.loc[dataset.obs_names]  # match and align
        super().__init__(
            data_path=dataset,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            dosage_key="Dose",
        )


class Nault10xRawPipeline(DatasetPipeline):
    """
    Single-cell transcriptomics shows dose-dependent disruption of hepatic zonation by TCDD in mice

    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184506
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        nault_data_raw = DATA_PATH / "nault2raw"
        nault_data = DATA_PATH / "nault2"
        cell_type_key = "celltype"
        read_adata = (nault_data_raw / "nault2raw.h5ad").exists()
        read_adata = False
        if read_adata:
            dataset = sc.read_h5ad(nault_data_raw / "nault2raw.h5ad")
        else:
            # Note:
            # genes.tsv contains only one column with the gene names
            # the read_10x_mtx requires two columns gene_ids, gene_names
            # we have bypassed the gene_ids by modifying the source code just for the loading
            dataset = sc.read_10x_mtx(
                nault_data_raw, var_names="gene_symbols", make_unique=True
            )
            
            meta = pd.read_csv(
                f"{nault_data}/DR-metadata_updated.tsv", sep="\t", index_col=0
            )
            meta = meta[["dose", "cell_type", "cell_type_treatment", "cell_type_dose"]]
            
            ontology_to_celltype = {
                "CL_0000632": "Stellate Cells",
                "CL_0019026": "Hepatocytes - portal",
                "CL_1000398": "Endothelial Cells",
                "CL_1000488": "Cholangiocytes",
                "CL_0000235": "Macrophage",
                "CL_0019029": "Hepatocytes - central",
                "CL_0000236": "B Cells",
                "CL_0000084": "T Cells",
                "CL_0009100": "Portal Fibroblasts",
                "CL_0000775": "Neutrophils",
                "CL_2000055": "Subtype 1"
            }
            
            dataset.obs = meta.loc[dataset.obs_names]
            dataset.obs["dose"] = dataset.obs["dose"].astype(str)
            dataset.obs["cell_type"] = dataset.obs["cell_type"].astype(str)
            dataset.obs["cell_type"] = dataset.obs["cell_type"].map(ontology_to_celltype)
            dataset.obs["cell_type_treatment"] = dataset.obs["cell_type_treatment"].astype(str)
            dataset.obs["cell_type_dose"] = dataset.obs["cell_type_dose"].astype(str)
            dataset.obs_names = dataset.obs_names.astype(str)
            
            dataset.var["gene_ids"] = dataset.var["gene_ids"].astype(str)
            dataset.var_names = dataset.var_names.astype(str)
            
            dataset.obs["celltype"] = dataset.obs["cell_type"]
            dataset.obs["Dose"] = dataset.obs["dose"]
            dataset.obs["Dose"] = dataset.obs["Dose"].astype(float)           
            
            dataset.write(nault_data_raw / "nault2raw.h5ad")
        super().__init__(
            data_path=dataset,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=preprocessing_pipeline,
            dosage_key="Dose",
        )


class NaultCrossStudyPipeline(DatasetPipeline):
    """
    Followed scgen approach when performed cross study analysis with pbmc and zheng

    - Filter the cells from datasets
    - Log tranform
    - pick the high variable ones
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        dataset_1_pipeline = NaultPipeline(
            preprocessing_pipeline=PreprocessingFilteringPipeline()
        )
        dataset_2_pipeline = Nault10xRawPipeline(
            preprocessing_pipeline=PreprocessingFilteringPipeline()
        )

        dataset_1_pipeline.dataset.obs["study"] = "nault1"
        dataset_2_pipeline.dataset.obs["study"] = "nault2"
        dataset = ad.concat(
            [dataset_1_pipeline.dataset, dataset_2_pipeline.dataset],
            join="outer",
            index_unique=None,
        )
        cell_type_key = "celltype"
        super().__init__(
            data_path=dataset,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=PreprocessingNoFilteringPipeline(),
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


class CrossStudyPipeline(DatasetPipeline):
    """
    preprocessed data from https://github.com/theislab/scgen-re producibility/blob/master/code/DataDownloader.py (Lotfollahi et al., 2019b)
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        cross_study_data = DATA_PATH / "cross_study" / "cross_study.h5ad"
        cell_type_key = "cell_type"
        dose_key = "Dose"
        super().__init__(
            data_path=cross_study_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            dosage_key=dose_key,
        )
        self.dataset.obs[dose_key] = 0


class CrossSpeciesPipeline(DatasetPipeline):
    """
    preprocessed data from https://github.com/theislab/scgen-re producibility/blob/master/code/DataDownloader.py (Lotfollahi et al., 2019b)
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        cross_species_data = DATA_PATH / "cross_species" / "cross_species.h5ad"
        cell_type_key = "species"
        dose_key = "Dose"
        super().__init__(
            data_path=cross_species_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            dosage_key=dose_key,
        )
        self.dataset.obs[dose_key] = 0


class ZhengPipeline(DatasetPipeline):
    """
    preprocessed data from https://github.com/theislab/scgen-re producibility/blob/master/code/DataDownloader.py (Lotfollahi et al., 2019b)
    """

    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingGenericPipeline(),
    ):
        pbmc_data = DATA_PATH / "cross_study" / "train_zheng.h5ad"
        cell_type_key = "cell_type"
        dose_key = "Dose"
        super().__init__(
            data_path=pbmc_data,
            cell_type_key=cell_type_key,
            preprocessing_pipeline=None,
            dosage_key=dose_key,
        )
        self.dataset.obs[dose_key] = 0


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
        control.obs["condition"] = "control"
        perturb.obs["condition"] = "stimulated"
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
        dataset_pipeline: Union[PbmcPipeline, CrossStudyPipeline, CrossSpeciesPipeline],
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


class CrossSpeciesConditionPipeline(SingleConditionDatasetPipeline):
    def __init__(
        self,
        dataset_pipeline: CrossSpeciesPipeline,
        perturbation: str = "lps",
        dosages: float = -1.0,  # fix: not used, just to have a consistent interface with other pipelines
    ) -> None:
        dataset = dataset_pipeline.dataset
        control = dataset[dataset.obs["condition"] == "unst"]
        perturb = dataset[dataset.obs["condition"] == "LPS6"]

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


class CrossStudyConditionPipeline(PbmcSinglePipeline):
    def __init__(
        self,
        dataset_pipeline: CrossStudyPipeline,
        perturbation: str = "ifn-b",
        dosages: float = -1.0,  # fix: not used, just to have a consistent interface with other pipelines
    ) -> None:

        self._dataset_ctrl_test = ZhengPipeline()
        super().__init__(
            dataset_pipeline=dataset_pipeline,
            dosages=-1.0,
            perturbation=perturbation,
        )

    def get_ctrl_test(self, target_cell_type: str) -> AnnData:
        dataset = self._dataset_ctrl_test.dataset
        return dataset[dataset.obs["cell_type"] == target_cell_type]


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

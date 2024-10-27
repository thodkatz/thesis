from thesis import ROOT
import scanpy as sc
from thesis.preprocessing import preprocess_sciplex3
from anndata import AnnData

data = ROOT / "data"


def get_nault_multi_dose():
    # dataset downloaded from scVIDR github provided drive link
    nault_data = data / "scvidr/nault2021_multiDose.h5ad"
    return sc.read_h5ad(nault_data)


def get_pbmc():
    nault_data = data / "pbmc/pbmc.h5ad"
    return sc.read_h5ad(nault_data)


def get_sciplex3():
    sciplex_data = data / "srivatsan_2020_sciplex3.h5ad"
    return sc.read_h5ad(sciplex_data)


def get_sciplex3_per_perturbation(perturbation_type: str, drug_dosage: int):
    sciplex = get_sciplex3()
    sciplex = sciplex[
        (sciplex.obs["perturbation"] == perturbation_type)
        | (sciplex.obs["perturbation"] == "control")
    ]
    sciplex = sciplex[
        (sciplex.obs["dose_value"] == drug_dosage)
        | (sciplex.obs["perturbation"] == "control")
    ]
    return preprocess_sciplex3(sciplex)


def get_control_perturb_pbmc(dataset: AnnData):
    control = dataset[dataset.obs["condition"] == "control"]
    perturb = dataset[dataset.obs["condition"] == "stimulated"]
    return control, perturb


def get_control_perturb_nault(dataset: AnnData, drug_dosage: int):
    control = dataset[dataset.obs["Dose"] == 0]
    assert drug_dosage in dataset.obs["Dose"].unique()
    perturb = dataset[dataset.obs["Dose"] == drug_dosage]
    control.obs["condition"] = "control"
    perturb.obs["condition"] = "stimulated"
    return control, perturb


def get_control_perturb_sciplex3(dataset: AnnData):
    num_perturbations = dataset.obs.perturbation.unique()
    assert len(num_perturbations) == 2
    dataset.obs["condition"] = dataset.obs["perturbation"].apply(
        lambda x: "control" if x == "control" else "stimulated"
    )
    control = dataset[dataset.obs.condition == "control"]
    perturb = dataset[dataset.obs.condition == "stimulated"]
    return control, perturb

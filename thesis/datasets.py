from matplotlib.pyplot import sci
from thesis import ROOT
import scanpy as sc
from thesis.preprocessing import preprocess_sciplex3

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

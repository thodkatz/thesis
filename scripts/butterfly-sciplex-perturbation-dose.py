from thesis.preprocessing import preprocess_sciplex3
from thesis import ROOT
from thesis.scbutterfly import run_sciplex3, run_sciplex3_no_reusing
import scanpy as sc

data = ROOT / "data"
sciplex_data = data / "srivatsan_2020_sciplex3.h5ad"
sciplex3 = sc.read_h5ad(sciplex_data)
sciplex3 = preprocess_sciplex3(sciplex3)

dataset1: sc.AnnData = sciplex3[
    (sciplex3.obs["perturbation"] == "Ellagic acid")
    | (sciplex3.obs["perturbation"] == "control")
]
dataset1 = dataset1[
    (dataset1.obs["dose_value"] == 10000) | (dataset1.obs["perturbation"] == "control")
]

dataset2: sc.AnnData = sciplex3[
    (sciplex3.obs["perturbation"] == "UNC1999")
    | (sciplex3.obs["perturbation"] == "control")
]
dataset2 = dataset2[
    (dataset2.obs["dose_value"] == 1000) | (dataset2.obs["perturbation"] == "control")
]

dataset3: sc.AnnData = sciplex3[
    (sciplex3.obs["perturbation"] == "Ellagic acid")
    | (sciplex3.obs["perturbation"] == "control")
]
dataset3 = dataset3[
    (dataset3.obs["dose_value"] == 1000) | (dataset3.obs["perturbation"] == "control")
]


def main():
    # run_sciplex3(name="ellagic_acid_dose_10000", dataset=dataset1)
    run_sciplex3(name="unc1999_dose_1000", dataset=dataset2)
    run_sciplex3(name="ellagic_acid_dose_1000", dataset=dataset3)


def main_no_reusing():
    run_sciplex3_no_reusing(name="ellagic_acid_dose_10000_no_reusing", dataset=dataset1)
    run_sciplex3_no_reusing(name="unc1999_dose_1000_no_reusing", dataset=dataset2)
    run_sciplex3_no_reusing(name="ellagic_acid_dose_1000_no_reusing", dataset=dataset3)


if __name__ == "__main__":
    main_no_reusing()

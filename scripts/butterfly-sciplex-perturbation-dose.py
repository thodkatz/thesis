from thesis.preprocessing import preprocess_sciplex3
from thesis import ROOT
from thesis.scbutterfly import run_sciplex3
import scanpy as sc

def main():
    data = ROOT / "data"
    sciplex_data = data / "srivatsan_2020_sciplex3.h5ad"
    sciplex3 = sc.read_h5ad(sciplex_data)
    sciplex3 = preprocess_sciplex3(sciplex3)
        
    dataset1: sc.AnnData = sciplex3[(sciplex3.obs['perturbation'] == 'Ellagic acid') | (sciplex3.obs['perturbation'] == 'control')]
    dataset2 = sciplex3[(sciplex3.obs['perturbation'] == 'Divalproex Sodium') | (sciplex3.obs['perturbation'] == 'control')]
    dataset1 = dataset1[(dataset1.obs['dose_value'] == 10000) | (dataset1.obs['perturbation'] == 'control')]
    dataset2 = dataset2[(dataset2.obs['dose_value'] == 1000) | (dataset2.obs['perturbation'] == 'control')]
    

    run_sciplex3(name="ellagic_acid_dose_10000", dataset=dataset1)
    run_sciplex3(name="dival_sodium_dose_1000", dataset=dataset2)
    
if __name__ == "__main__":
    main()
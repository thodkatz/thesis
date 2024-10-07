from thesis.preprocessing import preprocess_sciplex3
from thesis import ROOT
from thesis.scbutterfly import run_sciplex3
import scanpy as sc

def main():
    data = ROOT / "data"
    sciplex3 = sc.read_h5ad(data / "srivatsan_2020_sciplex3.h5ad")
    sciplex3 = preprocess_sciplex3(sciplex3)
        
    dataset1 = sciplex3[(sciplex3.obs['perturbation'] == 'Ellagic acid') | (sciplex3.obs['perturbation'] == 'control')]
    dataset2 = sciplex3[(sciplex3.obs['perturbation'] == 'Divalproex Sodium') | (sciplex3.obs['perturbation'] == 'control')]

    run_sciplex3("ellagic_acid", dataset1)
    run_sciplex3("dival_sodium", dataset2)
 
 
if __name__ == "__main__":
    main()
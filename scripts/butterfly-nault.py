import thesis.scbutterfly as scbutterfly
from thesis.preprocessing import preprocess_nault
from thesis.datasets import get_nault_multi_dose
import argparse
import torch

if __name__ == "__main__":
    dataset = get_nault_multi_dose()
    dataset = preprocess_nault(dataset)

    parser = argparse.ArgumentParser(description="Run pbmc")
    parser.add_argument("--dosage", type=int, required=False, help="drug dosage")
    parser.add_argument("--batch", type=int, required=False, help="batch id")
    args = parser.parse_args()
    print("CUDA", torch.cuda.is_available())
    
    scbutterfly.run_nault_dosage(experiment_name="", dataset=dataset, batch=args.batch, dosage=args.dosage)

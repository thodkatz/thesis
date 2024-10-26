import thesis.scbutterfly as scbutterfly
from thesis.datasets import get_pbmc
import argparse
import torch

if __name__ == "__main__":
    dataset = get_pbmc()
    name = "pbmc"
    
    parser = argparse.ArgumentParser(description="Run pbmc")
    parser.add_argument('--batch', type=int, required=False, help="batch id")
    args = parser.parse_args()
    print("CUDA", torch.cuda.is_available())
    scbutterfly.run_pbmc(experiment_name=name, dataset=dataset, batch=args.batch)

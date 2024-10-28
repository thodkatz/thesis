from thesis.model_pipelines import ButterflyPipeline
from thesis.datasets_pipelines import NaultNoFilteringPipeline
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pbmc")
    parser.add_argument("--dosage", type=int, required=False, help="drug dosage")
    parser.add_argument("--batch", type=int, required=False, help="batch id")
    args = parser.parse_args()
    print("CUDA", torch.cuda.is_available())

    butterfly_pmbc = ButterflyPipeline(
        dataset_pipeline=NaultNoFilteringPipeline(
            perturbation=args.perturbation, dosage=args.dosage
        ),
        experiment_name="",
    )
    butterfly_pmbc(batch=args.batch, append_metrics=True, save_plots=False)

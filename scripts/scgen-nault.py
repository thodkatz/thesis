from thesis.model_pipelines import ScGenPipeline
from thesis.datasets_pipelines import NaultPipeline
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scbutterfly nault")
    parser.add_argument("--dosage", type=float, required=False, help="drug dosage")
    parser.add_argument("--batch", type=int, required=False, help="batch id")
    args = parser.parse_args()
    print("CUDA", torch.cuda.is_available())


    scgen_pmbc = ScGenPipeline(
        dataset_pipeline=NaultPipeline(
            dosage=args.dosage
        ),
        experiment_name="",
    )
    scgen_pmbc(batch=args.batch, append_metrics=True, save_plots=False)

import argparse
import torch
from thesis.datasets_pipelines import NaultPipeline
from thesis.model_pipelines import ScPreGanPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scpregan nault")
    parser.add_argument("--dosage", type=float, required=False, help="drug dosage")
    parser.add_argument("--batch", type=int, required=False, help="batch id")
    args = parser.parse_args()
    print("CUDA", torch.cuda.is_available())

    scpregan_nault = ScPreGanPipeline(
        dataset_pipeline=NaultPipeline(
            dosage=args.dosage
        ),
        experiment_name="",
    )
    scpregan_nault(batch=args.batch, append_metrics=True, save_plots=False)

import argparse
import torch
from thesis.datasets_pipelines import PbmcPipeline
from thesis.model_pipelines import ScPreGanPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pbmc")
    parser.add_argument("--batch", type=int, required=False, help="batch id")
    args = parser.parse_args()
    print("CUDA", torch.cuda.is_available())

    scpregan_pbmc = ScPreGanPipeline(
        dataset_pipeline=PbmcPipeline(), experiment_name=""
    )
    scpregan_pbmc(batch=args.batch, append_metrics=True, save_plots=False)

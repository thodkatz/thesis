from thesis.model import (
    ButterflyPipeline,
    ButterflyPipelineNoReusing,
    ScGenPipeline,
    ScPreGanPipeline,
    ScPreGanReproduciblePipeline,
    VidrMultiplePipeline,
    VidrSinglePipeline,
)
from thesis.datasets import (
    NaultLiverTissuePipeline,
    NaultMultiplePipeline,
    NaultPipeline,
    NaultSinglePipeline,
    PbmcPipeline,
    PbmcSinglePipeline,
    Sciplex3Pipeline,
    Sciplex3SinglePipeline,
)
import argparse
import torch

from thesis.preprocessing import (
    PreprocessingGenericPipeline,
    PreprocessingNoFilteringPipeline,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scbutterfly nault")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--experiment", type=str, required=False, help="experiment name"
    )
    parser.add_argument("--perturbation", type=str, required=True, help="perturbation")
    parser.add_argument("--dosages", type=float, required=False, help="drug dosage")
    parser.add_argument("--batch", type=int, required=True, help="batch id")
    parser.add_argument(
        "--model",
        choices=[
            "scgen",
            "scbutterfly",
            "scbutterfly-no-reusing",
            "scpregan",
            "scpregan-reproducible",
            "vidr-single",
            "vidr-multi",
        ],
        help="Chooose model",
    )
    parser.add_argument(
        "--preprocessing",
        choices=["default", "no-filtering"],
        default="default",
        help="Chooose preprocessing",
    )
    parser.add_argument(
        "--dataset",
        choices=["pbmc", "nault", "sciplex3", "nault-multi", "nault-liver", "nault-liver-multi"],
        help="Chooose dataset",
    )
    args = parser.parse_args()

    print("CUDA", torch.cuda.is_available())

    model2class = {
        "scgen": ScGenPipeline,
        "scbutterfly": ButterflyPipeline,
        "scpregan": ScPreGanPipeline,
        "scpregan-reproducible": ScPreGanReproduciblePipeline,
        "scbutterfly-no-reusing": ButterflyPipelineNoReusing,
        "vidr-single": VidrSinglePipeline,
        "vidr-multi": VidrMultiplePipeline,
    }

    preprocessing2class = {
        "default": PreprocessingGenericPipeline,
        "no-filtering": PreprocessingNoFilteringPipeline,
    }

    condition2class = {
        "pbmc": PbmcSinglePipeline,
        "nault": NaultSinglePipeline,
        "sciplex3": Sciplex3SinglePipeline,
        "nault-multi": NaultMultiplePipeline,
        "nault-liver": NaultSinglePipeline,
        "nault-liver-multi": NaultMultiplePipeline
    }
    
    dataset2class = {
        "pbmc": PbmcPipeline,
        "nault": NaultPipeline,
        "sciplex3": Sciplex3Pipeline,
        "nault-multi": NaultPipeline,
        "nault-liver": NaultLiverTissuePipeline,
        "nault-liver-multi": NaultLiverTissuePipeline
    }
    
    dataset_pipeline = dataset2class[args.dataset](
        preprocessing_pipeline=preprocessing2class[args.preprocessing](),
    )

    dataset_condition_pipeline = condition2class[args.dataset](
        dataset_pipeline=dataset_pipeline,
        perturbation=args.perturbation,
        dosages=args.dosages,
    )

    model_pipeline = model2class[args.model](
        dataset_pipeline=dataset_condition_pipeline,
        experiment_name=args.experiment or "",
        debug=args.debug,
    )

    model_pipeline(
        batch=args.batch,
        append_metrics=True,
        save_plots=False,
        refresh_training=False,
        refresh_evaluation=True,
    )

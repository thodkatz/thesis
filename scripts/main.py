from thesis.model import (
    ButterflyPipeline,
    ButterflyPipelineNoReusing,
    ScGenPipeline,
    ScPreGanPipeline,
    ScPreGanReproduciblePipeline,
    VidrSinglePipeline,
)
from thesis.datasets import (
    DatasetSinglePerturbationMultipleDosePipeline,
    DatasetSinglePerturbationSingleDosePipeline,
    NaultPipeline,
    PbmcPipeline,
    Sciplex3Pipeline,
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
    parser.add_argument("--multi", action="store_true", help="multiple doses")
    parser.add_argument(
        "--experiment", type=str, required=False, help="experiment name"
    )
    parser.add_argument("--perturbation", type=str, required=True, help="perturbation")
    parser.add_argument("--dosage", type=float, required=False, help="drug dosage")
    parser.add_argument("--batch", type=int, required=True, help="batch id")
    parser.add_argument(
        "--model",
        choices=[
            "scgen",
            "scbutterfly",
            "scbutterfly-no-reusing",
            "scpregan",
            "scpregan-reproducible",
            "vidr",
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
        choices=["pbmc", "nault", "sciplex3"],
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
        "vidr": VidrSinglePipeline,
    }

    preprocessing2class = {
        "default": PreprocessingGenericPipeline,
        "no-filtering": PreprocessingNoFilteringPipeline,
    }

    datasets2class = {
        "pbmc": PbmcPipeline,
        "nault": NaultPipeline,
        "sciplex3": Sciplex3Pipeline,
    }
    
    if not args.multi:
        dataset_pipeline = DatasetSinglePerturbationSingleDosePipeline(
                dataset_pipeline=datasets2class[args.dataset](
                    preprocessing_pipeline=preprocessing2class[args.preprocessing]()
                ),
                perturbation=args.perturbation,
                dosage=args.dosage,
            )
    else:
        dataset_pipeline = DatasetSinglePerturbationMultipleDosePipeline(
                dataset_pipeline=datasets2class[args.dataset](
                    preprocessing_pipeline=preprocessing2class[args.preprocessing]()
                ),
                perturbation=args.perturbation,
            )     

    model_pipeline = model2class[args.model](
        dataset_pipeline=dataset_pipeline,
        experiment_name=args.experiment or "",
        debug=args.debug,
    )

    model_pipeline(
        batch=args.batch,
        append_metrics=True,
        save_plots=False,
        refresh_training=False,
        refresh_evaluation=False,
    )

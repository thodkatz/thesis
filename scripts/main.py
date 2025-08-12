from thesis.model import (
    ButterflyPipeline,
    ButterflyPipelineNoReusing,
    MultiTaskAaeAdversarialGaussianPipeline,
    MultiTaskAaeAdversarialPipeline,
    MultiTaskAaeAutoencoderAndOptimalTransportPipeline,
    MultiTaskAaeAutoencoderOptimalTransportPipeline,
    MultiTaskAaeAutoencoderPipeline,
    MultiTaskVaeAutoencoderAndOptimalTransportPipeline,
    MultiTaskVaeAutoencoderOptimalTransportPipeline,
    MultiTaskVaeAutoencoderPipeline,
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
    CrossStudyConditionPipeline,
    CrossStudyPipeline,
    CrossSpeciesPipeline,
    CrossSpeciesConditionPipeline,
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
    parser.add_argument("--seed", type=int, default=19193, help="random seed")
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
            "simple",
            "adversarial",
            "adversarial_gaussian",
            "simple_ot",
            "simple_and_ot",
            "simple_vae",
            "simple_vae_ot",
            "simple_vae_and_ot",
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
        choices=[
            "pbmc",
            "nault",
            "sciplex3",
            "nault-multi",
            "nault-liver",
            "nault-liver-multi",
            "cross-study",
            "cross-species",
        ],
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
        "simple": MultiTaskAaeAutoencoderPipeline,
        "adversarial": MultiTaskAaeAdversarialPipeline,
        "adversarial_gaussian": MultiTaskAaeAdversarialGaussianPipeline,
        "simple_ot": MultiTaskAaeAutoencoderOptimalTransportPipeline,
        "simple_and_ot": MultiTaskAaeAutoencoderAndOptimalTransportPipeline,
        "simple_vae": MultiTaskVaeAutoencoderPipeline,
        "simple_vae_ot": MultiTaskVaeAutoencoderOptimalTransportPipeline,
        "simple_vae_and_ot": MultiTaskVaeAutoencoderAndOptimalTransportPipeline,
    }
    
    def is_multi_task(model_name):
        return model_name in [
            "simple",
            "adversarial",
            "adversarial_gaussian",
            "simple_ot",
            "simple_and_ot",
            "simple_vae",
            "simple_vae_ot",
            "simple_vae_and_ot",
        ]

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
        "nault-liver-multi": NaultMultiplePipeline,
        "cross-study": CrossStudyConditionPipeline,
        "cross-species": CrossSpeciesConditionPipeline,
    }

    dataset2class = {
        "pbmc": PbmcPipeline,
        "nault": NaultPipeline,
        "sciplex3": Sciplex3Pipeline,
        "nault-multi": NaultPipeline,
        "nault-liver": NaultLiverTissuePipeline,
        "nault-liver-multi": NaultLiverTissuePipeline,
        "cross-study": CrossStudyPipeline,
        "cross-species": CrossSpeciesPipeline,
    }

    dataset_pipeline = dataset2class[args.dataset](
        preprocessing_pipeline=preprocessing2class[args.preprocessing](),
    )

    dataset_condition_pipeline = condition2class[args.dataset](
        dataset_pipeline=dataset_pipeline,
        perturbation=args.perturbation,
        dosages=args.dosages,
    )

    
    model_pipeline_class = model2class[args.model]

    if (
        is_multi_task(args.model)
        and args.dataset in ["nault-multi", "nault-liver-multi"]
   ):
        model_pipeline = model2class[args.model].get_multiple_condition(
            dataset_pipeline=dataset_condition_pipeline,
            experiment_name=args.experiment or "seed_" + str(args.seed),
            debug=args.debug,
            seed=args.seed,
        )
    else:
        model_pipeline = model_pipeline_class(
            dataset_pipeline=dataset_condition_pipeline,
            experiment_name=args.experiment or "seed_" + str(args.seed),
            debug=args.debug,
            seed=args.seed,
        )

    model_pipeline(
        batch=args.batch,
        append_metrics=False,
        save_plots=False,
        refresh_training=False,
        refresh_evaluation=False,
    )

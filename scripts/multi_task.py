from thesis.datasets import NaultLiverTissuePipeline, NaultPipeline, NaultMultiplePipeline, NaultSinglePipeline, PbmcPipeline, PbmcSinglePipeline, Sciplex3Pipeline, Sciplex3SinglePipeline
from thesis.model import (
    MultiTaskAaeAutoencoderPipeline,
    MultiTaskAaeAdversarialPipeline,
    MultiTaskAaeAdversarialGaussianPipeline,
    MultiTaskAaeAutoencoderOptimalTransportPipeline,
    MultiTaskAaeAutoencoderAndOptimalTransportPipeline,
    MultiTaskVaeAutoencoderPipeline,
    MultiTaskVaeAutoencoderOptimalTransportPipeline,
    MultiTaskVaeAutoencoderAndOptimalTransportPipeline
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model",
        choices=[
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
    parser.add_argument("--seed", type=int, default=19193, help="random seed")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--dataset",
        choices=["pbmc", "nault", "sciplex3", "nault-multi", "nault-liver", "nault-liver-multi"],
        help="Chooose dataset",
    )
    parser.add_argument("--perturbation", type=str, required=True, help="perturbation")
    parser.add_argument("--dosages", type=float, required=False, help="drug dosage")    

    args = parser.parse_args()

    model2class = {
        "simple": MultiTaskAaeAutoencoderPipeline,
        "adversarial": MultiTaskAaeAdversarialPipeline,
        "adversarial_gaussian": MultiTaskAaeAdversarialGaussianPipeline,
        "simple_ot": MultiTaskAaeAutoencoderOptimalTransportPipeline,
        "simple_and_ot": MultiTaskAaeAutoencoderAndOptimalTransportPipeline,
        "simple_vae": MultiTaskVaeAutoencoderPipeline,
        "simple_vae_ot": MultiTaskVaeAutoencoderOptimalTransportPipeline,
        "simple_vae_and_ot": MultiTaskVaeAutoencoderAndOptimalTransportPipeline
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

    dataset_pipeline = dataset2class[args.dataset]()

    dataset_condition_pipeline = condition2class[args.dataset](
        dataset_pipeline=dataset_pipeline,
        perturbation=args.perturbation,
        dosages=args.dosages,
    ) 

    multitask = model2class[args.model](
        dataset_pipeline=dataset_condition_pipeline,
        experiment_name="no_film_hidden_bugfix_seed_" + str(args.seed),
        debug=False,
        seed=args.seed,
    )

    cell_type_key = multitask.dataset_pipeline.cell_type_key
    cell_type_list = list(
        multitask.dataset_pipeline.dataset.obs[cell_type_key].cat.categories
    )
    print(cell_type_list[args.batch])
    # cell_type_index = cell_type_list.index("Hepatocytes - portal")

    multitask(
        batch=args.batch,
        append_metrics=True,
        save_plots=False,
        refresh_training=False,
        refresh_evaluation=False,
    )

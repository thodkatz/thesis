from thesis.datasets import NaultPipeline, NaultMultiplePipeline
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

    multitask = model2class[args.model](
        dataset_pipeline=NaultMultiplePipeline(
            NaultPipeline(),
            perturbation="tcdd",
        ),
        experiment_name="seed_" + str(args.seed),
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
        save_plots=True,
        refresh_training=False,
        refresh_evaluation=False,
    )

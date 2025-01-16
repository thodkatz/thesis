from thesis.datasets import NaultPipeline, NaultMultiplePipeline
from thesis.model import MultiTaskAaeAutoencoderPipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scbutterfly nault")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--experiment", type=str, required=True, help="experiment name"
    )
    parser.add_argument("--dosages", type=float, nargs='+', required=True, help="drug dosages")
    parser.add_argument("--seed", type=int, default=19193, help="random seed")

    args = parser.parse_args()
    

    multitask = MultiTaskAaeAutoencoderPipeline(
        dataset_pipeline=NaultMultiplePipeline(
            NaultPipeline(),
            perturbation="tcdd",
            dosages=args.dosages,
        ),
        experiment_name=args.experiment,
        debug=args.debug,
        seed=args.seed
    )
    
    cell_type_key = multitask.dataset_pipeline.cell_type_key
    cell_type_list = list(
        multitask.dataset_pipeline.dataset.obs[cell_type_key].cat.categories
    )
    cell_type_index = cell_type_list.index("Hepatocytes - portal")
        

    multitask(
        batch=cell_type_index,
        append_metrics=True,
        save_plots=True,
        refresh_training=False,
        refresh_evaluation=False,
    )
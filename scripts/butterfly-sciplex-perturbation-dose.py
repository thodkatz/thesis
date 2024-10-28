from thesis.model_pipelines import ButterflyPipeline
from thesis.datasets_pipelines import Sciplex3Pipeline
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sciplex3 with specific perturbation type and drug dosage. Do not reuse stimulated samples when performing optimal transport")
    parser.add_argument('--perturbation', type=str, required=True, help="Type of perturbation (e.g., 'control').")
    parser.add_argument('--dosage', type=int, required=True, help="Drug dosage to use (e.g., 10000).")
    parser.add_argument("--batch", type=int, required=False, help="batch id")
    args = parser.parse_args()
    
    
    butterfly_sciplex = ButterflyPipeline(
        dataset_pipeline=Sciplex3Pipeline(
            perturbation=args.perturbation, dosage=args.dosage
        ),
        experiment_name="",
    )
    butterfly_sciplex(batch=args.batch, append_metrics=True, save_plots=False)
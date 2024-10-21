from thesis.scbutterfly import run_sciplex3_no_reusing
from thesis.datasets import get_sciplex3_per_perturbation
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sciplex3 with specific perturbation type and drug dosage. Do not reuse stimulated samples when performing optimal transport")
    
    parser.add_argument('--name', type=str, required=True, help="Name of the experiment.")
    parser.add_argument('--perturbation_type', type=str, required=True, help="Type of perturbation (e.g., 'control').")
    parser.add_argument('--drug_dosage', type=int, required=True, help="Drug dosage to use (e.g., 10000).")
    
    args = parser.parse_args()
    
    dataset = get_sciplex3_per_perturbation(perturbation_type=args.perturbation_type, drug_dosage=args.drug_dosage)
    run_sciplex3_no_reusing(name=args.name, dataset=dataset, perturbation_name=args.perturbation_type, dosage=args.drug_dosage)
    
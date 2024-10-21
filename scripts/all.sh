#!/bin/bash

REPO=~/thesis/ssh/thesis
echo "Repo: "$REPO
HELPER=../embl-resources/slurm/submit_gpu.py
echo "sbatch script path: "$HELPER


butterfly_pbmc() {
    for i in {0..6}
    do
        echo "Number: $i"
        $HELPER $REPO/scripts/butterfly-pbmc.py --batch $i
    done
}

# butterfly_sciplex3_perturbation_dose_no_resuing() {
#     python $REPO/scripts/butterfly-sciplex3-perturbation-dose-no-resuing.py
# }

# butterfly_sciplex3_perturbation() {

# }

# butterfly_nault() {
    
# }

# butterfly_nault_no_filtering() {

# }

# dataset1: sc.AnnData = sciplex3[
#     (sciplex3.obs["perturbation"] == "Ellagic acid")
#     | (sciplex3.obs["perturbation"] == "control")
# ]
# dataset1 = dataset1[
#     (dataset1.obs["dose_value"] == 10000) | (dataset1.obs["perturbation"] == "control")
# ]

# dataset2: sc.AnnData = sciplex3[
#     (sciplex3.obs["perturbation"] == "UNC1999")
#     | (sciplex3.obs["perturbation"] == "control")
# ]
# dataset2 = dataset2[
#     (dataset2.obs["dose_value"] == 1000) | (dataset2.obs["perturbation"] == "control")
# ]

# dataset3: sc.AnnData = sciplex3[
#     (sciplex3.obs["perturbation"] == "Ellagic acid")
#     | (sciplex3.obs["perturbation"] == "control")
# ]
# dataset3 = dataset3[
#     (dataset3.obs["dose_value"] == 1000) | (dataset3.obs["perturbation"] == "control")
# ]


butterfly_pbmc

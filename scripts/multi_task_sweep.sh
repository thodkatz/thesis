#!/bin/bash

REPO=/g/kreshuk/katzalis/repos/thesis-tmp
echo "Repo: "$REPO
HELPER=$REPO/scripts/submit_gpu.py
echo "sbatch script path: "$HELPER


ENABLE_DEBUG="false"

if [ "$ENABLE_DEBUG" = "true" ]; then
    DEBUG='--debug --experiment playground'
    echo "Debug mode enabled"
else
    DEBUG=''
fi


# nault_multi() {
#     declare -A experiments=(
#         ["dosage01"]="0.01"
#         ["dosage03"]="0.03"
#         ["dosage1"]="0.1"
#         ["dosage3"]="0.3"
#         ["dosage1"]="1.0"
#         ["dosage3"]="3.0"
#         ["dosage10"]="10.0"
#         ["dosage30"]="30.0"
#         ["dosage30_10"]="30.0 10.0"
#         ["dosage30_10_3"]="30.0 10.0 3.0"
#         ["dosage30_10_3_1"]="30.0 10.0 3.0 1.0"
#         ["dosage30_10_3_1_03"]="30.0 10.0 3.0 0.3"
#         ["dosage30_10_3_1_03_01"]="30.0 10.0 3.0 0.3 0.1"
#         ["dosage30_10_3_1_03_01_003"]="30.0 10.0 3.0 0.3 0.1 0.03"             
#         ["all_dosages"]="30.0 10.0 3.0 1.0 0.3 0.1 0.03 0.01"
#         ["all_lowest_dosages"]="0.01 0.03 0.1 0.3 1.0 3.0"
#     )

#     for experiment in "${!experiments[@]}"; do
#         local dosages="${experiments[$experiment]}"
#         echo $experiment
#         echo $dosages
#         $HELPER $REPO/scripts/multi_task_sweep.py --dosages $dosages --experiment "$experiment"
#     done
# }


dosages=("0.01" "0.03" "0.1" "0.3" "1.0" "3.0" "10.0" "30.0")

generate_combinations() {
    local array=("$@")
    local n=${#array[@]}

    # Total combinations = 2^n - 1 (excluding the empty set)
    local total=$((2 ** n - 1))

    # Iterate through all possible combinations
    for ((i = 1; i <= total; i++)); do
        combination=""
        for ((j = 0; j < n; j++)); do
            # Check if the j-th element is in the current combination
            if ((i & (1 << j))); then
                combination+="${array[j]} "
            fi
        done
        local experiment="dosage$(printf "%s_""${combination[@]}" | sed 's/_$//' | sed 's/\./_/g' | sed 's/ /_/g')"
        echo $experiment
        echo $combination

        # seed 1 2 19193
        # be careful to not clog the cluster
        #for seed in 19193; do
            #$HELPER $REPO/scripts/multi_task_sweep.py --dosages $combination --experiment "${experiment}${seed}" --seed $seed
        #done

    done
}


generate_combinations "${dosages[@]}"




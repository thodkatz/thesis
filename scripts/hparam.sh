#!/bin/bash

REPO=/g/kreshuk/katzalis/repos/thesis-tmp
echo "Repo: "$REPO
HELPER=$REPO/scripts/submit_gpu.py
echo "sbatch script path: "$HELPER

run() {
    for trial in {1..100}; do
        $HELPER $REPO/experiments/multi_task_aae_hparam.py
    done
}

run

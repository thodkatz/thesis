#!/bin/bash

REPO=/g/kreshuk/katzalis/repos/thesis
echo "Repo: "$REPO
HELPER=$REPO/scripts/submit_gpu.py
echo "sbatch script path: "$HELPER

pbmc() {
    for batch in {0..6}; do
        for model in scbutterfly scgen scpregan; do
            echo "Model: $model, Number: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset pbmc --perturbation ifn-b --dosage 0.0
        done
    done
}

sciplex3() {
    for batch in {0..2}; do
        for model in scbutterfly scbutterfly-no-reusing scgen scpregan; do
            echo "Model: $model, Number: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset sciplex3 --perturbation 'Ellagic acid' --dosage 10000.0
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset sciplex3 --perturbation 'Divalproex Sodium' --dosage 1000.0
        done
    done
}

nault() {
    for dosage in 0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0; do
        for batch in {0..10}; do
            for model in scbutterfly scgen scpregan vidr; do
                echo "Model: $model, Dosage: $dosage, Batch: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault --dosage $dosage --perturbation tcdd
            done
        done
    done
}

nault_multi() {
    for batch in {0..10}; do
        for model in vidr; do
            echo "Model: $model, Dosage: $dosage, Batch: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault --dosage $dosage --perturbation tcdd --multi
        done
    done
}

#pbmc

nault_multi
nault

#sciplex3

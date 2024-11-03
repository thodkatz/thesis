#!/bin/bash

REPO=/g/kreshuk/katzalis/repos/thesis
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

pbmc() {
    for batch in {0..6}; do
        for model in scbutterfly scgen scpregan vidr-single; do
            echo "Model: $model, Number: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset pbmc --perturbation ifn-b --dosages -1.0 $DEBUG
        done
    done
}

sciplex3() {
    for batch in {0..2}; do
        for model in scbutterfly scgen scpregan vidr-single; do
            echo "Model: $model, Number: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset sciplex3 --perturbation 'Ellagic acid' --dosages 10000.0
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset sciplex3 --perturbation 'Divalproex Sodium' --dosages 1000.0
        done
    done
}

nault() {
    for dosage in 0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0; do
        for batch in {0..10}; do
            for model in scbutterfly scgen scpregan vidr-single; do
                echo "Model: $model, Dosage: $dosage, Batch: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault --dosages $dosage --perturbation tcdd $DEBUG
            done
        done
    done
}

nault_liver() {
    for dosage in 0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0; do
        for batch in {0..5}; do
            for model in scbutterfly scgen scpregan vidr-single; do
                echo "Model: $model, Dosage: $dosage, Batch: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault-liver --dosages $dosage --perturbation tcdd $DEBUG
            done
        done
    done  
}

nault_multi() {
    for batch in {0..10}; do
        for model in vidr-multi; do
            echo "Model: $model, Batch: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault-multi --perturbation tcdd $DEBUG
        done
    done
}

nault_multi_liver() {
    for batch in {0..5}; do
        for model in vidr-multi; do
            echo "Model: $model, Batch: $batch"
            $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault-liver-multi --perturbation tcdd $DEBUG
        done
    done
}

pbmc

nault_multi

nault

nault_liver

nault_multi_liver

#sciplex3

#!/bin/bash

REPO=/g/kreshuk/katzalis/repos/thesis
echo "Repo: "$REPO
HELPER=$REPO/scripts/submit_gpu_embl.py
echo "sbatch script path: "$HELPER

ENABLE_DEBUG="false"

if [ "$ENABLE_DEBUG" = "true" ]; then
    DEBUG='--debug --experiment playground'
    echo "Debug mode enabled"
else
    DEBUG=''
fi

pbmc() {
    for seed in 1 2; do
        for batch in {0..6}; do
            for model in vidr-single; do
                echo "Model: $model, Number: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset pbmc --perturbation ifn-b --dosages -1.0 $DEBUG --seed $seed
            done
        done
    done
}

cross_study() {
    for seed in 1 2 19193; do
        for batch in {0..6}; do
            for model in scbutterfly scgen scpregan vidr-single simple adversarial adversarial_gaussian simple_ot simple_and_ot simple_vae simple_vae_ot simple_vae_and_ot; do
                echo "Model: $model, Number: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset cross-study --perturbation ifn-b --dosages -1.0 $DEBUG --seed $seed
            done
        done
    done
}

cross_species() {
    for seed in 1 2 19193; do
        for batch in {0..3}; do
            for model in scbutterfly; do
                echo "Model: $model, Number: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset cross-species --perturbation lps --dosages -1.0 $DEBUG --seed $seed
            done
        done
    done
}


sciplex3() {
    for seed in 1 2 19193; do    
        for batch in {0..2}; do
            for model in scbutterfly scgen scpregan vidr-single; do
                echo "Model: $model, Number: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset sciplex3 --perturbation 'Ellagic acid' --dosages 10000.0
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset sciplex3 --perturbation 'Divalproex Sodium' --dosages 1000.0
            done
        done
    done        
}

nault() {
    for seed in 1 2; do
        for dosage in 0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0; do
            for batch in {0..10}; do
                for model in vidr-single; do
                    echo "Model: $model, Dosage: $dosage, Batch: $batch"
                    $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault --dosages $dosage --perturbation tcdd $DEBUG --seed $seed
                done
            done
        done
    done
}

nault_liver() {
    for seed in 2; do
        for dosage in 0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0; do
            for batch in {0..5}; do
                for model in scbutterfly scgen scpregan vidr-single; do
                    echo "Model: $model, Dosage: $dosage, Batch: $batch"
                    $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault-liver --dosages $dosage --perturbation tcdd $DEBUG --seed $seed
                done
            done
        done
    done
}

nault_multi() {
    for seed in 2; do
        for batch in {0..10}; do
            for model in vidr-multi; do
                echo "Model: $model, Batch: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault-multi --perturbation tcdd $DEBUG --seed $seed
            done
        done
    done
}


nault_multi_liver() {
    for seed in 2; do
        for batch in {0..5}; do
            for model in vidr-multi; do
                echo "Model: $model, Batch: $batch"
                $HELPER $REPO/scripts/main.py --batch $batch --model $model --dataset nault-liver-multi --perturbation tcdd $DEBUG --seed $seed
            done
        done
    done
}


multi_task() {
    for seed in 1 2 19193; do # 1 2 19193
       for dataset in nault-multi; do
            for batch in {0..10}; do # 0 1 2 3 4 5 6 7 8 9 10
                for model in simple adversarial adversarial_gaussian simple_ot simple_and_ot simple_vae simple_vae_ot simple_vae_and_ot; do # simple adversarial adversarial_gaussian simple_ot simple_and_ot simple_vae simple_vae_ot simple_vae_and_ot
                    echo "Batch: $batch"
                    $HELPER $REPO/scripts/multi_task.py --batch $batch --model $model --seed $seed --dataset $dataset --perturbation tcdd
                done
            done
        done
    done
}

#multi_task

# nault liver single for multi task ?

#cross_study

cross_species

#pbmc

#nault

# nault_liver

# nault_multi

# nault_multi_liver

#sciplex3

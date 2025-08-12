#!/bin/bash

for seed in 1; do # 1 2 19193
    for dataset in nault-multi; do
        for batch in 0; do # 0 1 2 3 4 5 6 7 8 9 10
            for model in simple adversarial adversarial_gaussian simple_ot simple_and_ot simple_vae simple_vae_ot simple_vae_and_ot; do # simple adversarial adversarial_gaussian simple_ot simple_and_ot simple_vae simple_vae_ot simple_vae_and_ot
                echo "Batch: $batch"
                python scripts/main.py --batch $batch --model $model --seed $seed --dataset $dataset --perturbation tcdd --experiment bugfix_seed_1
            done
        done
    done
done


for seed in 1; do
    for dosage in 30.0; do
        for batch in 0; do
            for model in scbutterfly; do
                echo "Model: $model, Dosage: $dosage, Batch: $batch"
                python scripts/main.py --batch $batch --model $model --dataset nault --dosages $dosage --perturbation tcdd --seed $seed --experiment seed_1
            done
        done
    done
done

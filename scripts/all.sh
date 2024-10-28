#!/bin/bash

REPO=/g/kreshuk/katzalis/repos/thesis
echo "Repo: "$REPO
HELPER=$REPO/scripts/submit_gpu.py
echo "sbatch script path: "$HELPER


pbmc() {
    for i in {0..6}
    do
        echo "Number: $i"
        $HELPER $REPO/scripts/butterfly-pbmc.py --batch $i
        $HELPER $REPO/scripts/scpregan-pbmc.py --batch $i
        $HELPER $REPO/scripts/scpregan-pbmc-reproducible.py --batch $i
    done
}

sciplex3() {
    for i in {0..2}
    do
        echo "Number: $i"
        $HELPER $REPO/scripts/butterfly-sciplex-perturbation-dose-no-reusing.py --perturbation "Ellagic acid" --dosage 10000 --batch $i
        $HELPER $REPO/scripts/butterfly-sciplex-perturbation-dose-no-reusing.py --perturbation "Divalproex Sodium" --dosage 1000 --batch $i
        $HELPER $REPO/scripts/butterfly-sciplex-perturbation-dose.py --perturbation "Ellagic acid" --dosage 10000 --batch $i
        $HELPER $REPO/scripts/butterfly-sciplex-perturbation-dose.py --perturbation "Divalproex Sodium" --dosage 1000 --batch $i
    done
}

nault() {
    for dosage in 0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0
    do
        for batch in {0..10}
        do
            echo "Dosage: $dosage, Batch: $batch"
            $HELPER $REPO/scripts/butterfly-nault.py --dosage $dosage --batch $batch
            $HELPER $REPO/scripts/scpregan-nault.py --dosage $dosage --batch $batch            
            #$HELPER $REPO/scripts/butterfly-nault-no-filtering.py --dosage $dosage --batch $batch
        done
    done
}

#pbmc

nault

#sciplex3
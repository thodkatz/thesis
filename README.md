# Multi-task learning in perturbation modeling

# Setup

Requirements:
- conda
- make

```
git clone git@github.com:thodkatz/thesis.git
cd thesis
conda env create -n <env name> -f environment
conda activate <env name>
make setup_env ENV_NAME=<env name>
```

This will install the necessarry dependencies and it will create two directories. The `data`, and the `saved_results`. In the first one we will store the datasets, and in the second one all the artifacts (e.g. pytorch state dict from the models, benchmarking results) from our experiments.

The datasets used for the benchmarking so far can be found here https://drive.google.com/drive/folders/1zcTdTAmcDprXYFJ8EGLado6AP0-H9fX-?usp=sharing. Download the content under the `data` dir. The file paths should be:
```
./data/pbmc/pbmc.h5ad
./data/scvidr/nault2021_multiDose.h5ad
```

# Benchmarking

Our analysis aims to benchmark the task of the out-of-distribution perturbation prediction. 

For example given a dataset with some control and perturbed cells and a set of cell types, the task is to predict the perturbation for a cell type that the model hasn't seen before.

Let's suppose that we have 2 cell types and 2 conditions for each cell type:


|cell type   | condition   |
|---|---|
|Hepatocytes - portal   |control   |
|Hepatocytes - portal   |stimulated   |
|B cells   |control   |
|B cells   |stimulated   |

If we have as a target the `Hepatocytes - portal`, then we hold as a testing set the `Hepatocytes - portal stimulated`, and we can use the rest for training. Then we have as input the `Hepatocytes - portal control`, and we try to predict the stimulated expression.

To run the benchmarking for the models `scButterfly`, `scGen`, `scPreGAN`, and `scVIDR`, you can use the `scripts/main.py`. For example:

```shell
python ./scripts/main.py --batch 4 --model scbutterfly  --dataset pbmc --perturbation ifn-b --dosages -1.0 --seed 1
```

This will run the pipeline for the `scButterfly`, for the `pbmc` dataset, having as a cell type target the `Hepatocytes - portal` (because the batch number `4` corresponds to the `Hepatocytes - portal` as the index of the list `sorted(self.dataset.obs[self.cell_type_key].unique().tolist())`), for the seed `1`.

The `pbmc` dataset have 7 cell types, so to test the models for all the available models and all the cell types, we could have a script such as:

```
#!/bin/bash

pbmc() {
    for batch in {0..6}; do
        for model in scbutterfly scgen scpregan vidr-single; do
            python ./scripts/main.py --batch $batch --model $model --dataset pbmc --perturbation ifn-b --dosages -1.0 --seed $seed
        done
    done
}


pbmc

```

This though it will take a lot of time, because we test a lot of models, for all the cell types as a target. Assuming that we have an hpc infrastructure, we can use slurm scripts to assign a job for each one of the above combinations. To accomplish that, there is an opinionated slurm script `./scripts/submit.gpu`, that is leveraged to run all the combinations for the rest of the datasets as well (e.g. `scvidr`) for all the models. The `scripts/all.sh` needs to be modified for your use case.
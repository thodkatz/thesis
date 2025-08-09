# Multi-task learning in perturbation modeling

**Abstract**

Advanced single-cell technologies have provided new insights on cellular responses to
perturbations, with significant potential for translational medicine. However, the inherent
complexity of biological systems and the technical limitations of the experimental protocols
present challenges for many proposed computational methods to algorithmically capture the
perturbation mechanisms. Multi-task learning is one of the methods that have been left un-
explored in this field. In this study, we aim to bridge this gap by unraveling its potential in
single-cell perturbation modeling. We have developed a multi-task autoencoder architecture
that predicts perturbed single-cell transcriptomic profiles for multiple perturbations achiev-
ing state-of-the-art performance while exhibiting greater scalability and efficiency compared
to existing methods.

The report can be found at `report/main.pdf`.

# Setup

Requirements:
- conda
- make
- Linux
- NVIDIA gpu

```
git clone --recurse-submodules git@github.com:thodkatz/thesis.git
cd thesis
conda env create -n <env name> -f environment.yml
conda activate <env name>
make setup_env
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

# Hyperparameter tuning

For hyperparameter tuning we used [optuna](https://github.com/optuna/optuna). To view the results, create a separate environment using pip.

```
# env and installation
cd optuna
virtualenv .venv
source .venv/bin/activate
pip install optuna-dashboard

# launch dashboard
optuna-dashboard sqlite:///db.sqlite3
# > Listening on http://127.0.0.1:8080


# if needed forward the port
ssh -L 18080:127.0.0.1:8080 username@server

# view the dashboard on localhost:18080

```
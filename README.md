# Multi-task learning in perturbation modeling

**Abstract**

Advanced single-cell technologies have provided new insights for the comprehension and
utilization of cellular responses to perturbations, with significant potential for biomedicine.
However, the inherent complexity of biological systems and the technical limitations of the
experimental protocols present challenges for many proposed computational methods to algorithmically capture the perturbation mechanisms. 
Multi-task learning is one of the methods that have been left unexplored in this field. In this study, we aim to bridge this gap by unraveling its potential in single-cell perturbation modeling. We have developed a multi-task autoencoder architecture that predicts perturbed single-cell transcriptomic profiles for multiple perturbations. This method achieves state-of-the-art performance while exhibiting greater scalability and efficiency compared to existing methods. 
Further investigation and refinement of this architecture as a theoretical and abstract model for simultaneously solving multiple related problems in perturbation modeling and broader bioinformatics would be of particular interest.


The report can be found at `report/main.pdf` and the Greek version at `report/greek/main_el.pdf`. There is also a Greek presentation available at `report/presentation/presentation_el.pdf`.

## Intro

Our analysis is based on the task of the out-of-distribution detection (OOD). Given single-cell transcriptomic profiles under control and perturbed conditions across several cell types, the goal is to predict the perturbed gene expression for an unseen cell type given its control profile. 

For example, let's suppose that we have two cell types and two conditions for each cell type (control and stimulated/perturbed):


|cell type   | condition   |
|---|---|
|Hepatocytes - portal   |control   |
|Hepatocytes - portal   |stimulated   |
|B cells   |control   |
|B cells   |stimulated   |

If our target is the cell type of `Hepatocytes - portal`, then we hold out its stimulated profiles, and we can use the rest for training. Then, using its control profile as input, we aim to predict its stimulated expression.

We have tested our model for three case studies. For the first one, we evaluated it on human peripheral blood mononuclear cells (PBMCs) stimulated by interferon-beta (IFN-b), where only one perturbation is observed. The second one includes multiple perturbations represented as different dosages of the TCDD drug administered to mice. The third case, like the first, involves a single perturbation, but applied across different species, aiming to predict the perturbed transcriptomic profile for an unseen species instead of an unseen cell type. These case studies are referred as `pbmc`, `nault`, and `cross-species` correspondingly.

## Directory structure

```
├── analysis
│   ├── analysis_literature_models.ipynb
│   ├── analysis_multi_task_aae.ipynb
│   ├── datasets.ipynb
│   ├── distance_metrics.ipynb
│   ├── metrics2.csv
│   ├── metrics.csv
│   ├── multi_task_aae.csv
│   ├── multi_task_aae_overview.csv
│   └── train_test_counts.ipynb
├── data
├── environment.yml
├── experiments
│   ├── butterfly.py
│   ├── count_parameters.py
│   ├── evaluation.ipynb
│   ├── multi_task_aae_cli.py
│   ├── multi_task_aae_hparam.py
│   ├── playground.ipynb
│   ├── scbutterfly-perturbation.ipynb
│   └── unifly-pbmc.ipynb
├── lib
│   ├── codex
│   ├── scButterfly
│   ├── scgen
│   ├── scPreGAN
│   ├── scVIDR
│   └── UnitedNet
├── Makefile
├── optuna
│   └── db.sqlite3
├── README.md
├── report
├── saved_results
├── scripts
│   ├── all.sh
│   ├── aristotelis.sh
│   ├── diagnostics.sh
│   ├── hparam.sh
│   ├── main.py
│   ├── multi_task_sweep.py
│   ├── multi_task_sweep.sh
│   ├── submit_gpu_aristotelis.py
│   └── submit_gpu_embl.py
├── setup.py
├── thesis
    ├── datasets.py
    ├── evaluation.py
    ├── model.py
    ├── multi_task_aae.py
    ├── preprocessing.py
    └── utils.py
```

- The `lib` directory contains the repositories of the literature models used for experimentation and benchmarking. These are fetched via git submodules as mentioned in [Setup](#setup).
- Under `report` we can find all the relevant files to generate our main report of this work (`report/main.pdf`).
- The `analysis` directory is used to create the plots of the evaluation data saved at `analysis/metrics.csv`.
- The core implementation of our multi-task models can be found at `thesis/multi_task_aae.py`.
- A common interface to benchmark all of these different models is implemented named as `ModelPipeline` in `thesis/model.py`.
- The `scripts` directory is used to automate the benchmarking of all the models across all case studies running on a cluster. More in [Scripts - HPC](#scripts)


## <a name="setup"></a> Setup

Requirements:
- conda
- make
- Linux
- cuda

```
git clone --recurse-submodules git@github.com:thodkatz/thesis.git
cd thesis
conda env create -n <env name> -f environment.yml
conda activate <env name>
make setup_env
```

This will install the necessary dependencies and it will create two directories. The `data`, and the `saved_results`.
Under the `data` directory, we should download the datasets from [here](https://drive.google.com/drive/folders/1zcTdTAmcDprXYFJ8EGLado6AP0-H9fX-?usp=sharing). The file paths should be:
```
./data/pbmc/pbmc.h5ad
./data/scvidr/nault2021_multiDose.h5ad
./data/cross_species/cross_species.h5ad
```

The `saved_results` is where all the artifacts (e.g. pytorch state dict from the models, evaluation metrics, plots) from our experiments will be saved.

## Usage

The literature models used for comparison with our multi-task variations are `scButterfly`, `scGen`, `scPreGAN`, and `scVIDR`.

To evaluate the models we can use the `./scripts/main.py`. For example:

```shell
python ./scripts/main.py --batch 4 --model scbutterfly  --dataset pbmc --perturbation ifn-b --dosages -1.0 --seed 1
```

This will run the evaluation pipeline for the `scButterfly`, on the `pbmc` dataset, holding-out the cell type with index `4`, (`Hepatocytes - portal`) for the seed `1`. Setting the dosages to `-1.0` is a convention to have a consistent interface across all studies including or not dosages. Under `saved_results`, a `metrics.csv` will be created with all the evaluation metrics comparing the predicted and the expected stimulated transcriptomic profiles.
Under `saved_results`, a directory named `ButterflyPipeline` will be created including all the artifacts from the experiment.

To evaluate the multi-task models on the multiple perturbations case study of `nault` along with the `scVIDR`'s multiple dosages version named as `vidr-multi`, we have:

```shell
python ./scripts/main.py --batch 4 --model vidr-multi  --dataset nault-multi --perturbation tcdd --seed 1
```

For one of our multi-task versions, the baseline one named as `simple`, we have:

```shell
python ./scripts/main.py --batch 4 --model simple --dataset nault-multi --perturbation tcdd --seed 1
```

The multiple perturbations models (our multi-task variations along with `vidr-multi`) can also use a list of dosages to be trained and evaluated:

```shell
python ./scripts/main.py --batch 4 --model simple --dataset nault-multi --dosages 0.01 0.1 30.0 --perturbation tcdd --seed 1
```

Because the `nault` dataset is considered a multiple perturbations dataset due to the different dosages, for single perturbation models such as  `scButterfly`, `scGen`, `scPreGAN`, and single perturbation version of `scVIDR` referred to as `vidr-single`, we need to specify a specific dosage to be considered as the perturbation:

```shell
python ./scripts/main.py --batch 4 --model scbutterfly --dataset nault --dosages 30.0 --perturbation tcdd --seed 1
```

Alternatively, using directly Python objects, we could test our models such as:

```python
from thesis.datasets import NaultPipeline, NaultSinglePipeline
from thesis.model import ButterflyPipeline


butterfly_nault = ButterflyPipeline(
    dataset_pipeline=NaultSinglePipeline(NaultPipeline(), dosages=0.01),
    experiment_name="playground",
    debug=False,
)

cell_type_key = butterfly_nault.dataset_pipeline.cell_type_key
cell_type_list = list(
    butterfly_nault.dataset_pipeline.dataset.obs[cell_type_key].cat.categories
)
cell_type_index = cell_type_list.index("Hepatocytes - portal")

butterfly_nault(
    batch=cell_type_index,
    append_metrics=False,
    save_plots=False,
    refresh_training=True,
    refresh_evaluation=True,
)
```

For the last one, there are several examples under `experiments/playground.ipynb`.



## <a name="scripts"></a> Scripts - HPC

Let's assume that we want to benchmark all the models having each possible cell type as a target for the `pbmc` case study, which consists of seven cell types. We could have a script such as:

```
#!/bin/bash

pbmc() {
    for batch in {0..6}; do
        for model in simple scbutterfly scgen scpregan vidr-single; do
            python ./scripts/main.py --batch $batch --model $model --dataset pbmc --perturbation ifn-b --dosages -1.0 --seed $seed
        done
    done
}


pbmc

```

However, the above script is very time-consuming. Assuming that we have a High-Performance Computing (HPC) infrastructure, we can use slurm scripts to assign a job for each one of the above combinations. For our use case, we have relied on the cluster of the European Molecular Biology Laboratory (EMBL). We have a cluster-specific script `scripts/submit_gpu_embl.py` used by the `scripts/all.sh` to benchmark our models for all the case studies.

## Hyperparameter tuning

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
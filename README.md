# Multi-task learning in perturbational modeling

# Usage

```shell
conda env create -n myenv -f environment.yml
pip install -e .[dev]
conda activate myenv
```

If you want to run the `unifly-pbmc.ipynb` experiment then additionally:
```shell
git submodule init
git submodule update --remote --merge
pip install -e ./lib/UnitedNet
pip install -e ./lib/scButterfly
```

For the `scbutterfly-sciplex.ipynb`:
```shell
git submodule init
git submodule update lib/scButterfly --remote --merge
pip install -e ./lib/scButterfly
```
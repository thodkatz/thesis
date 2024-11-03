from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="thesis",
    version="1.0.0",
    description="My repo for my thesis",
    author="Theodoros Katzalis",
    packages=find_packages(
        exclude=[
            "runs",
            "saved_results",
            "data",
            "experiments",
            "analysis",
            "lib",
            "scripts",
            "slurm_scripts",
        ]
    ),
    install_requires=[
        "scanpy",
        "numpy",
        "adata",
        "pertpy",
        "tensorboard",
        "scvi-tools<=0.20.0",  # scgen
        "jax[cuda12]" # for pertpy
    ],
    extras_require={
        "dev": [
            "matplotlib",
            "plotly" "seaborn",
            "pandas",
            "jupyter",
            "black",
            "pytest",
        ]
    },
)

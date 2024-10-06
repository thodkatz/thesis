from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="thesis",
    version="1.0.0",
    description="My repo for my thesis",
    author="Theodoros Katzalis",
    packages=find_packages(where="thesis"),
    install_requires=[
        "scanpy",
        "numpy",
        "adata",
        "pertpy",
        "scipy<1.13", # https://discourse.pymc.io/t/importerror-cannot-import-name-gaussian-from-scipy-signal/14170, due to pertpy
        "jax<0.4.24", # https://github.com/scverse/pertpy/issues/545, due to pertpy
    ],
    extras_require={
        "dev": [
            "matplotlib",
            "seaborn",
            "pandas",
            "jupyter"
        ]
    },
)

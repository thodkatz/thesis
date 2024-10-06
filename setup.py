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
        "tensorboard"
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

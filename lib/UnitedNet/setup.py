from distutils.core import setup
from setuptools import find_namespace_packages

setup(
    name='UnitedNet',
    version="0.1.0",
    packages=find_namespace_packages(exclude=["tests", "notebooks"]),
    url='',
    license='',
    author='thodkatz',
    author_email='',
    description='',
    install_requires=[
        "scanpy",
        "numpy",
        "pandas",
        "torch",
        "tabulate",
        "scikit-learn",
        "shap<=0.40.0",
        "mne-connectivity",
        "tensorboard"
        ],
    extras_require={
        "dev": [
            "matplotlib"
        ],
        "torch": [
            "torch"
        ]
    }
)

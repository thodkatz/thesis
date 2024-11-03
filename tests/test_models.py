from thesis.model import VidrSinglePipeline, VidrMultiplePipeline
from thesis.datasets import (
    PbmcPipeline,
    PbmcSinglePipeline,
    NaultMultiplePipeline,
    NaultPipeline,
)


def compare_adata(adata1, adata2):
    assert adata1.obs_keys() == adata2.obs_keys()
    assert adata1.var_keys() == adata2.var_keys()
    assert adata1.X.shape == adata2.X.shape


def test_vidr_pbmc():
    """
    For Vidr to work for the PBMC dataset, that doesn't contain dosages, we assume that the stimulated cells with ifn-b
    have a dosage of -1.0, and the control cells a dose of 0.
    """
    experiment_name = "test_experiment"
    dataset_pipeline = PbmcSinglePipeline(dataset_pipeline=PbmcPipeline())
    model = VidrSinglePipeline(
        experiment_name=experiment_name, dataset_pipeline=dataset_pipeline
    )
    dose_key = dataset_pipeline.dosage_key

    assert dataset_pipeline.dosages == [-1.0]

    assert sorted(model.dataset_pipeline.dataset.obs[dose_key].unique().tolist()) == [
        -1.0,
        0.0,
    ]
    compare_adata(
        model.dataset_pipeline.control,
        model.dataset_pipeline.dataset[
            model.dataset_pipeline.dataset.obs[dose_key] == 0
        ],
    )


def test_vidr_multiple_pipeline():
    experiment_name = "test_experiment"
    dataset_pipeline = NaultMultiplePipeline(
        dataset_pipeline=NaultPipeline(preprocessing_pipeline=None)
    )
    model = VidrMultiplePipeline(
        experiment_name=experiment_name, dataset_pipeline=dataset_pipeline
    )
    dose_key = dataset_pipeline.dosage_key

    assert (
        len(sorted(model.dataset_pipeline.dataset.obs[dose_key].unique().tolist())) == 9
    )

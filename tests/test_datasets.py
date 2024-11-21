from anndata import AnnData
from thesis.datasets import (
    DatasetPipeline,
    MultipleConditionDatasetPipeline,
    NaultLiverTissuePipeline,
)
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

def test_nault_liver_cell_types():
    nault = NaultLiverTissuePipeline()
    cell_type_key = nault.cell_type_key
    nault_cell_types = nault.dataset.obs[cell_type_key].unique().tolist()
    liver_cell_types = [
        "Hepatocytes - central",
        "Hepatocytes - portal",
        "Cholangiocytes",
        "Stellate Cells",
        "Portal Fibroblasts",
        "Endothelial Cells",
    ]
    assert sorted(liver_cell_types) == sorted(nault_cell_types)


def test_split_train_valid():
    X = np.random.rand(*(55, 3))
    

    obs = pd.DataFrame(
        {
            "Dose": [0.0] * 11 + [1.0] * 11 + [2.0] * 11 + [0.0] * 11 + [1.0] * 11,
            "celltype": ["cell1"] * 33 + ["cell2"] * 22,
        },
    )

    var = pd.DataFrame(
        {"gene_symbol": ["geneA", "geneB", "geneC"]},
    )

    adata = AnnData(X=X, obs=obs, var=var)

    dataset_pipeline = DatasetPipeline(
        data_path=adata,
        cell_type_key="celltype",
        dosage_key="Dose",
        preprocessing_pipeline=None,
    )

    single_condition_dataset = MultipleConditionDatasetPipeline(
        dataset_pipeline=dataset_pipeline, perturbation="", dosages=[1.0, 2.0]
    )

    target_cell_type = "cell2"

    train, validation = single_condition_dataset.split_dataset_to_train_validation(
        target_cell_type=target_cell_type, validation_split=0.8
    )
    control_stim = single_condition_dataset.get_ctrl_test(
        target_cell_type=target_cell_type
    )
    stimulated_stim = single_condition_dataset.get_stim_test(
        target_cell_type=target_cell_type
    )

    train_labels = (
        train.obs[["celltype", "Dose"]].reset_index(drop=True)
    )
    validation_labels = (
        validation.obs[["celltype", "Dose"]].reset_index(drop=True)
    )

    expected_train_labels = pd.DataFrame(
        {
            "celltype": ["cell1"] * 24 + ["cell2"] * 8,
            "Dose": [0.0] * 8 + [1.0] * 8 + [2.0] * 8 + [0.0] * 8,
        }
    ).reset_index(drop=True)

    expected_validation_labels = pd.DataFrame(
        {
            "celltype": ["cell1"] * 9 + ["cell2"] * 3,
            "Dose": [0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [0.0] * 3,
        }
    ).reset_index(drop=True)

    assert_frame_equal(train_labels, expected_train_labels)
    assert_frame_equal(validation_labels, expected_validation_labels)

    assert control_stim.obs["Dose"].unique().tolist() == [0.0]
    assert control_stim.obs["celltype"].unique().tolist() == ["cell2"]
    assert stimulated_stim.obs["Dose"].unique().tolist() == [1.0]
    assert stimulated_stim.obs["celltype"].unique().tolist() == ["cell2"]


test_split_train_valid()

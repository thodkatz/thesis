from thesis.datasets import NaultLiverTissuePipeline

def test_nault_liver_cell_types():
    nault = NaultLiverTissuePipeline()
    cell_type_key = nault.cell_type_key
    nault_cell_types = nault.dataset.obs[cell_type_key].unique().tolist()
    liver_cell_types = [
        'Hepatocytes - central',
        'Hepatocytes - portal',
        'Cholangiocytes',
        'Stellate Cells',
        'Portal Fibroblasts',
        'Endothelial Cells'
    ]
    assert sorted(liver_cell_types) == sorted(nault_cell_types)
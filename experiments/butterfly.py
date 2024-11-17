from thesis.datasets import NaultPipeline, NaultSinglePipeline
from thesis.model import ButterflyPipeline


butterfly_nault = ButterflyPipeline(
    dataset_pipeline=NaultSinglePipeline(NaultPipeline(), dosages=0.01),
    experiment_name="playground",
    debug=False,
)

cell_type_key = butterfly_nault.dataset_pipeline.cell_type_key
cell_type_list = list(butterfly_nault.dataset_pipeline.dataset.obs[cell_type_key].cat.categories)
cell_type_index = cell_type_list.index('Hepatocytes - portal')

butterfly_nault(batch=cell_type_index, append_metrics=False, save_plots=False, refresh_training=True, refresh_evaluation=True)
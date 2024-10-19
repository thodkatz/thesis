from anndata import AnnData
from scButterfly.train_model_perturb import Model
from scButterfly.split_datasets import unpaired_split_dataset_perturb, unpaired_split_dataset_perturb_no_reusing
import torch.nn as nn
from typing import List, Callable
from thesis import ROOT
import pandas as pd

def prepare_data_sciplex3(dataset: AnnData):
    dataset.X = dataset.X.toarray()
    
    num_perturbations = dataset.obs.perturbation.unique()
    assert len(num_perturbations) == 2
    dataset.obs['condition'] = dataset.obs['perturbation'].apply(lambda x: 'control' if x == 'control' else 'stimulated')
    
    control = dataset[dataset.obs.condition == 'control']
    perturb = dataset[dataset.obs.condition == 'stimulated']
    
    control.obs["cell_type"] = control.obs["celltype"]
    perturb.obs["cell_type"] = perturb.obs["celltype"]
    control.obs['condition'] = 'control'
    perturb.obs['condition'] = 'stimulated'
    control.obs.index = [str(i) for i in range(control.X.shape[0])]
    perturb.obs.index = [str(i) for i in range(perturb.X.shape[0])]
    return control, perturb
    

def run_dataset(name: str, file_path: str, control: AnnData, perturb: AnnData, id_list: List, batch_list: List[int], train=True):
    metrics_list = []
    for idx, batch in enumerate(batch_list):
        pd_metrics = run_batch(name, file_path, control, perturb, id_list, batch, train)
        pd_metrics['batch'] = idx
        metrics_list.append(pd_metrics)
    metrics = pd.concat(metrics_list)
    return metrics
    
    
def run_batch(name: str, file_path: str, control: AnnData, perturb: AnnData, id_list: List, batch: int, train=True):
    train_id_control, train_id_perturb, validation_id_control, validation_id_perturb, test_id_control, test_id_perturb = id_list[batch]

    RNA_input_dim = control.X.shape[1]
    ATAC_input_dim = perturb.X.shape[1]
    R_kl_div = 1 / RNA_input_dim * 20
    A_kl_div = 1 / ATAC_input_dim * 20
    kl_div = R_kl_div + A_kl_div
    
    file_path = f"{file_path}/batch{batch}"

    model = Model(
        R_encoder_nlayer = 2, 
        A_encoder_nlayer = 2,
        R_decoder_nlayer = 2, 
        A_decoder_nlayer = 2,
        R_encoder_dim_list = [RNA_input_dim, 256, 128],
        A_encoder_dim_list = [ATAC_input_dim, 128, 128],
        R_decoder_dim_list = [128, 256, RNA_input_dim],
        A_decoder_dim_list = [128, 128, ATAC_input_dim],
        R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        translator_embed_dim = 128, 
        translator_input_dim_r = 128,
        translator_input_dim_a = 128,
        translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer = 1,
        discriminator_dim_list_R = [128],
        discriminator_dim_list_A = [128],
        discriminator_act_list = [nn.Sigmoid()],
        dropout_rate = 0.1,
        R_noise_rate = 0.5,
        A_noise_rate = 0.5,
        chrom_list = [],
        logging_path = file_path,
        RNA_data = control,
        ATAC_data = perturb,
        name=f"butterfly/{name}/batch{batch}"
    )

    model.train(
        R_encoder_lr = 0.001,
        A_encoder_lr = 0.001,
        R_decoder_lr = 0.001,
        A_decoder_lr = 0.001,
        R_translator_lr = 0.001,
        A_translator_lr = 0.001,
        translator_lr = 0.001,
        discriminator_lr = 0.005,
        R2R_pretrain_epoch = 100,
        A2A_pretrain_epoch = 100,
        lock_encoder_and_decoder = False,
        translator_epoch = 200,
        patience = 50,
        batch_size = 64,
        r_loss = nn.MSELoss(size_average=True),
        a_loss = nn.MSELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        loss_weight = [1, 1, 1, R_kl_div, A_kl_div, kl_div],
        train_id_r = train_id_control,
        train_id_a = train_id_perturb,
        validation_id_r = validation_id_control, 
        validation_id_a = validation_id_perturb, 
        output_path = file_path,
        seed = 19193,
        kl_mean = True,
        R_pretrain_kl_warmup = 50,
        A_pretrain_kl_warmup = 50,
        translation_kl_warmup = 50,
        load_model = None if train else file_path,
        logging_path = file_path
    )

    test_file_path = f"{file_path}/test"
    pd_metrics = model.test(
        test_id_r = test_id_control,
        test_id_a = test_id_perturb,
        model_path = None,
        load_model = False,
        output_path = test_file_path,
        test_pca = True,
        test_DEGs = True,
        test_R2 = True,
        test_dotplot = True,
        output_data = False,
        return_predict = False
    )

    test_file_path = f"{file_path}/test_validation"
    model.test(
        test_id_r = validation_id_control,
        test_id_a = validation_id_perturb,
        model_path = None,
        load_model = False,
        output_path = test_file_path,
        test_pca = True,
        test_DEGs = True,
        test_R2 = True,
        test_dotplot = True,
        output_data = False,
        return_predict = False
    )

    test_file_path = f"{file_path}/test_train"
    model.test(
        test_id_r = train_id_control,
        test_id_a = train_id_perturb,
        model_path = None,
        load_model = False,
        output_path = test_file_path,
        test_pca = True,
        test_DEGs = True,
        test_R2 = True,
        test_dotplot = True,
        output_data = False,
        return_predict = False
    )
    return pd_metrics
    
def run_sciplex3(name: str, dataset: AnnData):
    _run_sciplex3(name=name, dataset=dataset, split_func=unpaired_split_dataset_perturb)
    
def run_sciplex3_no_reusing(name: str, dataset: AnnData):
    _run_sciplex3(name=name, dataset=dataset, split_func=unpaired_split_dataset_perturb_no_reusing)
    
    
def _run_sciplex3(name: str, dataset: AnnData, split_func: Callable):
    file_path = str(ROOT / "saved_results" / "butterfly" / "perturb" / name)
    batch_list = list(range(0, len(dataset.obs["celltype"].cat.categories)))
    control, perturb = prepare_data_sciplex3(dataset=dataset)
    id_list = split_func(control, perturb)
    pd_metrics = run_dataset(name=name, file_path=file_path, control=control, perturb=perturb, id_list=id_list, batch_list=batch_list)
    pd_metrics.to_csv(f"{file_path}/metrics.csv")
 
from typing import Type, Union
from thesis import SAVED_RESULTS_PATH
from thesis.datasets import (
    NaultMultiplePipeline,
    NaultSinglePipeline,
    NaultPipeline,
)
from thesis.model import ButterflyPipeline, MultiTaskAaeAdversarialPipeline, MultiTaskAaeAutoencoderPipeline, MultiTaskVaeAutoencoderPipeline
from torch import nn


def count_all_parameters(model: nn.Module, debug: bool = False):
    total_count = sum(p.numel() for p in model.parameters())    
    if not debug:
        return total_count
    count = 0
    print("-------------------")
    print("Model", model)
    for name, parameters in model.named_parameters():
        print("Name", name)
        print("number", parameters.numel())
        count += parameters.numel()
    for name, child in model.named_children():
        print("Name", name)
        print("child", child)
        # if hasattr(child, "weight") and child.weight is not None:
        #     print("Child weight shape:", child.weight.shape)
        # else:
        #     print("Child has no weight")
    print("total count", total_count)
    print("debug count", count)    
    print("-------------------")
    return total_count


def count_butterfly(dosage: float):
    butterfly_pipleline = ButterflyPipeline(
        dataset_pipeline=NaultSinglePipeline(NaultPipeline(), dosages=dosage),
        experiment_name="compute_efficiency",
    )

    file_path = SAVED_RESULTS_PATH / "count_parameters_buterfly"
    tensorboard_path = SAVED_RESULTS_PATH / "runs" / "count_parameters_butterfly"

    model = butterfly_pipleline.get_model(
        file_path=str(file_path), tensorboard_path=tensorboard_path
    )
    
    model.set_eval()
        
        
    count = 0
    
    for model in [model.RNA_encoder,
                  model.ATAC_decoder,
                  model.translator.RNA_encoder_l_mu,
                  model.translator.RNA_encoder_bn_mu,
                  model.translator.RNA_encoder_bn_d,
                  model.translator.RNA_encoder_l_d,
                  model.translator.RNA_decoder_bn,
                  model.translator.RNA_decoder_l,
                  model.translator.ATAC_decoder_bn,
                  model.translator.ATAC_decoder_l]:
        count += count_all_parameters(model, debug=True)

    return count


def count_multi_task_autoencoder(
    multi_task_pipeline_class: Union[
        Type[MultiTaskAaeAutoencoderPipeline], Type[MultiTaskVaeAutoencoderPipeline]
    ],
):
    multi_task_aae_pipeline = multi_task_pipeline_class(
        dataset_pipeline=NaultMultiplePipeline(NaultPipeline()),
        experiment_name="compute_efficiency",
    )

    model = multi_task_aae_pipeline.get_model()
    
    model.eval()
    
    encoder_params = count_all_parameters(model.encoder, debug=True)
    decoder_params = count_all_parameters(model.decoder, debug=True)
    print("encoder params", encoder_params)
    print("decoder params", decoder_params)

    # discrininator not used for multi task aae autoenocder pipeline although it is present in the model
    return encoder_params + decoder_params


def count_multi_task_with_discriminator():
    multi_task_aae_pipeline = MultiTaskAaeAdversarialPipeline(
        dataset_pipeline=NaultMultiplePipeline(NaultPipeline()),
        experiment_name="compute_efficiency",
    )

    model = multi_task_aae_pipeline.get_model()
    
    encoder_params = count_all_parameters(model.encoder)
    decoder_params = count_all_parameters(model.decoder)
    discriminator_params = count_all_parameters(model.discriminator)
    print("encoder params", encoder_params)
    print("decoder params", decoder_params)
    print("discriminator params", discriminator_params)

    return encoder_params + decoder_params + discriminator_params


if __name__ == "__main__":
    nault_dataset = NaultPipeline()
    dosages_unique = nault_dataset.get_dosages_unique()
    print(f"Total number of dosages: {len(dosages_unique)}")
   #counts = count_butterfly(dosage=dosages_unique[0])
    #print(f"Total number of parameters in Butterfly: {counts * len(dosages_unique)}")
    
    counts = count_multi_task_autoencoder(MultiTaskAaeAutoencoderPipeline)
    print(f"Total number of parameters in MultiTaskAaeAutoencoder: {counts}")
    
    #counts = count_multi_task_autoencoder(MultiTaskVaeAutoencoderPipeline)
    print(f"Total number of parameters in MultiTaskVaeAutoencoder: {counts}")
    
    #counts = count_multi_task_with_discriminator()
    print(f"Total number of parameters in MultiTaskAaeAdversarial: {counts}")
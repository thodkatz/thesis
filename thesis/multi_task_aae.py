from __future__ import annotations
import os
from pathlib import Path
from anndata import AnnData
import numpy as np
from torch import nn
from torch import Tensor
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import LambdaLR
from enum import Enum, auto

from tqdm import tqdm

from thesis.datasets import (
    NaultMultiplePipeline,
    NaultSinglePipeline,
)
from torch.utils.tensorboard import SummaryWriter
from scipy import sparse

from thesis.utils import SEED, seed_worker


Gamma = Tensor
Beta = Tensor
Dosage = float


class WeightInit(Enum):
    XAVIER = auto()
    KAIMING = auto()


class LayerActivation(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    SIGMOID = auto()
    SOFTMAX = auto()


class WeightNorm(Enum):
    BATCH = auto()
    LAYER = auto()
    SPECTRA = auto()


class FilmGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        layers_dims = [input_dim] + hidden_layer_dims
        last_hidden_dim = layers_dims[-1]

        if len(hidden_layer_dims) > 0:
            self._hidden_network = NetworkBlock(
                input_dim=input_dim,
                hidden_layer_dims=hidden_layer_dims[:-1],
                output_dim=last_hidden_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self._hidden_network = nn.Identity()

        self.gamma = NetworkBlock(
            input_dim=last_hidden_dim, hidden_layer_dims=[], output_dim=output_dim
        )

        self.beta = NetworkBlock(
            input_dim=last_hidden_dim, hidden_layer_dims=[], output_dim=output_dim
        )

    def forward(self, condition: Tensor) -> Tuple[Gamma, Beta]:
        condition = self._hidden_network(condition)
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma, beta

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  hidden_network={self._hidden_network},\n"
            f"  gamma={self.gamma},\n"
            f"  beta={self.beta}\n"
            f")"
        )


class FilmLayer(nn.Module):
    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        return gamma * x + beta

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"


class NetworkBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: List[int],
        output_dim: int,
        hidden_activation: LayerActivation = LayerActivation.LEAKY_RELU,
        output_activation: Optional[LayerActivation] = LayerActivation.LEAKY_RELU,
        norm_type: WeightNorm = WeightNorm.BATCH,
        dropout_rate: float = 0.1,
        mask_rate: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = hidden_layer_dims

        self.norm_type = norm_type
        self.weight_init_type = WeightInit.XAVIER

        if mask_rate is not None:
            self.mask_layer = nn.Dropout(mask_rate)
        else:
            self.mask_layer = nn.Identity()

        layers_dim = [input_dim] + hidden_layer_dims + [output_dim]
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for idx in range(len(hidden_layer_dims)):
            layer = nn.Linear(layers_dim[idx], layers_dim[idx + 1])
            self.weight_init(layer, self.weight_init_type)

            if self.norm_type == WeightNorm.BATCH:
                self.hidden_layers.append(layer)
                self.norm_layers.append(nn.BatchNorm1d(layers_dim[idx + 1]))
            elif self.norm_type == WeightNorm.LAYER:
                self.hidden_layers.append(layer)
                self.norm_layers.append(nn.LayerNorm(layers_dim[idx + 1]))
            elif self.norm_type == WeightNorm.SPECTRA:
                self.hidden_layers.append(spectral_norm(layer))
                self.norm_layers.append(nn.Identity())
            else:
                raise NotImplementedError

            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(layers_dim[-2], layers_dim[-1])
        self.weight_init(self.output_layer, self.weight_init_type)

        self.hidden_activation = self.get_activation(hidden_activation)

        if output_activation is not None:
            self.output_activation = self.get_activation(output_activation)
        else:
            self.output_activation = None

        assert (
            len(self.dropout_layers)
            == len(self.hidden_layers)
            == len(self.norm_layers)
            == len(hidden_layer_dims)
        )

    @staticmethod
    def get_activation(activation: LayerActivation):
        if activation == LayerActivation.RELU:
            return nn.ReLU()
        elif activation == LayerActivation.LEAKY_RELU:
            return nn.LeakyReLU()
        elif activation == LayerActivation.SIGMOID:
            return nn.Sigmoid()
        elif activation == LayerActivation.SOFTMAX:
            return nn.Softmax(dim=-1)
        else:
            raise NotImplementedError

    @staticmethod
    def weight_init(layer: nn.Linear, weight_init: WeightInit):
        if weight_init == WeightInit.XAVIER:
            nn.init.xavier_uniform_(layer.weight)
        elif weight_init == WeightInit.KAIMING:
            nn.init.kaiming_uniform_(layer.weight)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        x = self.mask_layer(x)

        for layer, norm, dropout in zip(
            self.hidden_layers, self.norm_layers, self.dropout_layers
        ):
            x = layer(x)
            x = norm(x)
            x = self.hidden_activation(x)
            x = dropout(x)
        return self._forward_output_layer(x)

    def _forward_output_layer(self, x: Tensor) -> Tensor:
        if self.output_activation is None:
            return self.output_layer(x)
        else:
            return self.output_activation(self.output_layer(x))

    @classmethod
    def create_symmetric_encoder_decoder(
        cls, input_dim: int, hidden_layers: List[int]
    ) -> Tuple[NetworkBlock, NetworkBlock]:
        reverse_hidden_layers = hidden_layers[::-1]
        encoder = NetworkBlock(
            input_dim=input_dim,
            hidden_layer_dims=hidden_layers,
            output_dim=hidden_layers[-1],
            mask_rate=0.5,
        )
        decoder = NetworkBlock(
            input_dim=reverse_hidden_layers[0],
            hidden_layer_dims=reverse_hidden_layers,
            output_dim=input_dim,
        )
        return encoder, decoder

    @classmethod
    def create_discriminator(
        cls,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        norm_type: WeightNorm = WeightNorm.SPECTRA,
    ):
        return NetworkBlock(
            input_dim=input_dim,
            hidden_layer_dims=hidden_layers,
            output_dim=output_dim,
            norm_type=norm_type,
            output_activation=None,  # we use BCEWithLogitsLoss that integrates sigmoid
        )

    def __str__(self) -> str:
        hidden_layers = ", ".join([str(layer) for layer in self.hidden_layers])
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_layers=[{hidden_layers}],\n"
            f"  output_layer={self.output_layer},\n"
            f"  weight_init={self.weight_init_type},\n"
            f"  norm_type={self.norm_type},\n"
            f"  hidden_activation={self.hidden_activation},\n"
            f"  output_activation={self.output_activation}\n"
            f")"
        )


class NetworkBlockFilm(NetworkBlock):
    def __init__(
        self,
        input_dim: int,
        film_layer_factory: FilmLayerFactory,
        hidden_layers: List[int],
        output_dim: int,
        hidden_activation: LayerActivation = LayerActivation.LEAKY_RELU,
        output_activation: Optional[LayerActivation] = LayerActivation.LEAKY_RELU,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_layer_dims=hidden_layers,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

        self.film_layer_factory = film_layer_factory

    def forward(self, x: Tensor, dosages: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
            film_generator = self.film_layer_factory.create_film_generator(
                dim=x.shape[1]
            )
            gamma, beta = film_generator(dosages)
            film_layer = FilmLayer()
            x = film_layer(x, gamma, beta)
            x = self.hidden_activation(x)
        return self._forward_output_layer(x)

    @classmethod
    def create_encoder_decoder_with_film(
        cls,
        input_dim: int,
        hidden_layers: List[int],
        film_layer_factory: FilmLayerFactory,
    ):
        reverse_hidden_layers = hidden_layers[::-1]
        encoder = NetworkBlock(
            input_dim=input_dim,
            hidden_layer_dims=hidden_layers,
            output_dim=hidden_layers[-1],
            mask_rate=0.5,
        )
        decoder = NetworkBlockFilm(
            input_dim=reverse_hidden_layers[0],
            hidden_layers=reverse_hidden_layers,
            output_dim=input_dim,
            film_layer_factory=film_layer_factory,
        )
        return encoder, decoder

    def __str__(self):
        network_block = super().__str__()
        return (
            f"{self.__class__.__name__}(\n"
            f"  {network_block},\n"
            f"  film_layer_factory={self.film_layer_factory}\n"
            f")"
        )


class FilmLayerFactory:
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.device = device

    def create_film_generator(self, dim: int) -> FilmGenerator:
        film_generator = FilmGenerator(
            input_dim=self.input_dim,
            hidden_layer_dims=self.hidden_layers,
            output_dim=dim,
        ).to(self.device)
        return film_generator

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_layers={self.hidden_layers},\n"
            f"  device={self.device}\n"
            f")"
        )


class MultiTaskAae(nn.Module):
    """
    Multi-task Adversarial Autoencoder

    """

    def __init__(
        self,
        num_features: int,
        hidden_layers_autoencoder: List[int],
        hidden_layers_discriminator: List[int],
        film_layer_factory: FilmLayerFactory,
    ):
        super().__init__()

        self.encoder, self.decoder = NetworkBlockFilm.create_encoder_decoder_with_film(
            input_dim=num_features,
            hidden_layers=hidden_layers_autoencoder,
            film_layer_factory=film_layer_factory,
        )

        self.discriminator = NetworkBlock.create_discriminator(
            input_dim=self.get_latent_dim(),
            hidden_layers=hidden_layers_discriminator,
            output_dim=1,  # control, not-control (perturbed)
        )

    def forward(self, x: Tensor, dosages: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x, dosages)
        return x

    def get_latent_representation(self, x: Tensor):
        with torch.no_grad():
            return self.encoder(x).detach().cpu().numpy()

    def get_latent_dim(self):
        return self.encoder.output_dim

    @classmethod
    def load(
        cls,
        num_features: int,
        hidden_layers_autoencoder: List[int],
        hidden_layers_discriminator: List[int],
        film_layer_factory: FilmLayerFactory,
        load_path: Path,
    ):
        model = MultiTaskAae(
            num_features=num_features,
            hidden_layers_autoencoder=hidden_layers_autoencoder,
            hidden_layers_discriminator=hidden_layers_discriminator,
            film_layer_factory=film_layer_factory,
        )
        model.load_state_dict(torch.load(load_path))
        return model

    def save(self, path: Path):
        os.makedirs(path.parent, exist_ok=True)
        torch.save(self.state_dict(), path)

        config_path = path.parent / "config.txt"
        with open(config_path, "w") as f:
            f.write(str(self))

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  encoder={self.encoder},\n"
            f"  decoder={self.decoder},\n"
            f"  discriminator={self.discriminator}\n"
            f")"
        )


class Aae(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_layers: List[int],
    ):
        super().__init__()

        self.encoder, self.decoder = NetworkBlock.create_symmetric_encoder_decoder(
            input_dim=num_features,
            hidden_layers=hidden_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent_representation(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.encoder(x)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  encoder={self.encoder},\n"
            f"  decoder={self.decoder}\n"
            f")"
        )


class NaultDataset(Dataset):
    def __init__(
        self,
        dataset_condition_pipeline: Union[NaultMultiplePipeline, NaultSinglePipeline],
        target_cell_type: str,
    ):
        self._dataset_condition_pipeline = dataset_condition_pipeline
        self.control_dose = self._dataset_condition_pipeline.control_dose
        self.target_cell_type = target_cell_type
        train_adata = dataset_condition_pipeline.get_train(
            target_cell_type=target_cell_type
        )
        print("Train data size:", len(train_adata))

        self.gene_expressions = self.get_gene_expressions(train_adata)

        self._dosages_unique = dataset_condition_pipeline.dataset_pipeline.get_dosages()
        self._dosage_to_idx = {
            dosage: idx for idx, dosage in enumerate(self._dosages_unique)
        }
        dosages = train_adata.obs[self._dataset_condition_pipeline.dosage_key].values

        self.is_control = torch.tensor(
            [(self.get_soft_labels_control(dosage)) for dosage in dosages]
        )

        self.dosages = self.get_one_hot_encoded_dosages(dosages)

        assert len(self.gene_expressions) == len(self.dosages)

        # sanity check
        dosages_to_test = self.get_dosages_to_test()
        dosages_to_train = self.get_dosages_to_train()
        dosages_to_train.remove(self.control_dose)
        assert dosages_to_test == dosages_to_train

    def get_soft_labels_control(self, dosage: float):
        if dosage == self.control_dose:
            return np.random.uniform(low=0.7, high=1)
        else:
            return np.random.uniform(low=0, high=0.3)

    def get_gene_expressions(self, adata: AnnData) -> Tensor:
        if sparse.issparse(adata.X):
            gene_expressions = adata.X.toarray()
        else:
            gene_expressions = adata.X

        return torch.tensor(gene_expressions)

    def get_one_hot_encoded_dosages(self, dosages: List[float]):
        return torch.eye(self.get_condition_len())[
            [self._dosage_to_idx[dosage] for dosage in dosages]
        ]

    def get_num_features(self) -> int:
        return self.gene_expressions.shape[1]

    def get_condition_labels(self) -> List[float]:
        return self._dosages_unique

    def get_condition_len(self) -> int:
        return len(self._dosages_unique)

    def get_dosages_to_test(self):
        dose_key = self._dataset_condition_pipeline.dosage_key
        return sorted(self.get_stim_test().obs[dose_key].unique().tolist())

    def get_stim_test(self, dose: Optional[float] = None):
        stim_test = self._dataset_condition_pipeline.get_stim_test(
            target_cell_type=self.target_cell_type
        )
        if dose is not None:
            return stim_test[
                stim_test.obs[self._dataset_condition_pipeline.dosage_key] == dose
            ]
        else:
            return stim_test

    def get_ctrl_test(self):
        return self._dataset_condition_pipeline.get_ctrl_test(
            target_cell_type=self.target_cell_type
        )

    def get_train(self):
        return self._dataset_condition_pipeline.get_train(
            target_cell_type=self.target_cell_type
        )

    def get_dosages_to_train(self):
        dose_key = self._dataset_condition_pipeline.dosage_key
        return sorted(self.get_train().obs[dose_key].unique().tolist())

    def __getitem__(self, index):
        return self.gene_expressions[index], self.dosages[index], self.is_control[index]

    def __len__(self):
        return len(self.gene_expressions)


class MultiTaskAdversarialAutoencoderUtils:
    def __init__(
        self,
        dataset: NaultDataset,
        model: MultiTaskAae,
    ):

        self.model = model
        self.dataset = dataset
        self.device = "cuda"
        self.model.to(self.device)

    def train(
        self,
        save_path: Path,
        tensorboard_path: Path,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        is_adversarial: bool = True,
    ):
        print("Torch seed", torch.initial_seed())
        os.makedirs(tensorboard_path, exist_ok=True)

        generator = torch.Generator()
        generator.manual_seed(SEED)

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        self._train_alternating(
            dataloader=dataloader,
            tensorboard_path=tensorboard_path,
            epochs=epochs,
            lr=lr,
            is_adversarial=is_adversarial,
        )

        self.model.save(save_path)

    def _train_alternating(
        self,
        dataloader,
        tensorboard_path: Path,
        epochs: int,
        lr: float,
        is_adversarial: bool = True,  # fix: flag argument
    ):
        writer = SummaryWriter(tensorboard_path)

        optimizer_encoder = torch.optim.Adam(self.model.encoder.parameters(), lr=lr)
        optimizer_decoder = torch.optim.Adam(self.model.decoder.parameters(), lr=lr)
        optimizer_discriminator = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr
        )

        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        warmup_steps = 100

        def warmup(step):
            if step < warmup_steps:
                # Linear warm-up
                return step / warmup_steps
            else:
                # cosine annealing?
                return 1

        scheduler_encoder = LambdaLR(optimizer_encoder, lr_lambda=warmup)
        scheduler_decoder = LambdaLR(optimizer_decoder, lr_lambda=warmup)
        scheduler_discriminator = LambdaLR(optimizer_discriminator, lr_lambda=warmup)

        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            reconstruction_loss_batches = []
            adversarial_loss_batches = []
            total_loss_batches = []
            discriminator_loss_batches = []
            for gene_expressions, dosages, is_control in dataloader:
                gene_expressions = gene_expressions.to(self.device)
                dosages = dosages.to(self.device)
                is_control = is_control.to(self.device)
                is_control = torch.reshape(is_control, (is_control.shape[0], 1))

                """
                Reconstruction loss
                """

                latent = self.model.encoder(gene_expressions)
                decoder_output = self.model.decoder(latent, dosages)

                reconstruction_loss = mse(decoder_output, gene_expressions)
                reconstruction_loss_batches.append(reconstruction_loss.item())

                if not is_adversarial:
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    reconstruction_loss.backward()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    scheduler_decoder.step()
                    scheduler_encoder.step()
                    continue

                """
                Adversarial loss
                """

                generator_output = latent
                discriminator_output = self.model.discriminator(generator_output)
                adv_loss = bce(discriminator_output, is_control)

                def is_discriminator_good_enough(d_loss):
                    return d_loss < 1

                if is_discriminator_good_enough(adv_loss):
                    coeff = 1
                    total_loss = reconstruction_loss - coeff * adv_loss
                else:
                    total_loss = reconstruction_loss

                adversarial_loss_batches.append(adv_loss.item())
                total_loss_batches.append(total_loss.item())

                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                total_loss.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()
                scheduler_encoder.step()
                scheduler_decoder.step()

                """
                Discriminator loss
                """

                disc_input = self.model.encoder(gene_expressions).detach()
                prediction_control_classification = self.model.discriminator(disc_input)
                discriminator_loss = bce(prediction_control_classification, is_control)
                discriminator_loss_batches.append(discriminator_loss.item())

                optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                optimizer_discriminator.step()
                scheduler_discriminator.step()

            reconstruction_loss = np.mean(reconstruction_loss_batches)
            writer.add_scalar("reconstruction_loss", reconstruction_loss, epoch)

            tqdm_text = f"Epoch [{epoch + 1}/{epochs}], rc loss: {reconstruction_loss}"
            if is_adversarial:
                adv_loss = np.mean(adversarial_loss_batches)
                total_loss = np.mean(total_loss_batches)
                discriminator_loss = np.mean(discriminator_loss_batches)

                writer.add_scalar("adv_loss", adv_loss, epoch)
                writer.add_scalar("total_loss", total_loss, epoch)
                writer.add_scalar("discriminator_loss", discriminator_loss, epoch)

                tqdm_text += f", adv loss: {adv_loss}, total loss: {total_loss}, discriminator loss: {discriminator_loss}"

            tqdm.write(tqdm_text)

    def predict(self):
        control_test_adata = self.dataset.get_ctrl_test()
        dosages_to_test = self.dataset.get_dosages_to_test()
        predictions = {}
        gene_expressions = self.dataset.get_gene_expressions(control_test_adata)

        self.model.eval()
        with torch.no_grad():
            for dosage in dosages_to_test:
                dosages_one_hot_encoded = self.dataset.get_one_hot_encoded_dosages(
                    [dosage] * len(control_test_adata)
                )
                gene_expressions = gene_expressions.to(self.device)
                dosages_one_hot_encoded = dosages_one_hot_encoded.to(self.device)

                predictions[dosage] = AnnData(
                    X=self.model(gene_expressions, dosages_one_hot_encoded)
                    .cpu()
                    .numpy(),
                    obs=control_test_adata.obs.copy(),
                    var=control_test_adata.var.copy(),
                    obsm=control_test_adata.obsm.copy(),
                )

        # assumption: returns sorted based on drug dosage to test excluding control
        return list(predictions.values())

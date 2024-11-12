from __future__ import annotations
import os
from pathlib import Path
import random
from anndata import AnnData
import numpy as np
from torch import nn
from torch import Tensor
from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from enum import Enum

from tqdm import tqdm

from thesis.datasets import (
    NaultMultiplePipeline,
)
from torch.utils.tensorboard import SummaryWriter
from scipy import sparse

from thesis.utils import SEED, seed_worker


Gamma = Tensor
Beta = Tensor
Dosage = float


class WeightInit(Enum):
    XAVIER = 0
    KAIMING = 1


class LayerActivation(Enum):
    RELU = 0
    LEAKY_RELU = 1
    SIGMOID = 2


class WeightNorm(Enum):
    BATCH = 0
    LAYER = 1


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


class FilmLayer(nn.Module):
    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        return gamma * x + beta


class Discriminator(nn.Module):
    pass


class NetworkBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: List[int],
        output_dim: int,
        hidden_activation: LayerActivation = LayerActivation.RELU,
        output_activation: LayerActivation = LayerActivation.RELU,
        norm_type: WeightNorm = WeightNorm.BATCH,
        dropout_rate: float = 0.1,
        mask_rate: Optional[float] = None,
    ):
        super().__init__()
        self.norm_type = norm_type

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
            self.weight_init(layer)

            if self.norm_type == WeightNorm.BATCH:
                self.hidden_layers.append(layer)
                self.norm_layers.append(nn.BatchNorm1d(layers_dim[idx + 1]))
            elif self.norm_type == WeightNorm.LAYER:
                self.hidden_layers.append(layer)
                self.norm_layers.append(nn.LayerNorm(layers_dim[idx + 1]))
                raise NotImplementedError

            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(layers_dim[-2], layers_dim[-1])
        self.weight_init(self.output_layer)

        self.hidden_activation = self.get_activation(hidden_activation)
        self.output_activation = self.get_activation(output_activation)

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
        else:
            raise NotImplementedError

    @staticmethod
    def weight_init(layer: nn.Linear, weight_init: WeightInit = WeightInit.XAVIER):
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


class NetworkBlockFilm(NetworkBlock):
    def __init__(
        self,
        input_dim: int,
        film_layer_factory: FilmLayerFactory,
        hidden_layers: List[int],
        output_dim: int,
        hidden_activation: LayerActivation = LayerActivation.RELU,
        output_activation: LayerActivation = LayerActivation.RELU,
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
        return self.output_activation(self.output_layer(x))

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


class MultiTaskAae(nn.Module):
    """
    Multi-task Adversarial Autoencoder

    """

    def __init__(
        self,
        num_features: int,
        hidden_layers: List[int],
        film_layer_factory: FilmLayerFactory,
    ):
        super().__init__()

        self.encoder, self.decoder = NetworkBlockFilm.create_encoder_decoder_with_film(
            input_dim=num_features,
            hidden_layers=hidden_layers,
            film_layer_factory=film_layer_factory,
        )

    def forward(self, x: Tensor, dosages: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x, dosages)
        return x

    def get_latent_representation(self, x: Tensor):
        with torch.no_grad():
            return self.encoder(x).detach().cpu().numpy()

    @classmethod
    def load(
        cls,
        num_features: int,
        hidden_layers: List[int],
        film_layer_factory: FilmLayerFactory,
        load_path: Path,
    ):
        model = MultiTaskAae(
            num_features=num_features,
            hidden_layers=hidden_layers,
            film_layer_factory=film_layer_factory,
        )
        model.load_state_dict(torch.load(load_path))
        return model

    def save(self, path: Path):
        os.makedirs(path.parent, exist_ok=True)
        torch.save(self.state_dict(), path)


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


class NaultDataset(Dataset):
    def __init__(
        self, dataset_condition_pipeline: NaultMultiplePipeline, target_cell_type: str
    ):
        self._dataset_condition_pipeline = dataset_condition_pipeline
        self._control_dose = self._dataset_condition_pipeline.control_dose
        self._target_cell_type = target_cell_type
        train_adata = dataset_condition_pipeline.get_train(
            target_cell_type=target_cell_type
        )

        self.gene_expressions = self.get_gene_expressions(train_adata)

        self._dosages_unique = dataset_condition_pipeline.dataset_pipeline.get_dosages()
        self._dosage_to_idx = {
            dosage: idx for idx, dosage in enumerate(self._dosages_unique)
        }
        dosages = train_adata.obs[self._dataset_condition_pipeline.dosage_key].values
        self.dosages = self.get_one_hot_encoded_dosages(dosages)

        assert len(self.gene_expressions) == len(self.dosages)

        # sanity check
        dosages_to_test = self.get_dosages_to_test()
        dosages_to_train = self.get_dosages_to_train()
        dosages_to_train.remove(self._control_dose)
        assert dosages_to_test == dosages_to_train

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
            target_cell_type=self._target_cell_type
        )
        if dose is not None:
            return stim_test[
                stim_test.obs[self._dataset_condition_pipeline.dosage_key] == dose
            ]
        else:
            return stim_test

    def get_ctrl_test(self):
        return self._dataset_condition_pipeline.get_ctrl_test(
            target_cell_type=self._target_cell_type
        )

    def get_train(self):
        return self._dataset_condition_pipeline.get_train(
            target_cell_type=self._target_cell_type
        )

    def get_dosages_to_train(self):
        dose_key = self._dataset_condition_pipeline.dosage_key
        return sorted(self.get_train().obs[dose_key].unique().tolist())

    def __getitem__(self, index):
        return self.gene_expressions[index], self.dosages[index]

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
        epochs: int = 200,
        batch_size: int = 512,
        lr: float = 0.0001,
    ):
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            for gene_expressions, dosages in dataloader:
                self.model.train()

                gene_expressions = gene_expressions.to(self.device)
                dosages = dosages.to(self.device)
                outputs = self.model(gene_expressions, dosages)

                reconstruction_loss = mse(outputs, gene_expressions)
                # print("Reconstruction loss:", reconstruction_loss.item())

                optimizer.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()

                writer.add_scalar(
                    "reconstruction_loss", reconstruction_loss.item(), epoch
                )
            tqdm.write(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {reconstruction_loss.item()}"
            )

        self.model.save(save_path)

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

from __future__ import annotations
from abc import ABC, abstractmethod
import os
from pathlib import Path
from anndata import AnnData
import numpy as np
from torch import nn
from torch import Tensor
from typing import List, Optional, Tuple, Union, Type
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from enum import Enum, auto

from tqdm import tqdm

from thesis.datasets import (
    DatasetPipeline,
    NaultMultiplePipeline,
    NaultPipeline,
    NaultSinglePipeline,
    SplitDatasetPipeline,
)
from torch.utils.tensorboard import SummaryWriter
from scipy import sparse

from thesis.evaluation import evaluation_out_of_sample
from thesis.utils import SeedSingleton, append_csv, pretty_print
from thesis import ROOT
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


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
        return pretty_print(self)


class FilmLayer(nn.Module):
    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        return gamma * x + beta

    def __str__(self) -> str:
        return pretty_print(self)


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
        return pretty_print(self)


class NetworkBlockFilm(NetworkBlock):
    def __init__(
        self,
        input_dim: int,
        film_layer_factory: FilmLayerFactory,
        hidden_layers: List[int],
        output_dim: int,
        hidden_activation: LayerActivation = LayerActivation.LEAKY_RELU,
        output_activation: Optional[LayerActivation] = LayerActivation.LEAKY_RELU,
        dropout_rate: float = 0.1,
        norm_type: WeightNorm = WeightNorm.BATCH,
        mask_rate: Optional[float] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_layer_dims=hidden_layers,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout_rate = dropout_rate,
            norm_type=norm_type
        )

        self.film_layer_factory = film_layer_factory

    def forward(self, x: Tensor, dosages: Tensor) -> Tensor:
        x = self.mask_layer(x)
        
        for layer, norm, dropout in zip(self.hidden_layers, self.norm_layers, self.dropout_layers):
            x = layer(x)
            film_generator = self.film_layer_factory.create_film_generator(
                dim=x.shape[1]
            )
            gamma, beta = film_generator(dosages)
            film_layer = FilmLayer()
            x = norm(x)
            x = film_layer(x, gamma, beta)
            x = self.hidden_activation(x)
            x = dropout(x)
        return self._forward_output_layer(x)

    @classmethod
    def create_encoder_decoder_with_film(
        cls,
        input_dim: int,
        hidden_layers: List[int],
        film_layer_factory: FilmLayerFactory,
        dropout_rate: float = 0.1,
        mask_rate: float = 0.5,
    ):
        reverse_hidden_layers = hidden_layers[::-1]
        encoder = NetworkBlock(
            input_dim=input_dim,
            hidden_layer_dims=hidden_layers,
            output_dim=hidden_layers[-1],
            mask_rate=mask_rate,
            dropout_rate=dropout_rate,
        )
        decoder = NetworkBlockFilm(
            input_dim=reverse_hidden_layers[0],
            hidden_layers=reverse_hidden_layers,
            output_dim=input_dim,
            film_layer_factory=film_layer_factory,
            dropout_rate=dropout_rate,
        )
        return encoder, decoder
    
    @classmethod
    def create_encoder_with_film_decoder_with_film(
        cls,
        input_dim: int,
        hidden_layers: List[int],
        film_layer_factory: FilmLayerFactory,
        dropout_rate: float = 0.1,
        mask_rate: float = 0.5,
    ):
        reverse_hidden_layers = hidden_layers[::-1]
        encoder = NetworkBlockFilm(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=hidden_layers[-1],
            film_layer_factory=film_layer_factory,
            dropout_rate=dropout_rate,
            mask_rate=mask_rate,
        )
        decoder = NetworkBlockFilm(
            input_dim=reverse_hidden_layers[0],
            hidden_layers=reverse_hidden_layers,
            output_dim=input_dim,
            film_layer_factory=film_layer_factory,
            dropout_rate=dropout_rate,
        )
        return encoder, decoder

    def __str__(self) -> str:
        return pretty_print(self)


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

    def __str__(self) -> str:
        return pretty_print(self)


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
        mask_rate: float = 0.5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder, self.decoder = NetworkBlockFilm.create_encoder_decoder_with_film(
            input_dim=num_features,
            hidden_layers=hidden_layers_autoencoder,
            film_layer_factory=film_layer_factory,
            mask_rate=mask_rate,
            dropout_rate=dropout_rate,
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
        mask_rate: float,
        dropout_rate: float,
        load_path: Path,
    ):
        model = MultiTaskAae(
            num_features=num_features,
            hidden_layers_autoencoder=hidden_layers_autoencoder,
            hidden_layers_discriminator=hidden_layers_discriminator,
            film_layer_factory=film_layer_factory,
            mask_rate=mask_rate,
            dropout_rate=dropout_rate
        )
        load_path = load_path / "model.pt"
        model.load_state_dict(torch.load(load_path))
        return model

    def save(self, path: Path):
        os.makedirs(path, exist_ok=True)
        model_path = path / "model.pt"
        torch.save(self.state_dict(), model_path)

    def __str__(self) -> str:
        return pretty_print(self)


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

    def __str__(self) -> str:
        return pretty_print(self)


class DosagesDataset(Dataset):
    def __init__(
        self,
        dataset_pipeline: DatasetPipeline,
        adata: AnnData,
    ):
        self._dataset_pipeline = dataset_pipeline
        self.control_dose = self._dataset_pipeline.control_dose

        self._low_control_label = 0.9
        self._high_control_label = 1.0
        self._high_perturbed_label = 1 - self._low_control_label
        self._low_perturbed_label = 0.0

        self.gene_expressions = self.get_gene_expressions(adata)

        self._dosages_unique = self._dataset_pipeline.get_dosages_unique()
        self._dosage_to_idx = {
            dosage: idx for idx, dosage in enumerate(self._dosages_unique)
        }
        dosages_numeric = adata.obs[self._dataset_pipeline.dosage_key].values

        self.is_control_soft_labels = torch.tensor(
            [(self.get_soft_labels_control(dosage)) for dosage in dosages_numeric]
        )

        self.dosages_one_hot_encoded = self.get_one_hot_encoded_dosages(dosages_numeric)

        assert len(self.gene_expressions) == len(self.dosages_one_hot_encoded)

    def get_soft_labels_control(self, dosage: float):
        if dosage == self.control_dose:
            return np.random.uniform(
                low=self._low_control_label, high=self._high_control_label
            )
        else:
            return np.random.uniform(
                low=self._low_perturbed_label, high=self._high_perturbed_label
            )

    @staticmethod
    def get_gene_expressions(adata: AnnData) -> Tensor:
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

    def get_dosages(self, adata: AnnData):
        self._dataset_pipeline.get_dosages_unique(adata=adata)

    def get_ctrl_bool_idx(self, dosages_one_hot_encoded: Tensor):
        assert len(dosages_one_hot_encoded.shape) == 2
        assert dosages_one_hot_encoded.shape[1] == self.get_condition_len()
        device = dosages_one_hot_encoded.device
        control_one_hot_encoded = self.get_one_hot_encoded_dosages(
            [self.control_dose]
        ).to(device)
        ctrl_idx = (dosages_one_hot_encoded == control_one_hot_encoded).all(dim=1)
        return ctrl_idx

    def __getitem__(self, index):
        return (
            self.gene_expressions[index],
            self.dosages_one_hot_encoded[index],
            self.is_control_soft_labels[index],
        )

    def __len__(self):
        return len(self.gene_expressions)

    def __str__(self):
        return f"Dataset(target: high_control: {self._high_control_label}, low_control: {self._low_control_label}, high_perturbed: {self._high_perturbed_label}, low_perturbed: {self._low_perturbed_label}, high_perturbed: {self._high_perturbed_label}"


class Trainer(ABC):
    def __init__(
        self,
        model: MultiTaskAae,
        split_dataset_pipeline: SplitDatasetPipeline,
        target_cell_type: str,
        train_tensorboard: Union[Path, SummaryWriter],
        val_tensorboard: Union[Path, SummaryWriter],
        batch_size: int,
        lr: float,
        device: str = "cuda",
        seed: int = 19193,
    ):
        self.model = model
        self.device = device
        self.target_cell_type = target_cell_type
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr


        def get_writer(tensorboard: Union[Path, SummaryWriter]):
            if isinstance(tensorboard, Path):
                os.makedirs(tensorboard, exist_ok=True)
                return SummaryWriter(tensorboard)
            else:
                return tensorboard

        self.writer_train = get_writer(train_tensorboard)
        self.writer_val = get_writer(val_tensorboard)

        generator = torch.Generator()
        
        if not SeedSingleton.is_set():
            SeedSingleton(seed=seed)
        else:
            print("seed already set", SeedSingleton.get_value())
            print("ModelPipeline: Torch initial seed", torch.random.initial_seed())
                        
        generator.manual_seed(SeedSingleton.get_value())
        
        print("Torch initial seed", torch.initial_seed())

        train_adata, validation_adata = (
            split_dataset_pipeline.split_dataset_to_train_validation(
                target_cell_type=target_cell_type
            )
        )
        self.train_dataset = DosagesDataset(
            dataset_pipeline=split_dataset_pipeline.dataset_pipeline, adata=train_adata
        )

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=SeedSingleton.get_dataloader_worker,
            generator=generator,
        )

        generator = torch.Generator()
        generator.manual_seed(SeedSingleton.get_value() + 1)

        self.validation_dataset = DosagesDataset(
            dataset_pipeline=split_dataset_pipeline.dataset_pipeline,
            adata=validation_adata,
        )

        self.validation_dataloader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=SeedSingleton.get_dataloader_worker,
            generator=generator,
        )

        print("train dataset len", len(self.train_dataset))
        print("validation dataset len", len(self.validation_dataset))

    def save(self, path: Path):
        self.model.save(path)
        with open(path / "config.txt", "w") as f:
            f.write(str(self))

    @abstractmethod
    def train(self):
        pass

    def __call__(self):
        self.train()

    def __str__(self) -> str:
        return pretty_print(self)


class MultiTaskAutoencoderTrainer(Trainer):
    def __init__(
        self,
        model: MultiTaskAae,
        split_dataset_pipeline: SplitDatasetPipeline,
        target_cell_type: str,
        train_tensorboard: Union[Path, SummaryWriter],
        val_tensorboard: Union[Path, SummaryWriter],
        device: str = "cuda",
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        seed: int = 19193
    ):

        super().__init__(
            model=model,
            train_tensorboard=train_tensorboard,
            val_tensorboard=val_tensorboard,
            device=device,
            lr=lr,
            batch_size=batch_size,
            split_dataset_pipeline=split_dataset_pipeline,
            target_cell_type=target_cell_type,
            seed=seed
        )

        self.epochs = epochs

        self.mse = nn.MSELoss()

        self.optimizer_encoder = torch.optim.Adam(
            self.model.encoder.parameters(), lr=self.lr
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.model.decoder.parameters(), lr=self.lr
        )

        self.warmup_learning_rate_steps = 100

        def warmup_learning_rate(step):
            if step < self.warmup_learning_rate_steps:
                # Linear warm-up
                return step / self.warmup_learning_rate_steps
            else:
                # cosine annealing?
                return 1

        self.scheduler_encoder = LambdaLR(
            self.optimizer_encoder, lr_lambda=warmup_learning_rate
        )
        self.scheduler_decoder = LambdaLR(
            self.optimizer_decoder, lr_lambda=warmup_learning_rate
        )

    def _step_autoencoder(self, loss: Tensor):
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_decoder.step()
        self.scheduler_decoder.step()
        self.scheduler_encoder.step()

    def _run_per_epoch(self, epoch: int):
        def _run(dataloader: DataLoader, is_train: bool = True):
            reconstruction_loss_batches = []
            for gene_expressions, dosages, is_control in dataloader:
                gene_expressions = gene_expressions.to(self.device)
                dosages = dosages.to(self.device)
                is_control = is_control.to(self.device)
                is_control = torch.reshape(is_control, (is_control.shape[0], 1))

                latent = self.model.encoder(gene_expressions)
                decoder_output = self.model.decoder(latent, dosages)

                rc_loss = self.mse(decoder_output, gene_expressions)
                if is_train:
                    self._step_autoencoder(rc_loss)
                reconstruction_loss_batches.append(rc_loss.item())

            reconstruction_loss = np.mean(reconstruction_loss_batches)
            if is_train:
                writer = self.writer_train
                rc_loss_name = "rc loss"
            else:
                writer = self.writer_val
                rc_loss_name = "rc val loss"
            writer.add_scalar("reconstruction_loss", reconstruction_loss, epoch)
            tqdm_text = f"Epoch [{epoch + 1}/{self.epochs}], {rc_loss_name}: {reconstruction_loss}"
            tqdm.write(tqdm_text)

        self.model.train()
        _run(self.train_dataloader, is_train=True)
        self.model.eval()
        with torch.no_grad():
            _run(self.validation_dataloader, is_train=False)

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self._run_per_epoch(epoch)


class MultiTaskAdversarialAutoencoderTrainer(Trainer):
    def __init__(
        self,
        model: MultiTaskAae,
        split_dataset_pipeline: SplitDatasetPipeline,
        target_cell_type: str,
        tensorboard_path: Path,
        device: str = "cuda",
        coeff_adversarial: float = 0.1,
        discriminator_pretrain_epochs: int = 50,
        autoencoder_pretrain_epochs: int = 50,
        adversarial_epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 64,
        seed: int = 19193,
    ):
        train_tensorboard = tensorboard_path / "train"
        val_tensorboard = tensorboard_path / "val"
        super().__init__(
            model=model,
            train_tensorboard=train_tensorboard,
            val_tensorboard=val_tensorboard,
            device=device,
            lr=lr,
            batch_size=batch_size,
            split_dataset_pipeline=split_dataset_pipeline,
            target_cell_type=target_cell_type,
            seed=seed
        )

        self.coeff_adversarial = coeff_adversarial
        self.coeff_reconstruction = 1.0 - self.coeff_adversarial

        self.adversarial_epochs = adversarial_epochs
        self.discriminator_epochs = discriminator_pretrain_epochs
        self.autoencoder_epochs = autoencoder_pretrain_epochs

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.optimizer_encoder = torch.optim.Adam(
            self.model.encoder.parameters(), lr=lr
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.model.decoder.parameters(), lr=lr
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr
        )

        self.autoencoder_trainer = MultiTaskAutoencoderTrainer(
            model=self.model,
            train_tensorboard=self.writer_train,
            val_tensorboard=self.writer_val,
            device=self.device,
            epochs=self.autoencoder_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            split_dataset_pipeline=split_dataset_pipeline,
            target_cell_type=self.target_cell_type,
            seed=seed
        )

        self.warmup_learning_rate_steps = 100

        def warmup_learning_rate(step):
            if step < self.warmup_learning_rate_steps:
                # Linear warm-up
                return step / self.warmup_learning_rate_steps
            else:
                # cosine annealing?
                return 1

        self.scheduler_encoder_adversarial = LambdaLR(
            self.optimizer_encoder, lr_lambda=warmup_learning_rate
        )
        self.scheduler_decoder_adversarial = LambdaLR(
            self.optimizer_decoder, lr_lambda=warmup_learning_rate
        )
        self.scheduler_discriminator_pretrain = LambdaLR(
            self.optimizer_discriminator, lr_lambda=warmup_learning_rate
        )
        self.scheduler_discriminator_adversarial = LambdaLR(
            self.optimizer_discriminator, lr_lambda=warmup_learning_rate
        )

        # self.scheduler_encoder_plateu = ReduceLROnPlateau(
        #     self.optimizer_encoder, mode="min", factor=0.1, patience=5
        # )
        # self.scheduler_decoder_plateu = ReduceLROnPlateau(
        #     self.optimizer_decoder, mode="min", factor=0.1, patience=5
        # )
        # self.scheduler_discriminator_plateu = ReduceLROnPlateau(
        #     self.optimizer_discriminator, mode="min", factor=0.1, patience=5
        # )

    def _get_discriminator_loss(
        self, gene_expression: Tensor, is_control_soft_labels: Tensor
    ):
        latent = self.model.encoder(gene_expression).detach()
        discriminator_output = self.model.discriminator(latent)

        discriminator_loss = self.bce(input=discriminator_output, target=is_control_soft_labels)
        return discriminator_loss

    def _get_reconstruction_loss(
        self, gene_expression: Tensor, dosages_one_hot_encoded: Tensor
    ):
        latent = self.model.encoder(gene_expression)
        decoder_output = self.model.decoder(latent, dosages_one_hot_encoded)
        reconstruction_loss = self.mse(input=decoder_output, target=gene_expression)
        return reconstruction_loss, latent

    def _get_adversarial_loss(
        self,
        generator_output: Tensor,
        dosages_one_hot_encoded: Tensor,
        is_control_soft_labels: Tensor,
    ) -> Optional[Tensor]:
        """
        Generator (Encoder) tries to fool perturbed as control
        
        If no perturbed samples in the batch, return None
        """
        perturb_idx = ~self.train_dataset.get_ctrl_bool_idx(dosages_one_hot_encoded)
        real_perturb_labels = is_control_soft_labels[perturb_idx]
        if real_perturb_labels.numel() == 0:
            return None
        perturb_latent = generator_output[perturb_idx]
        fake_perturb_labels = torch.ones_like(real_perturb_labels) - real_perturb_labels
        discriminator_output = self.model.discriminator(perturb_latent)
        adv_loss = self.bce(input=discriminator_output, target=fake_perturb_labels)
        return adv_loss
    

    def _get_total_loss(self, rc_loss: Tensor, adv_loss: Tensor):
        total_loss = (
            self.coeff_reconstruction * rc_loss + self.coeff_adversarial * adv_loss
        )
        return total_loss

    def _run_pretrain_discriminator_per_epoch(self, epoch: int):
        def run(dataloader: DataLoader, is_train: bool = True):
            discriminator_loss_batches = []
            for (
                gene_expressions,
                dosages_one_hot_encoded,
                is_control_soft_labels,
            ) in dataloader:
                gene_expressions = gene_expressions.to(self.device)
                dosages_one_hot_encoded = dosages_one_hot_encoded.to(self.device)
                is_control_soft_labels = is_control_soft_labels.to(self.device)
                is_control_soft_labels = torch.reshape(
                    is_control_soft_labels, (is_control_soft_labels.shape[0], 1)
                )

                discriminator_loss = self._get_discriminator_loss(
                    gene_expressions, is_control_soft_labels
                )

                if is_train:
                    self._step_pretrain_discriminator(discriminator_loss)

                discriminator_loss_batches.append(discriminator_loss.item())

            discriminator_loss = np.mean(discriminator_loss_batches)
            if is_train:
                writer = self.writer_train
                discriminator_loss_name = "discriminator loss"
            else:
                writer = self.writer_val
                discriminator_loss_name = "validation discriminator loss"
            writer.add_scalar("pretrain/discriminator_loss", discriminator_loss, epoch)
            tqdm_text = (
                f"Epoch: {epoch}, {discriminator_loss_name}: {discriminator_loss:.4f}"
            )
            tqdm.write(tqdm_text)

        self.model.train()
        run(self.train_dataloader, is_train=True)
        self.model.eval()
        with torch.no_grad():
            run(self.validation_dataloader, is_train=False)

    def _step_adv_autoencoder(self, loss: Tensor):
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_decoder.step()
        self.scheduler_encoder_adversarial.step()
        self.scheduler_decoder_adversarial.step()

    def _step_adv_discriminator(self, loss: Tensor):
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()
        self.scheduler_discriminator_adversarial.step()

    def _step_pretrain_discriminator(self, loss: Tensor):
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()
        self.scheduler_discriminator_pretrain.step()

    def _run_adversarial_per_epoch(self, epoch: int):
        def run(dataloader: DataLoader, is_train: bool = True):
            reconstruction_loss_batches = []
            adversarial_loss_batches = []
            total_loss_batches = []
            discriminator_loss_batches = []
            for (
                gene_expressions,
                dosages_one_hot_encoded,
                is_control_soft_labels,
            ) in dataloader:
                gene_expressions = gene_expressions.to(self.device)
                dosages_one_hot_encoded = dosages_one_hot_encoded.to(self.device)
                is_control_soft_labels = is_control_soft_labels.to(self.device)
                is_control_soft_labels = torch.reshape(
                    is_control_soft_labels, (is_control_soft_labels.shape[0], 1)
                )

                reconstruction_loss, latent = self._get_reconstruction_loss(
                    gene_expressions, dosages_one_hot_encoded
                )
                adv_loss = self._get_adversarial_loss(
                    latent, dosages_one_hot_encoded, is_control_soft_labels
                )
                
                if adv_loss is not None:
                    total_loss = self._get_total_loss(reconstruction_loss, adv_loss)
                    adversarial_loss_batches.append(adv_loss.item())
                else:
                    total_loss = reconstruction_loss

                if is_train:
                    self._step_adv_autoencoder(total_loss)

                reconstruction_loss_batches.append(reconstruction_loss.item())
                total_loss_batches.append(total_loss.item())

                discriminator_loss = self._get_discriminator_loss(
                    gene_expressions, is_control_soft_labels
                )

                if is_train:
                    self._step_adv_discriminator(discriminator_loss)

                discriminator_loss_batches.append(discriminator_loss.item())

            reconstruction_loss = np.mean(reconstruction_loss_batches)
            adv_loss = np.mean(adversarial_loss_batches)
            total_loss = np.mean(total_loss_batches)
            discriminator_loss = np.mean(discriminator_loss_batches)

            # self.scheduler_encoder_plateu.step(total_loss)
            # self.scheduler_decoder_plateu.step(reconstruction_loss)
            # self.scheduler_discriminator_plateu.step(discriminator_loss)

            if is_train:
                writer = self.writer_train
                rc_loss_name = "rc loss"
                adv_loss_name = "adv loss"
                total_loss_name = "total loss"
                discriminator_loss_name = "discriminator loss"
            else:
                writer = self.writer_val
                rc_loss_name = "rc val loss"
                adv_loss_name = "adv val loss"
                total_loss_name = "total val loss"
                discriminator_loss_name = "discriminator val loss"

            writer.add_scalar("adv/reconstruction_loss", reconstruction_loss, epoch)
            writer.add_scalar("adv/adv_loss", adv_loss, epoch)
            writer.add_scalar("adv/total_loss", total_loss, epoch)
            writer.add_scalar("adv/discriminator_loss", discriminator_loss, epoch)

            tqdm_text = f"Epoch [{epoch + 1}/{self.adversarial_epochs}], {rc_loss_name}: {reconstruction_loss}"
            tqdm_text += f", {adv_loss_name}: {adv_loss}, {total_loss_name}: {total_loss}, {discriminator_loss_name}: {discriminator_loss}"

            tqdm.write(tqdm_text)

        self.model.train()
        run(self.train_dataloader, is_train=True)
        self.model.eval()
        with torch.no_grad():
            run(self.validation_dataloader, is_train=False)

    def train(self):
        """
        Pretrain autoencoder
        """
        self.autoencoder_trainer.train()

        """
        Pretrain discriminator
        """
        for epoch in tqdm(
            range(self.discriminator_epochs), desc="Pretraining Discriminator"
        ):
            self._run_pretrain_discriminator_per_epoch(epoch)

        """
        Adversarial joint training of autoencoder with discriminator
        """
        for epoch in tqdm(
            range(self.adversarial_epochs), desc="Training Adversarial Epochs"
        ):
            self._run_adversarial_per_epoch(epoch)


class MultiTaskAdversarialSwapAutoencoderTrainer(MultiTaskAdversarialAutoencoderTrainer):
    def _get_adversarial_loss(
        self,
        generator_output: Tensor,
        dosages_one_hot_encoded: Tensor,
        is_control_soft_labels: Tensor,
    ) -> Optional[Tensor]:
        """
        Generator (Encoder) tries to fool both control and perturbed as perturbed and control respectively
        """
        fake_perturb_labels = torch.ones_like(is_control_soft_labels) - is_control_soft_labels
        discriminator_output = self.model.discriminator(generator_output)
        adv_loss = self.bce(input=discriminator_output, target=fake_perturb_labels)
        return adv_loss
    
class MultiTaskAdversarialGaussianAutoencoderTrainer(MultiTaskAdversarialAutoencoderTrainer):
    def _get_adversarial_loss(
        self,
        generator_output: Tensor,
        dosages_one_hot_encoded: Tensor,
        is_control_soft_labels: Tensor,
    ) -> Optional[Tensor]:
        """
        Generator (Encoder) tries to fool discriminator that gene expressions follow gaussian distribution
        """
        discriminator_output = self.model.discriminator(generator_output)
        adv_loss = self.bce(input=discriminator_output, target=torch.ones_like(discriminator_output))
        return adv_loss
    
    def _get_discriminator_loss(
        self, gene_expression: Tensor, is_control_soft_labels: Tensor
    ): 
        latent = self.model.encoder(gene_expression).detach()
        discriminator_output_genes = self.model.discriminator(latent)
        discriminator_loss_genes = self.bce(
            input=discriminator_output_genes, target=torch.zeros_like(discriminator_output_genes)
        )
        
        gaussian_input = torch.normal(
            mean=0.0, std=1.0, size=latent.size(), device=self.device
        )
        discriminator_output_gaussian = self.model.discriminator(gaussian_input)
        discriminator_loss_gaussian = self.bce(
            input=discriminator_output_gaussian, target=torch.ones_like(discriminator_output_gaussian)
        )
        
        discriminator_loss = 0.5 * discriminator_loss_gaussian + 0.5 * discriminator_loss_genes
        return discriminator_loss


class MultiTaskAdversarialAutoencoderUtils:
    def __init__(
        self,
        split_dataset_pipeline: SplitDatasetPipeline,
        target_cell_type: str,
        model: MultiTaskAae,
    ):

        self.model = model
        self.split_dataset_pipeline = split_dataset_pipeline
        self.target_cell_type = target_cell_type
        self.device = "cuda"
        self.model.to(self.device)

    def train(
        self,
        save_path: Path,
        trainer: Trainer,
    ):
        trainer()
        trainer.save(save_path)

    def predict(self):
        control_test_adata = self.split_dataset_pipeline.get_ctrl_test(
            target_cell_type=self.target_cell_type
        )
        stim_test_adata = self.split_dataset_pipeline.get_stim_test(
            target_cell_type=self.target_cell_type
        )
        dosages_to_test = self.split_dataset_pipeline.get_dosages_unique(
            stim_test_adata
        )
        predictions = {}

        gene_expressions = DosagesDataset.get_gene_expressions(control_test_adata)

        stim_test_dataset = DosagesDataset(
            dataset_pipeline=self.split_dataset_pipeline.dataset_pipeline,
            adata=stim_test_adata,
        )

        self.model.eval()
        with torch.no_grad():
            for dosage in dosages_to_test:
                dosages_one_hot_encoded = stim_test_dataset.get_one_hot_encoded_dosages(
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


def run_multi_task_aae(
    *,
    batch_size: int,
    learning_rate: float,
    autoencoder_pretrain_epochs: int,
    discriminator_pretrain_epochs: int,
    adversarial_epochs: int,
    coeff_adversarial: float,
    hidden_layers_autoencoder: list,
    hidden_layers_discriminator: list,
    hidden_layers_film: list,
    seed: int,
    saved_results_path: Path,
    trainer_class: Type[MultiTaskAdversarialAutoencoderTrainer],
    mask_rate: float = 0.5,
    dropout_rate: float = 0.1,
    overwrite: bool = False,
):
    experiment_name = (
        f"layers_ae_{hidden_layers_autoencoder}_disc_{hidden_layers_discriminator}_film_{hidden_layers_film}_"
        f"lr_{learning_rate}_batch_{batch_size}_ae_epochs_{autoencoder_pretrain_epochs}_"
        f"dis_epochs_{discriminator_pretrain_epochs}_adv_epochs_{adversarial_epochs}_"
        f"coef_adv_{coeff_adversarial}_"
        f"mask_rate_{mask_rate}_dropout_{dropout_rate}_"
        f"seed_{seed}_{trainer_class.__name__}"
    )
    print(experiment_name)

    multi_task_aae_path = saved_results_path / "multi_task_aae" / experiment_name

    tensorboard_path = saved_results_path / "runs" / "multi_task_aae" / experiment_name

    model_path = multi_task_aae_path / "model.pt"

    figures_path = multi_task_aae_path / "figures"

    os.makedirs(multi_task_aae_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    is_single = False
    # %%

    if not is_single:
        dataset_pipeline = NaultMultiplePipeline(dataset_pipeline=NaultPipeline())
    else:
        dataset_pipeline = NaultSinglePipeline(
            dataset_pipeline=NaultPipeline(), dosages=30.0
        )

    condition_len = len(dataset_pipeline.get_dosages_unique())
    num_features = dataset_pipeline.get_num_genes()

    film_factory = FilmLayerFactory(
        input_dim=condition_len,
        hidden_layers=hidden_layers_film,
    )

    if model_path.exists() and not overwrite:
        print("Loading model")
        model = MultiTaskAae.load(
            num_features=num_features,
            hidden_layers_autoencoder=hidden_layers_autoencoder,
            hidden_layers_discriminator=hidden_layers_discriminator,
            film_layer_factory=film_factory,
            load_path=multi_task_aae_path,
            mask_rate=mask_rate,
            dropout_rate=dropout_rate,
        )
        model = model.to("cuda")
    else:
        model = MultiTaskAae(
            num_features=num_features,
            hidden_layers_autoencoder=hidden_layers_autoencoder,
            hidden_layers_discriminator=hidden_layers_discriminator,
            film_layer_factory=film_factory,
            mask_rate=mask_rate,
            dropout_rate=dropout_rate
        )

    target_cell_type = "Hepatocytes - portal"

    model_utils = MultiTaskAdversarialAutoencoderUtils(
        split_dataset_pipeline=dataset_pipeline,
        target_cell_type=target_cell_type,
        model=model,
    )

    if model_path.exists() and not overwrite:
        pass
    else:
        trainer = trainer_class(
            model=model,
            tensorboard_path=tensorboard_path,
            split_dataset_pipeline=dataset_pipeline,
            target_cell_type=target_cell_type,
            device="cuda",
            coeff_adversarial=coeff_adversarial,
            autoencoder_pretrain_epochs=autoencoder_pretrain_epochs,
            discriminator_pretrain_epochs=discriminator_pretrain_epochs,
            adversarial_epochs=adversarial_epochs,
            lr=learning_rate,
            batch_size=batch_size,
            seed=seed,
        )

        model_utils.train(trainer=trainer, save_path=multi_task_aae_path)

    predictions = model_utils.predict()

    dfs = []

    stim_test = dataset_pipeline.get_stim_test(target_cell_type=target_cell_type)
    control_test = dataset_pipeline.get_ctrl_test(target_cell_type=target_cell_type)

    dosages_to_test = dataset_pipeline.get_dosages_unique(stim_test)

    for idx, dosage in enumerate(dosages_to_test):
        evaluation_path = multi_task_aae_path / f"dosage{dosage}"

        df, _ = evaluation_out_of_sample(
            control=control_test,
            ground_truth=stim_test[
                stim_test.obs[dataset_pipeline.dosage_key] == dosage
            ],
            predicted=predictions[idx],
            output_path=evaluation_path,
            save_plots=False,
            cell_type_key=dataset_pipeline.cell_type_key,
            skip_distances=True,
        )
        df["dose"] = dosage
        df["experiment"] = experiment_name
        append_csv(df, ROOT / "analysis" / "multi_task_aae.csv")
        dfs.append(df)

        print("Finished evaluation for dosage", dosage)

    all_df = pd.concat(dfs, axis=0)

    overview_df = pd.DataFrame()
    overview_df["experiment"] = [experiment_name]
    overview_df["cell_type_test"] = target_cell_type
    overview_df["DEGs"] = all_df["DEGs"].mean()
    overview_df["r2mean_all_boostrap_mean"] = all_df["r2mean_all_boostrap_mean"].mean()
    overview_df["r2mean_top20_boostrap_mean"] = all_df[
        "r2mean_top20_boostrap_mean"
    ].mean()
    overview_df["r2mean_top100_boostrap_mean"] = all_df[
        "r2mean_top100_boostrap_mean"
    ].mean()
    append_csv(overview_df, ROOT / "analysis" / "multi_task_aae_overview.csv")

    train_adata, validation_adata = dataset_pipeline.split_dataset_to_train_validation(
        target_cell_type=target_cell_type
    )

    def umaps(adata: AnnData, title: str = ""):
        tensor = DosagesDataset.get_gene_expressions(adata).to("cuda")
        latent = AnnData(
            X=model.get_latent_representation(tensor), obs=adata.obs.copy()
        )

        latent.obs["Dose"] = latent.obs["Dose"].astype("category")

        sc.pp.neighbors(latent)
        sc.tl.umap(latent)

        sc.pl.umap(latent, color=["Dose"], show=False)
        plt.savefig(
            f"{figures_path}/multi_task_aae_umap_dose_{experiment_name}_{title}.pdf",
            dpi=150,
            bbox_inches="tight",
        )

        sc.pl.umap(latent, color=["celltype"], show=False)
        plt.savefig(
            f"{figures_path}/multi_task_aae_umap_celltype_{experiment_name}_{title}.pdf",
            dpi=150,
            bbox_inches="tight",
        )

    umaps(adata=train_adata, title="train")
    umaps(adata=validation_adata, title="validation")
    umaps(adata=stim_test, title="stim")

    return (
        overview_df["DEGs"].tolist()[0],
        overview_df["r2mean_all_boostrap_mean"].tolist()[0],
        overview_df["r2mean_top20_boostrap_mean"].tolist()[0],
        overview_df["r2mean_top100_boostrap_mean"].tolist()[0],
    )
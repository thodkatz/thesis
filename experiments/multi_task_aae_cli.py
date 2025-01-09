from thesis import SAVED_RESULTS_PATH
from thesis.multi_task_aae import (
    MultiTaskAdversarialAutoencoderTrainer,
    run_multi_task_aae,
)
import argparse

parser = argparse.ArgumentParser(description="Run multi-task aae")
parser.add_argument("--batch_size", type=int, required=False, help="Batch size")
parser.add_argument("--lr", type=float, required=False, help="Learning rate")
parser.add_argument(
    "--autoencoder_pretrain_epochs",
    type=int,
    required=False,
    help="Autoencoder pretrain epochs",
)
parser.add_argument(
    "--discriminator_pretrain_epochs",
    type=int,
    required=False,
    help="Discriminator pretrain epochs",
)
parser.add_argument(
    "--adversarial_epochs", type=int, required=False, help="Adversarial epochs"
)
parser.add_argument(
    "--coeff_adversarial", type=float, required=False, help="Adversarial coefficient"
)
parser.add_argument(
    "--hidden_layers_ae",
    type=int,
    nargs="+",
    required=False,
    help="Hidden layers as space-separated integers (e.g., 64 128 256)",
)
parser.add_argument(
    "--hidden_layers_disc",
    type=int,
    nargs="+",
    required=False,
    help="Hidden layers as space-separated integers (e.g., 64 128 256)",
)
parser.add_argument(
    "--hidden_layers_film",
    type=int,
    nargs="+",
    required=False,
    help="Hidden layers as space-separated integers (e.g., 64 128 256)",
)
parser.add_argument(
    "--seed", type=int, required=False, help="Random seed for reproducibility"
)
parser.add_argument(
    "--dropout", type=float, required=False, help="Dropout rate"
)
parser.add_argument(
    "--mask", type=float, required=False, help="Mask rate"
)
args = parser.parse_args()


BASELINE_METRICS = [
    "DEGs",
    "r2mean_all_boostrap_mean",
    "r2mean_top20_boostrap_mean",
    "r2mean_top100_boostrap_mean",
]
DISTANCE_METRICS = ["edistance", "wasserstein", "euclidean", "mean_pairwise", "mmd"]

METRICS = BASELINE_METRICS + DISTANCE_METRICS

batch_size = args.batch_size or 256
learning_rate = args.lr or 1e-4
autoencoder_pretrain_epochs = args.autoencoder_pretrain_epochs or 100
hidden_layers_autoencoder = args.hidden_layers_ae or [32, 32]
hidden_layers_film = args.hidden_layers_film or []
mask_rate = args.mask or 0.5
dropout_rate = args.dropout or 0.1

adversarial_epochs = args.adversarial_epochs or 100
if adversarial_epochs == 0:
    discriminator_pretrain_epochs = 0
    coeff_adversarial = 0
    hidden_layers_discriminator = []
else:
    discriminator_pretrain_epochs = args.discriminator_pretrain_epochs or 10
    coeff_adversarial = args.coeff_adversarial or 0.05
    hidden_layers_discriminator = args.hidden_layers_disc or [32, 32]

seed = args.seed or 19193


run_multi_task_aae(
    batch_size=batch_size,
    learning_rate=learning_rate,
    autoencoder_pretrain_epochs=autoencoder_pretrain_epochs,
    discriminator_pretrain_epochs=discriminator_pretrain_epochs,
    adversarial_epochs=adversarial_epochs,
    coeff_adversarial=coeff_adversarial,
    hidden_layers_autoencoder=hidden_layers_autoencoder,
    hidden_layers_discriminator=hidden_layers_discriminator,
    hidden_layers_film=hidden_layers_film,
    seed=seed,
    dropout_rate=dropout_rate,
    mask_rate=mask_rate,
    trainer_class=MultiTaskAdversarialAutoencoderTrainer,
    saved_results_path=SAVED_RESULTS_PATH)

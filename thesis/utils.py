import random
import numpy as np
from pandas import DataFrame
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import scvi  # just importing it, sets the global seed to 0
import torch

SEED = 19193


@dataclass(frozen=True)
class FileModelUtils:
    model_name: str
    dataset_name: str
    experiment_name: str
    perturbation: str
    root_path: Path
    dosages: List[float] = field(default_factory=lambda: [0])
    cell_type_key: str = "celltype"
    dose_key: str = "Dose"

    def is_multi_dose(self) -> bool:
        return len(self.dosages) > 1

    def get_batch_path(self, batch: int, prefix: Optional[Path] = None) -> Path:
        return self.get_perturbation_path(prefix=prefix) / f"batch{batch}"

    def get_perturbation_path(self, prefix: Optional[Path] = None) -> Path:
        prefix = prefix or self.root_path
        if self.is_multi_dose():
            dosage = "dosages"
        else:
            dosage = f"dosage{str(self.dosages[0])}"
        return (
            prefix
            / self.model_name
            / self.experiment_name
            / self.dataset_name
            / self.perturbation
            / dosage
        )

    def get_batch_metrics_path(self, batch: int) -> Path:
        return self.get_batch_path(batch=batch) / "metrics.csv"

    def get_dose_path_multi(self, batch: int, dosage: float) -> Path:
        assert self.is_multi_dose()
        return self.get_batch_path(batch=batch) / f"dose{dosage}"

    def get_dose_path_multi_metrics(self, batch: int, dosage: float) -> Path:
        return self.get_dose_path_multi(batch=batch, dosage=dosage) / "metrics.csv"

    def is_finished_evaluation(self, batch: int, refresh: bool = False) -> bool:
        exists = self.get_batch_metrics_path(batch=batch).exists()
        print("Metrics path exist", exists, "refresh", refresh)
        return exists and not refresh

    def is_finished_evaluation_multi(
        self, batch: int, dosage: float, refresh: bool = False
    ) -> bool:
        exists = self.get_dose_path_multi_metrics(batch=batch, dosage=dosage).exists()
        print("Metrics path exist", exists, "refresh", refresh)
        return exists and not refresh

    def is_finished_batch_training(self, batch: int, refresh: bool = False) -> bool:
        exists = self.get_training_finished_flag_batch_file(batch=batch).exists()
        print("Training finished flag exist", exists, "refresh", refresh)
        return exists and not refresh

    def log_training_batch_is_finished(self, batch: int):
        training_flag_file = self.get_training_finished_flag_batch_file(batch=batch)
        print("Writing training finished flag to", training_flag_file)
        with open(training_flag_file, "w") as f:
            f.write("")

    def get_training_finished_flag_file(self, root: Path) -> Path:
        return root / "training_finished.txt"

    def get_training_finished_flag_batch_file(self, batch: int) -> Path:
        return self.get_training_finished_flag_file(self.get_batch_path(batch=batch))

    def get_log_path(self) -> Path:
        return self.root_path / "runs"

    def get_batch_log_path(self, batch: int) -> Path:
        return self.get_batch_path(prefix=self.get_log_path(), batch=batch)


def append_csv(df: DataFrame, path: Path):
    if not path.exists():
        header_df = DataFrame(columns=df.columns)
        header_df.to_csv(path, index=False)
    print("Writing metrics to", path)
    df.to_csv(path, mode="a", header=False, index=False)


def setup_seed():
    seed = SEED
    print(f"Seed has been set {seed}")
    scvi.settings.seed = seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    """
    For the Dataloaders

    todo: use this for scButterfly dataloading
    todo: inverstiage if scvi handles the randomness of dataloaders
    """
    # worker_seed = (torch.initial_seed() + worker_id) % 2**32
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pretty_print(obj):
    attributes = "\n".join(
        f"  {key}: {value.__class__.__name__ if hasattr(value, '__class__') else value}"
        for key, value in obj.__dict__.items()
    )
    return f"{obj.__class__.__name__}(\n{attributes}\n)"

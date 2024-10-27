import pandas as pd
from pandas import DataFrame
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    dataset_name: str
    experiment_name: str
    perturbation: str
    root_path: Path
    dosage: int = 0
    cell_type_key: str = "celltype"
    
    def get_batch_path(self, batch: int, prefix: Optional[Path] = None) -> Path:
        return self.get_perturbation_path(prefix=prefix) / f"batch{batch}"
    
    def get_perturbation_path(self, prefix: Optional[Path] = None) -> Path:
        prefix = prefix or self.root_path
        return prefix / self.model_name / self.experiment_name / self.dataset_name / self.perturbation / f'dosage{self.dosage}'
    
    def get_batch_metrics_path(self, batch: int) -> Path:
        return self.get_batch_path(batch=batch) / "metrics.csv"
    
    def is_finished_batch(self, batch: int) -> bool:
        return self.get_batch_metrics_path(batch=batch).exists()
    
    def get_log_path(self) -> Path:
        return self.root_path / "runs"
        
    def get_batch_log_path(self, batch: int) -> Path:
        return self.get_batch_path(prefix=self.get_log_path(), batch=batch)
    

def append_csv(df: DataFrame, path: Path):
    if not path.exists():
        header_df = DataFrame(columns=df.columns)
        header_df.to_csv(path, index=False)
    print("Writing metrics to", path)
    df.to_csv(path, mode='a', header=False, index=False)
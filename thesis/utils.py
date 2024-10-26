import pandas as pd
from pandas import DataFrame
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    dataset_name: str
    experiment_name: str
    perturbation: str
    output_path: Path
    dosage: int = 0
    cell_type_key: str = "celltype"
    
    def get_experiment_path(self) -> Path:
        return self.output_path / self.experiment_name
    
    def get_batch_path(self, batch: int) -> Path:
        return self.get_experiment_path() / f"batch{batch}"
    
    def get_batch_metrics_path(self) -> Path:
        return self.get_experiment_path() / "metrics.csv"
    
    def is_finished_batch(self, batch: int) -> bool:
        return self.get_batch_metrics_path().exists()
    

def append_csv(df: DataFrame, path: Path):
    if not path.exists():
        header_df = DataFrame(columns=df.columns)
        header_df.to_csv(path, index=False)
    print("Writing metrics to", path)
    df.to_csv(path, mode='a', header=False, index=False)
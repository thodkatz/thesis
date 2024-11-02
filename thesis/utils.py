from pandas import DataFrame
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class FileModelUtils:
    model_name: str
    dataset_name: str
    experiment_name: str
    perturbation: str
    root_path: Path
    dosages: List[float] = field(default_factory=lambda: [0])
    cell_type_key: str = "celltype",
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
            dosage = f'dosage{str(self.dosages[0])}'
        return prefix / self.model_name / self.experiment_name / self.dataset_name / self.perturbation / dosage
    
    def get_batch_metrics_path(self, batch: int) -> Path:
        return self.get_batch_path(batch=batch) / "metrics.csv"
    
    def is_finished_evaluation(self, batch: int, refresh: bool  = False) -> bool:
        exists = self.get_batch_metrics_path(batch=batch).exists()
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
    df.to_csv(path, mode='a', header=False, index=False)
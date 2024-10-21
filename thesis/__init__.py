from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
SAVED_RESULTS_PATH = ROOT / "saved_results"
METRICS_PATH = SAVED_RESULTS_PATH / "metrics.csv"

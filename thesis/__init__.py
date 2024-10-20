from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
SAVED_RESULTS_PATH = ROOT / "saved_results"
METRICS_PATH = SAVED_RESULTS_PATH / "metrics.csv"

if not METRICS_PATH.exists():
    df = pd.DataFrame(columns=[
        "model",
        "dataset",
        "r2mean",
        "r2mean_top100",
        "cell_type_test",
        "average_mean_diff",
        "average_mean_expressed_diff",
        "average_fractions_diff",
        "average_fractions_degs_diff",
        "average_mean_degs_diff",])
    df.to_csv(METRICS_PATH)
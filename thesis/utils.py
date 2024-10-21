import pandas as pd
from pandas import DataFrame
from pathlib import Path

def append_csv(df: DataFrame, path: Path):
    if not path.exists():
        header_df = DataFrame(columns=df.columns)
        header_df.to_csv(path, index=False)
    print("Writing metrics to", path)
    df.to_csv(path, mode='a', header=False, index=False)
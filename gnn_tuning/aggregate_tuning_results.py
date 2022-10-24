"""Aggregate individual tuning results csv files into a single csv."""
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HOMEDIR = Path(__file__).resolve().parents[1]
HP_TUNE_OUTPUT_DIR = HOMEDIR / "result" / "gene_classification_gnn_hp_tune"
OUTPATH = f"{HP_TUNE_OUTPUT_DIR}.csv"
TARGET_FILE = "score.csv"


def get_result(dir_: str) -> pd.DataFrame:
    """Return dataframe given the directory that contains a target csv file.

    Extract settings and runid from the directory name and add to dataframe.

    """
    terms = dir_.split(os.path.sep)[-2:]
    settings = terms[0]
    runid = terms[1].split("_")[-1]

    df = pd.read_csv(os.path.join(dir_, TARGET_FILE))
    df[["Settings", "RunID"]] = settings, runid

    return df


def main():
    print(f"Start aggregating results from {HP_TUNE_OUTPUT_DIR}")
    pbar = tqdm(list(os.walk(HP_TUNE_OUTPUT_DIR)))
    df = pd.concat(get_result(dir_) for dir_, _, files in pbar if TARGET_FILE in files)

    df.to_csv(OUTPATH, index=False)
    print(f"{df}\n\nResults saved to {OUTPATH}")


if __name__ == "__main__":
    main()

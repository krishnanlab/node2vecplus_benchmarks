import logging
import os
import pathlib
from typing import Union, List

import numpy as np
from NLEval.label import filters
from NLEval.label import LabelsetCollection
from NLEval.label.split import RatioPartition

from util import config_logger
from common_var import DATA_DIR, LABEL_PATH_DICT, PUBMED_COUNT_PATH

config_logger()
logger = logging.getLogger(__name__)

PPI_DIR = DATA_DIR / "networks/ppi"
STRING_NETWORK_PATH = PPI_DIR / "STRING.npz"
HB_NETWORK_DIR = PPI_DIR / "gtexcoexp"
GTX_NETWORK_DIR = PPI_DIR / "humanbase"
HBGTX_NETWORK_PATHS = (
    os.popen(f"ls {HB_NETWORK_DIR}/*.npz").read().split()
    + os.popen(f"ls {GTX_NETWORK_DIR}/*.npz").read().split()
)

DATASETS = ["GOBP", "DisGeNet"]

MIN_NUM_POS = 10  # minimum number of positives per split
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train/val/test split ratios


def filter_lsc(network_paths: Union[str, List[str]], name: str, dataset: str):
    """Filter label set collection using network genes.

    Args:
        network_paths: Path(s) to the network(s).
        name: Name of the network genes.
        dataset: Name of the dataset.

    """
    # Load netowrk genes
    network_paths = network_paths if isinstance(network_paths, list) else [network_paths]
    gene_ids = tuple(set.intersection(*[set(np.load(i)["IDs"].tolist()) for i in network_paths]))
    logger.info(f"Loaded {len(gene_ids):,} number of genes")

    # Apply basic filtering (gene intersection & labelset size)
    lsc = LabelsetCollection.from_gmt(LABEL_PATH_DICT[dataset])
    lsc.iapply(filters.EntityExistenceFilter(gene_ids), progress_bar=True)
    lsc.iapply(filters.LabelsetRangeFilterSize(min_val=50), progress_bar=True)
    lsc.load_entity_properties(PUBMED_COUNT_PATH, "PubMed Count", 0, int)

    # Apply (study-bias holdout) split size filtering
    splitter = RatioPartition(*SPLIT_RATIOS, ascending=False)
    split_filter = filters.LabelsetRangeFilterSplit(MIN_NUM_POS, splitter, property_name="PubMed Count")
    lsc.iapply(split_filter, progress_bar=True)

    # Generate label and masks
    y, masks = lsc.split(splitter, target_ids=gene_ids, property_name="PubMed Count")
    logger.info(f"{y.sum(0)=}")
    logger.info(f"{y[masks['test'][:, 0]].sum(0)=}")
    logger.info(f"{y.shape=}")

    # Save labelset npz
    np.savez(
        DATA_DIR / f"labels/gene_classification/{name}_{dataset}_label_split.npz",
        y=y,
        train_idx=np.where(masks["train"][:, 0])[0],
        valid_idx=np.where(masks["val"][:, 0])[0],
        test_idx=np.where(masks["test"][:, 0])[0],
        label_ids=lsc.label_ids,
        gene_ids=gene_ids,
    )


def main():
    for dataset in DATASETS:
        filter_lsc(STRING_NETWORK_PATH, "STRING", dataset)
        filter_lsc(HBGTX_NETWORK_PATHS, "HBGTX", dataset)


if __name__ == "__main__":
    main()

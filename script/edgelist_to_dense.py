"""Convert edge list files to dense numpy array for faster loading."""
import argparse
import logging

import numpy as np
from NLEval.graph import DenseGraph

from util import config_logger

config_logger()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert edgelist to dense arrays")
    parser.add_argument("-i", "--input", help="Full path to the input file", required=True)
    parser.add_argument("-o", "--output", help="Full path to the output file", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(args)

    logger.info(f"Start loading edge list from {args.input}")
    g = DenseGraph.from_edgelist(args.input, weighted=True, directed=False)

    logger.info(f"Saving dense array to {args.output}")
    g.save_npz(args.output, key_map={"adj": "data", "node_ids": "IDs"})

if __name__ == "__main__":
    main()

"""Convert edge list files to dense numpy array for faster loading."""
import argparse
import logging
from typing import Dict, List, Tuple

import mygene
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


def get_entrez_conversion(genes: Tuple[str, ...]) -> Tuple[Dict[str, str], List[str]]:
    # Query mapping from gene IDs to entrez IDs
    mg = mygene.MyGeneInfo()
    df = mg.getgenes(genes, fields="entrezgene", as_dataframe=1, species="human")
    df = df[~df.entrezgene.isna()]
    id_to_entrez = dict(zip(df.index, df.entrezgene))
    entrez_to_id = dict(zip(df.entrezgene, df.index))
    logger.info(f"Converting gene IDs to entrez, {df.shape[0]:,} (out of {len(genes):,}) found.")

    # Exclude gene if multiple IDs are mapped to the same entrez ID
    entrez_genes = (df.entrezgene.value_counts() == 1).index.tolist()
    gene_ids = list(map(entrez_to_id.get, entrez_genes))
    logger.info(f"{len(gene_ids):,} (out of {df.shape[0]:,}) mapped, excluding low quality map.")

    return id_to_entrez, gene_ids


def process_graph(g: DenseGraph) -> DenseGraph:
    """Obtain the largest component induced by mapped entrez genes."""
    id_to_entrez, gene_ids = get_entrez_conversion(g.node_ids)

    # Take largest component in the induced subgraph and reassign entrez
    g.log_level = "INFO"
    g = g.induced_subgraph(gene_ids).largest_connected_subgraph()
    g = DenseGraph.from_mat(g.mat, list(map(id_to_entrez.get, g.node_ids)))

    return g


def main():
    args = parse_args()
    logger.info(args)

    # Load graph from edge list (exclude negative edges)
    logger.info(f"Start loading edge list from {args.input}")
    g = DenseGraph.from_edgelist(args.input, weighted=True, directed=False, cut_threshold=0)

    # Convert genes to entrez and take largest component
    g = process_graph(g)

    g.save_npz(args.output, key_map={"adj": "data", "node_ids": "IDs"})
    logger.info(f"Dense array saved to {args.output}")


if __name__ == "__main__":
    main()

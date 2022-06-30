import logging
import os
import os.path as osp
import pathlib

import numpy as np
import numba
import yaml
from gensim.models import Word2Vec
from pecanpy import pecanpy
from sklearn.metrics import average_precision_score

from common_var import *


def config_logger():
    """Configure logger using the config file."""
    homedir = pathlib.Path(__file__).absolute().parent
    with open(osp.join(homedir, "logging.yaml"), "r") as f:
        logging.config.dictConfig(yaml.safe_load(f.read()))


def check_dirs(dirs):
    """Check directory and create if not exist"""
    for directory in dirs:
        try: 
            os.makedirs(directory)
        except FileExistsError:
            pass


def get_network_fp(network: str):
    """Get the path fo the network file under data/networks/ppi"""
    filename = f"{network}.npz"
    for path, _, files in os.walk(NETWORK_DIR):
        if filename in files:
            filepath = os.path.join(path, filename)
            print(f"Found network at {filepath}")
            return filepath
    else:
        raise FileNotFoundError(f"Cannot locate {filename}")


def score_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    prior = y_true.sum() / y_true.size
    auprc = average_precision_score(y_true, y_pred)
    return np.log2(auprc / prior)


def align_gene_ids(adj_ids, y, train_idx, valid_idx, test_idx, gene_ids):
    """Align label split IDs using network node IDs"""
    # Train/val/test split stats before alignment
    old_stats = [y[idx].sum(0).tolist() for idx in [train_idx, valid_idx, test_idx]]

    # Pad the label matrix if necessary (not all network genes are in the label
    # matrix, but the converse must be true though)
    missing_genes = set(adj_ids.tolist()) - set(gene_ids.tolist())
    y = np.vstack((y, np.zeros((len(missing_genes), y.shape[1]))))
    gene_ids = np.array(gene_ids.tolist() + list(missing_genes))

    # Map from id to index in the label split
    id_map = {j:i for i,j in enumerate(gene_ids)}

    # Index aligning label split ids to network node ids
    aligned_idx = np.array([id_map[i] for i in adj_ids])

    # Align label genes with network genes
    y[:] = y[aligned_idx]
    gene_ids[:] = gene_ids[aligned_idx]

    # Align indices
    aligned_idx_reverse = np.empty(aligned_idx.size, dtype=int)
    aligned_idx_reverse[aligned_idx] = np.arange(aligned_idx.size)
    train_idx[:] = aligned_idx_reverse[train_idx]
    valid_idx[:] = aligned_idx_reverse[valid_idx]
    test_idx[:] = aligned_idx_reverse[test_idx]

    # Train/val/test split stats after alignment
    new_stats = [y[idx].sum(0).tolist() for idx in [train_idx, valid_idx, test_idx]]

    # Check to see if the alignment is correct
    assert adj_ids.tolist() == gene_ids.tolist()
    assert old_stats == new_stats

    return y, gene_ids



def embed(network_fp, dim, extend, p, q, workers, gamma):
    # initialize DenseOTF graph
    adj_mat, IDs = np.load(network_fp).values()
    g = pecanpy.DenseOTF.from_mat(adj_mat, IDs, p=p, q=q, workers=workers,
                                  extend=extend, gamma=gamma)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=W2V_NUMWALKS, walk_length=W2V_WALKLENGTH)
    w2v = Word2Vec(walks, vector_size=dim, window=W2V_WINDOW,
                   min_count=0, sg=1, workers=workers, epochs=W2V_EPOCHS)

    # sort embeddings by IDs
    IDmap = {j:i for i,j in enumerate(w2v.wv.index_to_key)}
    idx_ary = [IDmap[i] for i in IDs]
    X_emd = w2v.wv.vectors[idx_ary]

    return X_emd, IDs

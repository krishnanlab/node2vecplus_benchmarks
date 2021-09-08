import os

import numpy as np
import numba
from sklearn.metrics import average_precision_score

from pecanpy import node2vec
from gensim.models import Word2Vec

from common_var import *


try:
    numba.set_num_threads(NUM_THREADS)
except ValueError:
    pass


def check_dirs(dirs):
    """Check directory and create if not exist"""
    for directory in dirs:
        try: 
            os.makedirs(directory)
        except FileExistsError:
            pass


def score_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    prior = y_true.sum() / y_true.size
    auprc = average_precision_score(y_true, y_pred)
    return np.log2(auprc / prior)


def align_gene_ids(adj_ids, y, train_idx, valid_idx, test_idx, gene_ids):
    """Align label split IDs using network node IDs"""
    # map from id to index in the label split
    id_map = {j:i for i,j in enumerate(gene_ids)}

    # index aligning label split ids to network node ids
    aligned_idx = np.array([id_map[i] for i in adj_ids])

    # apply alignment
    y[:] = y[aligned_idx]
    gene_ids[:] = gene_ids[aligned_idx]
    train_idx[:] = aligned_idx[train_idx]
    valid_idx[:] = aligned_idx[valid_idx]
    test_idx[:] = aligned_idx[test_idx]


def embed(network_fp, dim, extend, p, q):
    # initialize DenseOTF graph
    adj_mat, IDs = np.load(network_fp).values()
    g = node2vec.DenseOTF(p=p, q=q, workers=NUM_THREADS, verbose=False, extend=extend)
    g.from_mat(adj_mat, IDs)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=W2V_NUMWALKS, walk_length=W2V_WALKLENGTH)
    w2v = Word2Vec(walks, vector_size=dim, window=W2V_WINDOW,
                   min_count=0, sg=1, workers=NUM_THREADS, epochs=W2V_EPOCHS)

    # sort embeddings by IDs
    IDmap = {j:i for i,j in enumerate(w2v.wv.index_to_key)}
    idx_ary = [IDmap[i] for i in IDs]
    X_emd = w2v.wv.vectors[idx_ary]

    return X_emd, IDs


def embed_sparse(network_fp, dim, extend, p, q, weighted=True):
    # initialize SparseOTF graph
    g = node2vec.SparseOTF(p=p, q=q, workers=NUM_THREADS, verbose=False, extend=extend)
    g.read_edg(network_fp, weighted, directed=False)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=W2V_NUMWALKS, walk_length=W2V_WALKLENGTH)
    w2v = Word2Vec(walks, vector_size=dim, window=W2V_WINDOW,
                   min_count=0, sg=1, workers=NUM_THREADS, epochs=W2V_EPOCHS)

    # return embeddings with node IDs
    X_emd = w2v.wv.vectors
    IDs = np.array(w2v.wv.index_to_key)

    return X_emd, IDs


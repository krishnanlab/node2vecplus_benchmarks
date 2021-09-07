import numpy as np
from sklearn.metrics import average_precision_score


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


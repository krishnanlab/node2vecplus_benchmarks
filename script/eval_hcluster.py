import argparse

import numpy as np
import numba
numba.set_num_threads(1)

from pecanpy import node2vec
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score


DATA_DIR = "../data"
NETWORK_DIR = f"{DATA_DIR}/networks/synthetic"
LABEL_DIR = f"{DATA_DIR}/labels/hierarchical_cluster"

###DEFAULT HYPERPARAMS###
p = 1
dim = 16
num_walks = 10
walk_length = 80
window_size = 10
epochs = 1
#########################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on hierarchical cluster datasets")

    parser.add_argument("--network", required=True,
        help="Name of hierarchical cluster graph to use")

    parser.add_argument("--q", required=True, type=float,
        help="in-out bias parameter q")

    parser.add_argument("--extend", action="store_true",
        help="Use node2vec+ if specified, otherwise use node2vec")

    parser.add_argument("--random_state", type=int, default=0,
        help="Random state used for generating random splits")

    args = parser.parse_args()
    print(args)

    return args


def embed(args):
    network = args.network
    extend = args.extend
    q = args.q

    network_fp = f"{NETWORK_DIR}/{network}.npz"

    # initialize DenseOTF graph
    adj_mat, IDs = np.load(network_fp).values()
    g = node2vec.DenseOTF(p=p, q=q, workers=1, verbose=False, extend=extend)
    g.from_mat(adj_mat, IDs)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)
    w2v = Word2Vec(walks, vector_size=dim, window=window_size, min_count=0,
                   sg=1, workers=1, epochs=epochs)

    # sort embeddings by IDs
    idx_ary = np.array(w2v.wv.index_to_key, dtype=int).argsort()
    X_emd = w2v.wv.vectors[idx_ary]

    return X_emd


def evaluate(args, X_emd):
    task = 'cluster'
    network = args.network
    random_state = args.random_state

    label_fp = f"{LABEL_DIR}/{network}_{task}_labels.txt"

    y_combined = np.loadtxt(label_fp, dtype=int)
    y = np.zeros((y_combined.size, y_combined.max() + 1), dtype=bool)
    for i, class_idx in enumerate(y_combined):
        y[i, class_idx] = True

    mdl = LogisticRegression(penalty='l2', multi_class='multinomial')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=random_state)

    train_idx, test_idx = next(sss.split(X_emd, y))
    mdl.fit(X_emd[train_idx], y_combined[train_idx])

    train_score = f1_score(y_combined[train_idx], mdl.predict(X_emd[train_idx]), average='macro')
    test_score = f1_score(y_combined[test_idx], mdl.predict(X_emd[test_idx]), average='macro')

    print(f"Training score = {train_score:.2f}, Testing score = {test_score:.2f}")


def main():
    args = parse_args()
    X_emd = embed(args)
    evaluate(args, X_emd)


if __name__ == '__main__':
    main()


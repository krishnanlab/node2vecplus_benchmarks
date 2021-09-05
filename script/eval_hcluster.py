import os
import argparse

import numpy as np
import pandas as pd
import numba
numba.set_num_threads(1)

from pecanpy import node2vec
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score


DATA_DIR = "../data"
RESULT_DIR = "../result"
OUTPUT_DIR = f"{RESULT_DIR}/hierarchical_cluster"
NETWORK_DIR = f"{DATA_DIR}/networks/synthetic"
LABEL_DIR = f"{DATA_DIR}/labels/hierarchical_cluster"

# check if output directory exist and create it not
if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TASK_LIST = ['cluster', 'level']
REPETITION = 10

###DEFAULT HYPER PARAMS###
HPARAM_P = 1
HPARAM_DIM = 16
HPARAM_NUMWALKS = 10
HPARAM_WALKLENGTH = 80
HPARAM_WINDOW = 10
HPARAM_EPOCHS = 1
##########################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on hierarchical cluster datasets")

    parser.add_argument("--network", required=True,
        help="Name of hierarchical cluster graph to use")

    parser.add_argument("--q", required=True, type=float,
        help="in-out bias parameter q")

    parser.add_argument("--extend", action="store_true",
        help="Use node2vec+ if specified, otherwise use node2vec")

    parser.add_argument("--nooutput", action='store_true',
        help="Disable output if specified, and print results to screen")

    parser.add_argument("--random_state", type=int, default=0,
        help="Random state used for generating random splits")

    args = parser.parse_args()
    print(args)

    return args


def _embed(network_fp, extend, q):
    # initialize DenseOTF graph
    adj_mat, IDs = np.load(network_fp).values()
    g = node2vec.DenseOTF(p=HPARAM_P, q=q, workers=1, verbose=False, extend=extend)
    g.from_mat(adj_mat, IDs)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=HPARAM_NUMWALKS, walk_length=HPARAM_WALKLENGTH)
    w2v = Word2Vec(walks, vector_size=HPARAM_DIM, window=HPARAM_WINDOW,
                   min_count=0, sg=1, workers=1, epochs=HPARAM_EPOCHS)

    # sort embeddings by IDs
    idx_ary = np.array(w2v.wv.index_to_key, dtype=int).argsort()
    X_emd = w2v.wv.vectors[idx_ary]

    return X_emd


def _evaluate(X_emd, label_fp, random_state):
    # load labels and convert to one-hot encoded representation for stratified split
    y_combined = np.loadtxt(label_fp, dtype=int)
    y = np.zeros((y_combined.size, y_combined.max() + 1), dtype=bool)
    for i, class_idx in enumerate(y_combined):
        y[i, class_idx] = True

    # initialize classifiaction model and split generator
    mdl = LogisticRegression(penalty='l2', multi_class='multinomial')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=random_state)

    # generate split and train model
    train_idx, test_idx = next(sss.split(X_emd, y))
    mdl.fit(X_emd[train_idx], y_combined[train_idx])

    # evaluate performance
    train_score = f1_score(y_combined[train_idx], mdl.predict(X_emd[train_idx]), average='macro')
    test_score = f1_score(y_combined[test_idx], mdl.predict(X_emd[test_idx]), average='macro')

    return train_score, test_score


def evaluate(args):
    network = args.network
    network_name = network.split('s')[0].split('c')[0]
    extend = args.extend
    q = args.q
    random_state = args.random_state
    nooutput = args.nooutput

    network_fp = f"{NETWORK_DIR}/{network}.npz"
    output_fp = f"{OUTPUT_DIR}/{network}_n2v{'plus' if extend else ''}_q={q}.csv"

    # run evaluation with repetitions on both tasks
    result_df_list = []
    for _ in range(REPETITION):
        X_emd = _embed(network_fp, extend, q)

        train_score_list, test_score_list = [], []
        for task in TASK_LIST:
            label_fp = f"{LABEL_DIR}/{network_name}_{task}_labels.txt"
            train_score, test_score = _evaluate(X_emd, label_fp, random_state)

            train_score_list.append(train_score)
            test_score_list.append(test_score)

        result_df_list.append(pd.DataFrame())
        result_df_list[-1]['Training score'] = train_score_list        
        result_df_list[-1]['Testing score'] = test_score_list        
        result_df_list[-1]['Task'] = TASK_LIST

    # save or print results
    result_df = pd.concat(result_df_list).sort_values('Task')
    if nooutput:
        print(result_df)
    else:
        result_df.to_csv(output_fp, index=False)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()


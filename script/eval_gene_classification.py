import os
import argparse
from time import time

import numpy as np
import pandas as pd
import numba
NUM_THREADS = 4
numba.set_num_threads(NUM_THREADS)

from pecanpy import node2vec
from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


DATA_DIR = "../data"
RESULT_DIR = "../result"
OUTPUT_DIR = f"{RESULT_DIR}/gene_classification"
NETWORK_DIR = f"{DATA_DIR}/networks/ppi"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

# check if output directory exist and create it not
if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DATASET_LIST = ['GOBP', 'KEGGBP', 'DisGeNet']

###DEFAULT HYPER PARAMS###
HPARAM_DIM = 128
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

    parser.add_argument("--p", required=True, type=float,
        help="return bias parameter p")

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


def _embed(network_fp, extend, p, q):
    # initialize DenseOTF graph
    adj_mat, IDs = np.load(network_fp).values()
    g = node2vec.DenseOTF(p=p, q=q, workers=NUM_THREADS, verbose=False, extend=extend)
    g.from_mat(adj_mat, IDs)

    # simulate random walks and genearte embedings
    walks = g.simulate_walks(num_walks=HPARAM_NUMWALKS, walk_length=HPARAM_WALKLENGTH)
    w2v = Word2Vec(walks, vector_size=HPARAM_DIM, window=HPARAM_WINDOW,
                   min_count=0, sg=1, workers=NUM_THREADS, epochs=HPARAM_EPOCHS)

    # sort embeddings by IDs
    IDmap = {j:i for i,j in enumerate(w2v.wv.index_to_key)}
    idx_ary = [IDmap[i] for i in IDs]
    X_emd = w2v.wv.vectors[idx_ary]

    return X_emd


def _score_func(y_true, y_pred):
    prior = y_true.sum() / y_true.size
    auprc = average_precision_score(y_true, y_pred)
    return np.log2(auprc / prior)


def _evaluate(X_emd, label_fp, random_state):
    # load labels and study-bias holdout splits
    y, train_idx, valid_idx, test_idx, label_ids, _ = np.load(label_fp).values()
    n_tasks = label_ids.size

    # initialize classifiaction model and split generator

    # train and evaluate predictions for each task
    train_score_list, valid_score_list, test_score_list = [], [], []
    for task_idx in range(n_tasks):
        mdl = LogisticRegression(penalty='l2', solver='liblinear', max_iter=500)
        mdl.fit(X_emd[train_idx], y[train_idx, task_idx])

        train_score_list.append(_score_func(y[train_idx, task_idx], mdl.decision_function(X_emd[train_idx])))
        valid_score_list.append(_score_func(y[valid_idx, task_idx], mdl.decision_function(X_emd[valid_idx])))
        test_score_list.append(_score_func(y[test_idx, task_idx], mdl.decision_function(X_emd[test_idx])))

    df = pd.DataFrame()
    df['Training score'] = train_score_list
    df['Validation score'] = valid_score_list
    df['Testing score'] = test_score_list
    df['Task'] = list(label_ids)

    return df


def evaluate(args):
    network = args.network
    extend = args.extend
    p = args.p
    q = args.q
    random_state = args.random_state
    nooutput = args.nooutput

    pq = f"p={p}_q={q}"
    method = 'Node2vec+' if extend else 'Node2vec'
    network_fp = f"{NETWORK_DIR}/{network}.npz"
    output_fp = f"{OUTPUT_DIR}/{network}_n2v{'plus' if extend else ''}_q={q}.csv"

    # generate embeddings and report time usage
    t = time()
    X_emd = _embed(network_fp, extend, p, q)
    t = time() - t
    print(f"Took {int(t/3600):02d}:{int(t/60):02d}:{t%60:05.02f} to generate embeddings using {method}")

    # run evaluation with repetitions for all datasets and report time usage
    t = time()
    result_df_list = []
    for dataset in DATASET_LIST:
        label_fp = f"{LABEL_DIR}/{network}_{dataset}_label_split.npz"

        df = _evaluate(X_emd, label_fp, random_state)
        df['Dataset'], df['Network'], df['Method'] = dataset, network, method
        df['p'], df['q'], df['pq'] = p, q, pq
        result_df_list.append(df)
    t = time() - t
    print(f"Took {int(t/3600):02d}:{int(t/60):02d}:{t%60:05.02f} to evaluate")
        
    # combine results into a single dataframe
    result_df = pd.concat(result_df_list).sort_values('Task')

    # save or print results
    if nooutput:
        print(result_df)
    else:
        result_df.to_csv(output_fp, index=False)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()


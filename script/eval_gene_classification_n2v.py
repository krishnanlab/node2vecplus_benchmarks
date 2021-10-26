import os
import argparse
from time import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from util import *


OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2v"
CV_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2v_cv"
NETWORK_DIR = f"{DATA_DIR}/networks/ppi"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

check_dirs([RESULT_DIR, OUTPUT_DIR, CV_OUTPUT_DIR])

DATASET_LIST = ['GOBP', 'KEGGBP', 'DisGeNet']

###DEFAULT HYPER PARAMS###
HPARAM_DIM = 128
##########################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on hierarchical cluster datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

    parser.add_argument('--test', action='store_true',
        help="Toggle test mode, run with more workers")

    args = parser.parse_args()
    print(args)

    return args


def _evaluate(X_emd, IDs, label_fp, random_state, df_info):
    # load labels and study-bias holdout splits
    y, train_idx, valid_idx, test_idx, label_ids, gene_ids = np.load(label_fp).values()
    align_gene_ids(IDs, y, train_idx, valid_idx, test_idx, gene_ids)  # align node ids
    train_valid_test_idx = train_idx, valid_idx, test_idx
    n_tasks = label_ids.size

    # train and evaluate predictions for each task
    score_lists = [], [], []
    for task_idx in range(n_tasks):
        mdl = LogisticRegression(penalty='l2', solver='liblinear', max_iter=500)
        mdl.fit(X_emd[train_idx], y[train_idx, task_idx])

        for score_list, idx in zip(score_lists, train_valid_test_idx):
            score_list.append(score_func(y[idx, task_idx], mdl.decision_function(X_emd[idx])))

    df = pd.DataFrame()
    df['Training score'], df['Validation score'], df['Testing score'] = score_lists
    df['Task'] = list(label_ids)
    for name, val in df_info.items():
        df[name] = val

    return df


def _evaluate_cv(X_emd, IDs, label_fp, random_state, df_info):
    # load labels and study-bias holdout splits
    y, train_idx, valid_idx, test_idx, label_ids, gene_ids = np.load(label_fp).values()
    align_gene_ids(IDs, y, train_idx, valid_idx, test_idx, gene_ids)  # align node ids
    n_tasks = label_ids.size

    eval_idx = np.concatenate([train_idx, valid_idx, test_idx])
    y_eval = y[eval_idx]
    X_emd_eval = X_emd[eval_idx]

    # initialize splitter and model
    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    mdl = LogisticRegression(penalty='l2', solver='liblinear', max_iter=500)

    # train and evaluate predictions for each task
    score_lists = [], [], []
    for task_idx in range(n_tasks):
        task_score_lists = [], [], []
        for train_valid_idx, test_idx in skf1.split(y_eval, y_eval[:, task_idx]):
            y_train_valid = y_eval[train_valid_idx, task_idx]

            for train_idx, valid_idx in skf2.split(y_train_valid, y_train_valid):
                train_idx = train_valid_idx[train_idx]
                valid_idx = train_valid_idx[valid_idx]
                train_valid_test_idx = train_idx, valid_idx, test_idx

                mdl.fit(X_emd_eval[train_idx], y_eval[train_idx, task_idx])
                for task_score_list, idx in zip(task_score_lists, train_valid_test_idx):
                    task_score_list.append(score_func(y_eval[idx, task_idx],
                                           mdl.decision_function(X_emd_eval[idx])))

        for score_list, task_score_list in zip(score_lists, task_score_lists):
            score_list.append(np.mean(task_score_list))

    df = pd.DataFrame()
    df['Training score'], df['Validation score'], df['Testing score'] = score_lists
    df['Task'] = list(label_ids)
    for name, val in df_info.items():
        df[name] = val

    return df


def evaluate(args):
    network = args.network
    extend = args.extend
    p = args.p
    q = args.q
    random_state = args.random_state
    nooutput = args.nooutput

    if args.test:
        NUM_THREADS = 128
    else:
        NUM_THREADS = 4

    try:
        numba.set_num_threads(NUM_THREADS)
    except ValueError:
        pass

    pq = f"p={p}_q={q}"
    method = 'Node2vec+' if extend else 'Node2vec'
    network_fp = f"{NETWORK_DIR}/{network}.npz"
    output_fn = f"{network}_n2v{'plus' if extend else ''}_p={p}_q={q}.csv"

    # generate embeddings and report time usage
    t = time()
    X_emd, IDs = embed(network_fp, HPARAM_DIM, extend, p, q, NUM_THREADS)
    t = time() - t
    print(f"Took {int(t/3600):02d}:{int(t/60):02d}:{t%60:05.02f} to generate embeddings using {method}")

    # evaluate based on both study-bias holdout and 5-fold cross validation
    eval_funcs = [_evaluate, _evaluate_cv]
    outdirs = [OUTPUT_DIR, CV_OUTPUT_DIR]
    for eval_func, outdir in zip(eval_funcs, outdirs):
        # run evaluation with repetitions for all datasets and report time usage
        t = time()
        result_df_list = []
        for dataset in DATASET_LIST:
            label_fp = f"{LABEL_DIR}/{network}_{dataset}_label_split.npz"

            df_info = {'Dataset': dataset, 'Network': network,
                       'Method': method, 'p': p, 'q': q, 'pq': p}
            df = eval_func(X_emd, IDs, label_fp, random_state, df_info)
            result_df_list.append(df)
        t = time() - t
        print(f"Took {int(t/3600):02d}:{int(t/60):02d}:{t%60:05.02f} to evaluate")

        # combine results into a single dataframe
        result_df = pd.concat(result_df_list).sort_values('Task')

        # save or print results
        if nooutput:
            print(result_df[['Training score', 'Validation score', 'Testing score']].describe())
        else:
            print(result_df[['Training score', 'Validation score', 'Testing score']].describe())
            output_fp = f"{outdir}/{output_fn}"
            result_df.to_csv(output_fp, index=False)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()


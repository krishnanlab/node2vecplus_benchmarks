import os
import argparse
from time import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from util import *


N2V_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2v"
N2VPLUS_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_n2vplus"
N2V_TISSUE_OUTPUT_DIR = f"{RESULT_DIR}/tissue_gene_classification_n2v"
N2VPLUS_TISSUE_OUTPUT_DIR = f"{RESULT_DIR}/tissue_gene_classification_n2vplus"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

check_dirs([RESULT_DIR, N2VPLUS_OUTPUT_DIR, N2V_OUTPUT_DIR, N2V_TISSUE_OUTPUT_DIR, N2VPLUS_TISSUE_OUTPUT_DIR])

###DEFAULT HYPER PARAMS###
HPARAM_DIM = 128
##########################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on gene classification datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gene_universe", required=True, help="Name of the gene universe")
    parser.add_argument("--network", required=True, help="Name of hierarchical cluster graph to use")
    parser.add_argument("--task", default="standard", help="'standard': GOBP, DisGeNet or 'tissue': GOBP-tissue")
    parser.add_argument("--p", required=True, type=float, help="return bias parameter p")
    parser.add_argument("--q", required=True, type=float, help="in-out bias parameter q")
    parser.add_argument("--extend", action="store_true", help="Use node2vec+ if specified, otherwise use node2vec")
    parser.add_argument("--gamma", type=float, default=0, help="Noisy edge threshold parameter.")
    parser.add_argument("--nooutput", action="store_true", help="Disable output if specified, and print results to screen")
    parser.add_argument("--random_state", type=int, default=0, help="Random state used for generating random splits")
    parser.add_argument("--test", action="store_true", help="Toggle test mode, run with more workers")

    args = parser.parse_args()
    print(args)

    return args


def _evaluate(X_emd, IDs, label_fp, random_state, df_info):
    # load labels and study-bias holdout splits
    y, train_idx, valid_idx, test_idx, label_ids, gene_ids = np.load(label_fp).values()
    y, gene_ids = align_gene_ids(IDs, y, train_idx, valid_idx, test_idx, gene_ids)
    train_valid_test_idx = train_idx, valid_idx, test_idx
    n_tasks = label_ids.size

    # train and evaluate predictions for each task
    score_lists = [], [], []
    for task_idx in range(n_tasks):
        mdl = LogisticRegression(penalty="l2", solver="liblinear", max_iter=500)
        mdl.fit(X_emd[train_idx], y[train_idx, task_idx])

        for score_list, idx in zip(score_lists, train_valid_test_idx):
            score_list.append(score_func(y[idx, task_idx], mdl.decision_function(X_emd[idx])))

    df = pd.DataFrame()
    df["Training score"], df["Validation score"], df["Testing score"] = score_lists
    df["Task"] = list(label_ids)
    for name, val in df_info.items():
        df[name] = val

    return df


def evaluate(args):
    gene_universe = args.gene_universe
    network = args.network
    extend = args.extend
    p = args.p
    q = args.q
    gamma = args.gamma
    random_state = args.random_state
    nooutput = args.nooutput

    if args.task == "standard":
        datasets = ["GOBP", "DisGeNet"]
    elif args.task == "tissue":
        datasets = ["GOBP-tissue"]
    else:
        raise ValueError(f"Unknown task {task}")

    if args.test:
        NUM_THREADS = 128
    else:
        NUM_THREADS = 4

    try:
        numba.set_num_threads(NUM_THREADS)
    except ValueError:
        pass

    method = "Node2vec+" if extend else "Node2vec"
    output_fn = f"{network}_n2v{'plus' if extend else ''}_{p=}_{q=}_{gamma=}.csv"
    network_fp = get_network_fp(network)

    # Generate embeddings
    t = time()
    X_emd, IDs = embed(network_fp, HPARAM_DIM, extend, p, q, NUM_THREADS, gamma)
    t = time() - t
    print(f"Took {int(t/3600):02d}:{int(t/60):02d}:{t%60:05.02f} to generate embeddings using {method}")

    # Run evaluation on all datasets
    t = time()
    result_df_list = []
    for dataset in datasets:
        label_fp = f"{LABEL_DIR}/{gene_universe}_{dataset}_label_split.npz"

        df_info = {"Dataset": dataset, "Network": network, "Method": method,
                   "p": p, "q": q, "gamma": gamma}
        df = _evaluate(X_emd, IDs, label_fp, random_state, df_info)
        result_df_list.append(df)
    t = time() - t
    print(f"Took {int(t/3600):02d}:{int(t/60):02d}:{t%60:05.02f} to evaluate")

    # combine results into a single dataframe
    result_df = pd.concat(result_df_list).sort_values("Task")

    # Print results summary (and save)
    print(result_df[["Training score", "Validation score", "Testing score"]].describe())
    if not nooutput:
        if args.task == "standard":
            output_dir = N2VPLUS_OUTPUT_DIR if extend else N2V_OUTPUT_DIR
        else:
            output_dir = N2VPLUS_TISSUE_OUTPUT_DIR if extend else N2V_TISSUE_OUTPUT_DIR

        output_fp = f"{output_dir}/{output_fn}"
        result_df.to_csv(output_fp, index=False)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

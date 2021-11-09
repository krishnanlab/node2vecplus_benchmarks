import argparse
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from util import *


OUTPUT_DIR = f"{RESULT_DIR}/product_lite"
TEST_EMD_FP = f"{DATA_DIR}/networks/product_lite/ProductLite_n2v.emd.npz"
NETWORK_FP = f"{DATA_DIR}/networks/product_lite/ProductLite.csr.npz"
LABEL_FP = f"{DATA_DIR}/labels/ProductLite.tsv"

check_dirs([RESULT_DIR, OUTPUT_DIR])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on Amazon co-review graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--q", type=float, default=1,
        help="in-out bias parameter q")

    parser.add_argument("--extend", action="store_true",
        help="Use node2vec+ if specified, otherwise use node2vec")

    parser.add_argument("--run_idx", type=int, default=0,
        help="Run index for differentiating different runs")

    parser.add_argument("--nooutput", action="store_true",
        help="Disable output if specified, and print results to screen")

    parser.add_argument("--workers", type=int, default=28,
        help="Number of workers for embedding")

    parser.add_argument("--random_state", type=int, default=0,
        help="Random state used for generating random splits")

    parser.add_argument("--test", action="store_true",
        help="Toggle test mode, use precomputed embeddings with subdataset")

    args = parser.parse_args()
    print(args)

    return args


def evaluate(args, X_emd, y, random_state):
    sss = StratifiedShuffleSplit(n_splits=10, train_size=0.01, random_state=random_state)
    mdl = LogisticRegression(penalty="l2", multi_class="multinomial", max_iter=200)

    train_scores, test_scores = [], []
    for i, (train_idx, test_idx) in enumerate(sss.split(X_emd, y)):
        t = time()
        mdl.fit(X_emd[train_idx], y[train_idx])
        t = time() - t

        train_scores.append(mdl.score(X_emd[train_idx], y[train_idx]) * 100)
        test_scores.append(mdl.score(X_emd[test_idx], y[test_idx]) * 100)

        print(f"Iter{i}: train={train_scores[-1]:.2f}, "
              f"test={test_scores[-1]:.2f}, time={t:.2f} sec")

    if not args.nooutput:
        output_fp = f"{OUTPUT_DIR}/n2v{'plus' if args.extend else ''}_q={args.q}_{args.run_idx}.csv"
        method = "Node2vec+" if args.extend else "Node2vec"
        with open(output_fp, "w") as f:
            f.write("Fold,Training score,Testing score,Method,q\n")
            for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
                f.write(f"{i},{train_score},{test_score},{method},{args.q}\n")


def get_emds(args):
    if args.test:
        emd_npz = np.load(TEST_EMD_FP)
        X_emd = emd_npz["data"]
        emd_IDs = emd_npz["IDs"]
    else:
        g = node2vec.SparseOTF(1, args.q, args.workers, False, args.extend)
        g.read_npz(NETWORK_FP, True, False)
        X_emd = g.embed()
        emd_IDs = g.IDlst

    emd_IDmap = {j: i for i, j in enumerate(emd_IDs)}

    return X_emd, emd_IDmap


def get_labels(args, emd_IDmap):
    label_dict = {}
    label_count = {}
    tot_count = 0

    with open(LABEL_FP, "r") as f:
        for line in f:
            name, label = line.strip().split("\t")
            label_dict[name] = label

            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
            tot_count += 1

    assert tot_count == len(emd_IDmap)

    # only use top 10 classes for evaluation
    print("Extracting top 10 abundant categories / classses for evaluation")
    top_classes = sorted(label_count, key=label_count.get, reverse=True)[:10]
    for idx, class_id in enumerate(top_classes):
        count = label_count[class_id]
        print(f"class {idx}: n = {count:,d} ({100 * count / tot_count:05.2f}%)")

    tot_cvg = sum([label_count[i] for i in top_classes]) / tot_count * 100
    print(f"Total coverage: {tot_cvg:05.2f}%")

    eval_ind = np.zeros(tot_count, dtype=bool)
    y = np.zeros(tot_count, dtype=int) - 1
    for product, idx in emd_IDmap.items():
        label = label_dict[product]
        if label in top_classes:
            eval_ind[idx] = True
            y[idx] = top_classes.index(label)

    return y, eval_ind


def main():
    args = parse_args()
    X_emd, emd_IDmap = get_emds(args)
    y, eval_ind = get_labels(args, emd_IDmap)
    evaluate(args, X_emd[eval_ind], y[eval_ind], args.random_state)


if __name__ == "__main__":
    main()

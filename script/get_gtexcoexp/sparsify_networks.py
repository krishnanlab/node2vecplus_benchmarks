"""Finding the optimal cut threshold for GTEx coexpression networks.

For each network, gradually increase the cut threshold below which the edges
are removed. Stop once the number of components becomes above one (i.e., the
graph is broken), this threshold will be the optimal cut threshold for the
network. At the end, report the minimum of the optimal thresholds and create
sparsified versions of the networks accordingly.

"""
import pathlib
from glob import glob

import numpy as np
from NLEval import graph

MIN_CUT_THRESHOLD = 0
MAX_CUT_THRESHOLD = 1
NUM_SPACE = 20
SUFX = "Top"  # suffix for the name of the sparsified networks

HOMEDIR = pathlib.Path(__file__).resolve().parents[2]
DATADIR = HOMEDIR / "data" / "networks" / "ppi" / "gtexcoexp"
print(f"{HOMEDIR=}\n{DATADIR=}")

filepaths = list(filter(lambda x: SUFX not in x, glob(f"{DATADIR}/*.npz")))
filenames = [pathlib.Path(i).with_suffix("").name for i in filepaths]

# Finding the optimal cut threshold across all networks
cut_list = []
for filepath, filename in zip(filepaths, filenames):
    raw = np.load(filepath)
    g = graph.DenseGraph.from_mat(raw["data"], raw["IDs"].tolist())
    tot_num_edges = (g.mat != 0).sum()

    for i in np.linspace(MIN_CUT_THRESHOLD, MAX_CUT_THRESHOLD, NUM_SPACE):
        g.mat[g.mat < i] = 0
        if len(g.connected_components()) > 1:
            break
        cut = i
        percent_edges = (g.mat != 0).sum() / tot_num_edges
    cut_list.append(cut)

    print(f"{filename:<25}Max cut: {cut:.4f}  %Edges: {percent_edges:.2%}")

optim_cut = min(cut_list)
print(f"Global cut: {optim_cut}\nStart generating sparsified networks...")

# Writing sparsified networks
for filepath, filename in zip(filepaths, filenames):
    raw = np.load(filepath)
    adjmat = raw["data"]
    adjmat[adjmat < optim_cut] = 0

    terms = filename.split("-")
    filename_new = "-".join([f"{terms[0]}{SUFX}", *terms[1:]])
    print(f"Writting {filename_new}")

    np.savez(DATADIR / filename_new, data=adjmat, IDs=raw["IDs"])

print("Done!")

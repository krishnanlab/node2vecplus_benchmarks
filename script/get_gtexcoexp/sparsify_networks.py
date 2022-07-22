"""Finding the optimal cut threshold for GTEx coexpression networks.

For each network, gradually increase the cut threshold below which the edges
are removed. Stop once the number of nodes in the largest connected component
drop below a certain value, determined by the fraction with respect to the
original number of nodes. This threshold will be the optimal cut threshold for
the network. At the end, report the minimum of the optimal thresholds and
create sparsified versions of the networks accordingly.

"""
import pathlib
from glob import glob

import numpy as np
from NLEval import graph

MIN_CUT_THRESHOLD = 0
MAX_CUT_THRESHOLD = 3
NUM_SPACE = 20
MIN_NODE_PRESERVED_RATIO = 0.98
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
    tot_num_nodes = g.size
    tot_num_edges = (g.mat != 0).sum()

    # Gradually increase cut threshold and stop whenever the fraction of nodes
    # removed drop below the prespecified value
    for i in np.linspace(MIN_CUT_THRESHOLD, MAX_CUT_THRESHOLD, NUM_SPACE):
        g.mat[g.mat < i] = 0
        g = g.largest_connected_subgraph()

        if g.size / tot_num_nodes < MIN_NODE_PRESERVED_RATIO:
            break
        percent_nodes = g.size / tot_num_nodes
        percent_edges = (g.mat != 0).sum() / tot_num_edges
        cut = i

    cut_list.append(cut)
    print(
        f"{filename:<25}Max cut: {cut:.4f}  "
        f"%Nodes: {percent_nodes:7.2%}  "
        f"%Edges: {percent_edges:7.2%}",
    )

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

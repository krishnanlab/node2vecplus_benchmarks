import argparse
import itertools
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv
from torch_geometric.data import Data

from util import *


OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_gnn"
NETWORK_DIR = f"{DATA_DIR}/networks/ppi"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

check_dirs([RESULT_DIR, OUTPUT_DIR])

#####DEFAULT PARAMETERS#####
EVAL_STEPS = 100
SCHEDULER_PATIENCE = 500

LR = 0.01
EPOCHS = 2500
WEIGHT_DECAY = 0.00001

HPARAM_GCN_DIM = 128
HPARAM_GCN_LR = 0.01
HPARAM_GCN_RESIDUAL = True
HPARAM_GCN_NUM_LAYERS = 5

HPARAM_SAGE_DIM = 128
HPARAM_SAGE_RESIDUAL = False
HPARAM_SAGE_NUM_LAYERS = 5
############################


class GNN(nn.Module):
    def __init__(
        self,
        conv_module,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        *,
        act=F.relu,
        residual: bool = True,
        dropout: float = 0.1,
        pre_mp_layers: int = 1,
        post_mp_layers: int = 2,
    ):
        super().__init__()

        self.act = act
        self.residual = residual
        self.dropout = dropout

        assert pre_mp_layers > 0
        assert num_layers > 0
        assert post_mp_layers > 0

        dim_in = in_channels
        self.pre_mp = nn.ModuleList()
        for i in range(pre_mp_layers):
            self.pre_mp.append(nn.Linear(dim_in, hidden_channels))
            dim_in = hidden_channels

        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_module(dim_in, hidden_channels))
            dim_in = hidden_channels
        dim_out = out_channels if post_mp_layers == 0 else hidden_channels
        self.convs.append(conv_module(dim_in, dim_out))

        self.post_mp = nn.ModuleList()
        for i in range(post_mp_layers - 1):
            self.post_mp.append(nn.Linear(hidden_channels, hidden_channels))
        self.post_mp.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for i in itertools.chain(self.pre_mp, self.convs, self.post_mp):
            i.reset_parameters()

    def forward(self, x, adj):
        for mp in self.pre_mp:
            x = mp(x)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_new = self.act(conv(x, adj).squeeze())
            x = x + x_new if self.residual else x_new

        for mp in self.post_mp[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.act(mp(x))
        x = self.post_mp[-1](x)

        return x


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx):
    model.eval()
    y_pred = model(data.x, data.adj)

    score_lst_lst = []
    for idx in train_idx, valid_idx, test_idx:
        score_lst = [score_func(data.y[idx, i].cpu(), y_pred[idx, i].cpu())
                     for i in range(data.y.shape[1])]
        score_lst_lst.append(score_lst)

    return score_lst_lst


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation for gene classification using GNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gene_universe", required=True, help="Name of the gene universe")
    parser.add_argument("--network", required=True, help="Name of the protein protein interaction network")
    parser.add_argument("--dataset", required=True, help="Name of geneset collection")
    parser.add_argument("--device", type=int, default=0, help="Device number indicating which GPU to use, default is 0")
    parser.add_argument("--nooutput", action="store_true", help="Disable output if specified, and print results to screen")
    parser.add_argument("--use_sage", action="store_true", help="Use GraphSAGE instead of GCN, defulat is using GCN")
    parser.add_argument("--test", action="store_true", help="Toggle test mode, run with fewer epochs")

    args = parser.parse_args()
    print(args)

    return args


def load_data(gene_universe, network, dataset, use_sage, device):
    # load network
    network_fp = get_network_fp(network)
    adj_mat, adj_ids = np.load(network_fp).values()
    adj = torch.tensor(adj_mat).float()  # dense adj

    # load labels with splits and align node ids
    label_fp = f"{LABEL_DIR}/{gene_universe}_{dataset}_label_split.npz"
    y, train_idx, valid_idx, test_idx, label_ids, gene_ids = np.load(label_fp).values()
    y, gene_ids = align_gene_ids(adj_ids, y, train_idx, valid_idx, test_idx, gene_ids)

    # converting tor torch tensor
    y = torch.tensor(y)
    train_idx = torch.tensor(train_idx, dtype=torch.long).to(device)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long).to(device)
    test_idx = torch.tensor(test_idx, dtype=torch.long).to(device)

    # construct node features
    if use_sage:
        x = torch.reshape(torch.tensor(adj_mat.sum(axis=0)).float(), (adj_mat.shape[0], -1))
    else:
        x = torch.ones(adj_mat.shape[0], 1)

    return adj, x, y, train_idx, valid_idx, test_idx, label_ids


def main(args):
    gene_universe = args.gene_universe
    network = args.network
    dataset = args.dataset
    nooutput = args.nooutput
    device = args.device
    use_sage = args.use_sage
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    method = "sage" if use_sage else "gcn"

    if args.test:
        print("WARNING: runing in testing mode")
        EPOCHS = 100
        EVAL_STEPS = 10
    else:
        EPOCHS = 30000
        EVAL_STEPS = 100

    # load and constructing data object
    adj, x, y, train_idx, valid_idx, test_idx, label_ids = load_data(gene_universe, network, dataset, use_sage, device)
    data = Data(x=x, y=y)
    data.adj = adj
    data = data.to(device)

    # initialize model and optimizer
    if use_sage:  # GraphSAGE, use degree as feature
        model_args = (DenseSAGEConv, x.shape[1], HPARAM_SAGE_DIM, y.shape[1], HPARAM_SAGE_NUM_LAYERS)
        model_kwargs = {"residual": HPARAM_SAGE_RESIDUAL}
    else:  # GCN, use constant feature
        model_args = (DenseGCNConv, x.shape[1], HPARAM_GCN_DIM, y.shape[1], HPARAM_GCN_NUM_LAYERS)
        model_kwargs = {"residual": HPARAM_GCN_RESIDUAL}

    model = GNN(*model_args, **model_kwargs).to(device)
    patience = max(1, int(SCHEDULER_PATIENCE // EVAL_STEPS))
    print(f"GNN model:\n{model}")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience, verbose=True)

    # Train model and record best performance
    best_epoch = 0
    best_valid_score = 0
    best_results = None
    tic = time.perf_counter()
    for epoch in range(1, 1 + EPOCHS):
        loss = train(model, data, train_idx, optimizer)

        if epoch % EVAL_STEPS == 0:
            results = test(model, data, train_idx, valid_idx, test_idx)
            train_score, valid_score, test_score = [sum(score_lst) / len(score_lst)
                                                    for score_lst in results]
            scheduler.step(valid_score)
            if valid_score > best_valid_score or best_results is None:
                best_epoch = epoch
                best_results = results
                best_valid_score = valid_score

            elapsed = time.perf_counter() - tic
            print(
                f"Epoch: {epoch:4d}, Loss: {loss:.4f}, "
                f"Train: {train_score:.4f}, Valid: {valid_score:.4f}, "
                f"Test: {test_score:.4f}, Best epoch so far: {best_epoch:4d} "
                f"({elapsed / EVAL_STEPS / 60 * 1000:.2f} min/kEpoch)",
            )
            tic = time.perf_counter()

    # Format final results
    result_df = pd.DataFrame()
    result_df["Training score"], result_df["Validation score"], result_df["Testing score"] = best_results
    result_df["Task"] = list(label_ids)
    result_df["Dataset"], result_df["Network"], result_df["Method"] = dataset, network, method

    # Save or print results
    if nooutput:
        print(result_df)
    else:
        result_df.to_csv(f"{OUTPUT_DIR}/{network}_{dataset}_{method}.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

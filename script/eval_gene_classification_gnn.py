import argparse
import itertools
import os
import time
from pprint import pprint
from typing import Tuple

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
HP_TUNE_OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_gnn_hp_tune"
NETWORK_DIR = f"{DATA_DIR}/networks/ppi"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

check_dirs([RESULT_DIR, OUTPUT_DIR, HP_TUNE_OUTPUT_DIR])

########## PARAMETERS ##########
EPOCHS = 50_000
EVAL_STEPS = 100
SCHEDULER_PATIENCE = 500

HPARAM_GCN_DIM = 64
HPARAM_GCN_NUM_LAYERS = 5
HPARAM_GCN_RESIDUAL = True
HPARAM_GCN_LR = 0.01
HPARAM_GCN_DROPOUT = 0.1
HPARAM_GCN_WEIGHT_DECAY = 1e-6

HPARAM_SAGE_DIM = 64
HPARAM_SAGE_NUM_LAYERS = 5
HPARAM_SAGE_RESIDUAL = False
HPARAM_SAGE_LR = 0.001
HPARAM_SAGE_DROPOUT = 0.1
HPARAM_SAGE_WEIGHT_DECAY = 1e-5
################################


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
    parser.add_argument("--runid", type=int, default=0, help="Run ID, for separating results from different runs")
    parser.add_argument("--hp_tune", action="store_true", help="Hyper-parameter tuning mode, use parsed params.")
    parser.add_argument("--nooutput", action="store_true", help="Disable output if specified")
    parser.add_argument("--use_sage", action="store_true", help="Use GraphSAGE instead of GCN, defulat is using GCN")
    parser.add_argument("--test", action="store_true", help="Toggle test mode, run with fewer epochs")
    parser.add_argument("--dry_run", action="store_true", help="Print model and parameter settings and exit.")

    group_hp = parser.add_argument_group("Hyper-parameters")
    group_hp.add_argument("--dim", type=int, default=64, help="Convolution layer hidden dimension")
    group_hp.add_argument("--num_layers", type=int, default=3, help="Number of convolution layers")
    group_hp.add_argument("--residual", action="store_true", help="Whether or not to add residual connection (skipsum)")
    group_hp.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    group_hp.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (l2 regularization)")
    group_hp.add_argument("--dropout", type=float, default=0.1, help="Dropout probability (excluding pre-mlp)")

    args = parser.parse_args()
    pprint(vars(args))

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


def parse_hparams(args: argparse.Namespace) -> Tuple[int, int, bool, float, float, float]:
    # Load default hyper-parameter settings
    if args.use_sage:
        dim = HPARAM_SAGE_DIM
        num_layers = HPARAM_SAGE_NUM_LAYERS
        residual = HPARAM_SAGE_RESIDUAL
        lr = HPARAM_SAGE_LR
        dropout = HPARAM_SAGE_DROPOUT
        weight_decay = HPARAM_SAGE_WEIGHT_DECAY
    else:
        dim = HPARAM_GCN_DIM
        num_layers = HPARAM_GCN_NUM_LAYERS
        residual = HPARAM_GCN_RESIDUAL
        lr = HPARAM_GCN_LR
        dropout = HPARAM_GCN_DROPOUT
        weight_decay = HPARAM_GCN_WEIGHT_DECAY

    # Overwrite hyper-parameters if hp_tune mode is set to True
    if args.hp_tune:
        dim = args.dim
        num_layers = args.num_layers
        residual = args.residual
        lr = args.lr
        dropout = args.dropout
        weight_decay = args.weight_decay

    print("\nHyper-parameter settings:")
    print(f"{dim=}\n{num_layers=}\n{residual=}\n{lr=}\n{dropout=}\n{weight_decay=}\n")

    return dim, num_layers, residual, lr, dropout, weight_decay


def main():
    args = parse_args()
    gene_universe = args.gene_universe
    network = args.network
    dataset = args.dataset
    nooutput = args.nooutput
    device = args.device
    hp_tune = args.hp_tune
    dry_run = args.dry_run
    use_sage = args.use_sage
    runid = args.runid
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    method = "sage" if use_sage else "gcn"

    # Parse hyper parameters and training settings
    dim, num_layers, residual, lr, dropout, weight_decay = parse_hparams(args)
    epochs, eval_steps = EPOCHS, EVAL_STEPS
    if args.test:
        print("WARNING: runing in testing mode")
        epochs = 100
        eval_steps = 10
    elif args.hp_tune:
        epochs = 2500
    patience = max(1, int(SCHEDULER_PATIENCE // eval_steps))

    # Get output path
    if not hp_tune:
        out_path = f"{OUTPUT_DIR}/{network}_{dataset}_{method}.csv"
    else:
        hparams_str = f"dim={dim}_num-layers={num_layers}_residual={1 if residual else 0}"
        hparams_str += f"_lr={lr:.0e}_dropout={dropout}_weight-decay={weight_decay:.0e}"
        out_path = f"{HP_TUNE_OUTPUT_DIR}/{method}/{hparams_str}/{network}_{dataset}_{runid}/score.csv"
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)

    # Load data
    adj, x, y, train_idx, valid_idx, test_idx, label_ids = load_data(gene_universe, network, dataset, use_sage, device)
    data = Data(x=x, y=y, adj=adj).to(device)

    # Initialize model and optimizer
    conv_module = DenseSAGEConv if use_sage else DenseGCNConv
    model = GNN(conv_module, x.shape[1], dim, y.shape[1], num_layers, residual=residual, dropout=dropout).to(device)
    print(f"GNN model:\n{model}")

    if dry_run:
        print(f"Results will be saved to {out_path}")
        exit(0)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=patience, verbose=True)

    # Train model and record best performance
    best_epoch = 0
    best_valid_score = 0
    best_results = None
    tic = time.perf_counter()
    for epoch in range(1, 1 + epochs):
        loss = train(model, data, train_idx, optimizer)

        if epoch % eval_steps == 0:
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
                f"({elapsed / eval_steps / 60 * 1000:.2f} min/kEpoch)",
            )
            tic = time.perf_counter()

    # Format final results
    result_df = pd.DataFrame()
    result_df["Training score"], result_df["Validation score"], result_df["Testing score"] = best_results
    result_df["Task"] = list(label_ids)
    result_df["Dataset"], result_df["Network"], result_df["Method"] = dataset, network, method

    # Save or print results
    print(result_df)
    if not nooutput:
        result_df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

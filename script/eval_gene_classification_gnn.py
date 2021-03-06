import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv
from torch_geometric.data import Data

from util import *


OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_gnn"
NETWORK_DIR = f"{DATA_DIR}/networks/ppi"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

check_dirs([RESULT_DIR, OUTPUT_DIR])

####DEFAULT PARAMETERS####
EVAL_STEPS = 100
HPARAM_GCN_DIM = 128
HPARAM_GCN_LR = 0.01
HPARAM_SAGE_DIM = 128
HPARAM_SAGE_LR = 0.0005
HPARAM_NUM_LAYERS = 5
HPARAM_EPOCHS = 5000
##########################


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(DenseGCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                DenseGCNConv(hidden_channels, hidden_channels))
        self.convs.append(DenseGCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = F.relu(x)
        x = torch.reshape(self.convs[-1](x, adj), (adj.shape[0], -1))
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(DenseSAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                DenseSAGEConv(hidden_channels, hidden_channels))
        self.convs.append(DenseSAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
        x = torch.reshape(self.convs[-1](x, adj), (adj.shape[0], -1))
        return x


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

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
        score_lst = [score_func(data.y[idx,i].cpu(), y_pred[idx,i].cpu())
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
    adj = torch.tensor(adj_mat).float() # dense adj

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
        HPARAM_EPOCHS = 100
        EVAL_STEPS = 10
    else:
        HPARAM_EPOCHS = 30000
        EVAL_STEPS = 100

    # load and constructing data object
    adj, x, y, train_idx, valid_idx, test_idx, label_ids= load_data(gene_universe, network, dataset, use_sage, device)
    data = Data(x=x, y=y)
    data.adj = adj
    data = data.to(device)
    
    # initialize model and optimizer
    if use_sage:  # GraphSAGE, use degree as feature
        model = SAGE(x.shape[1], HPARAM_SAGE_DIM, y.shape[1], HPARAM_NUM_LAYERS).to(device)
        lr = HPARAM_SAGE_LR
    else:  # GCN, use constant feature
        model = GCN(x.shape[1], HPARAM_GCN_DIM, y.shape[1], HPARAM_NUM_LAYERS).to(device)
        lr = HPARAM_GCN_LR
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model and record best performance
    best_epoch = 0
    best_valid_score = 0
    best_results = None
    for epoch in range(1, 1 + HPARAM_EPOCHS):
        loss = train(model, data, train_idx, optimizer)

        if epoch % EVAL_STEPS  == 0:
            results = test(model, data, train_idx, valid_idx, test_idx)
            train_score, valid_score, test_score = [sum(score_lst) / len(score_lst) 
                                                  for score_lst in results]
            if valid_score > best_valid_score:
                best_epoch = epoch
                best_results = results
                best_valid_score = valid_score

            print(
                f"Epoch: {epoch:4d}, Loss: {loss:.4f}, "
                f"Train: {train_score:.4f}, Valid: {valid_score:.4f}, "
                f"Test: {test_score:.4f}, Best epoch so far: {best_epoch:4d}",
            )

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

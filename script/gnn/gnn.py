import argparse

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, DenseGCNConv, SAGEConv, DenseSAGEConv
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from sys import path
path.append("../../../NetworkLearningEval/src/")
from NLEval.metrics import auPRC


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = F.relu(x)
        x = self.convs[-1](x, adj)
        return x


class DenseGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(DenseGCN, self).__init__()

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
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
        x = self.convs[-1](x, adj)
        return x


class DenseSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(DenseSAGE, self).__init__()

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


def train(model, data, train_idx, optimizer, pos_weight):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer.zero_grad()
    out = model(data.x, data.adj)
    loss = criterion(out, data.y.to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx):
    model.eval()
    y_pred = model(data.x, data.adj)

    score_lst_lst = []
    for idx in train_idx, valid_idx, test_idx:
        score_lst = [auPRC(data.y[idx,i].cpu(), y_pred[idx,i].cpu()) 
                     for i in range(data.y.shape[1])]
        score_lst_lst.append(score_lst)

    return score_lst_lst


def main():
    parser = argparse.ArgumentParser(description="GNN")
    parser.add_argument('--network_path', nargs="?", required=True)
    parser.add_argument('--dataset_path', nargs="?", required=True)
    parser.add_argument('--out_path', nargs="?", default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_dense', action='store_true')
    args = parser.parse_args()
    print(args)

    network_path = args.network_path
    dataset_path = args.dataset_path
    out_path = args.out_path
    device = args.device
    dim = args.dim
    num_layers = args.num_layers
    runs = args.runs
    lr = args.lr
    epochs = args.epochs
    eval_steps = args.eval_steps
    use_sage = args.use_sage
    use_dense = args.use_dense

    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    
    adj_mat = np.load(network_path)['data']
    y, train_idx, valid_idx, test_idx, _, _ = np.load(dataset_path).values()

    tot_num = train_idx.size + valid_idx.size + test_idx.size
    pos_weight = (tot_num - y.sum(axis=0)) / y.sum(axis=0)
    print("pos_weight = ", pos_weight)

    if out_path is not None:
        with open(out_path, 'w') as f:
            f.write(f"{args}\npos_weight = {pos_weight}\n\n")
    
    if use_dense:
        adj = torch.tensor(adj_mat).float() # dense adj
    else:
        adj = SparseTensor.from_dense(torch.tensor(adj_mat)).float() # sparse adj
    y = torch.tensor(y)
    pos_weight = torch.tensor(pos_weight).to(device)
    
    if not use_sage:
        x = torch.ones(adj_mat.shape[0], 1)  # trivial feature
        if use_dense:
            model = DenseGCN(x.shape[1], dim, y.shape[1], num_layers).to(device)
        else:
            model = GCN(x.shape[1], dim, y.shape[1], num_layers).to(device)
    else:
        x = torch.reshape(torch.tensor(adj_mat.sum(axis=0)).float(), (adj_mat.shape[0], -1)) # degree as feature
        if use_dense:
            model = DenseSAGE(x.shape[1], dim, y.shape[1], num_layers).to(device)
        else:
            model = SAGE(x.shape[1], dim, y.shape[1], num_layers).to(device)
    
    data = Data(x=x, y=y)
    data.adj = adj
    
    data = data.to(device)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long).to(device)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long).to(device)
    test_idx = torch.tensor(test_idx, dtype=torch.long).to(device)
    
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx, optimizer, pos_weight)
    
            if epoch % eval_steps  == 0:
                result = test(model, data, train_idx, valid_idx, test_idx)
                train_score, val_score, test_score = [sum(score_lst) / len(score_lst) 
                                                      for score_lst in result]
                out_str = f'Run: {run + 1:02d}, ' + \
                          f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ' + \
                          f'Train: {train_score:.4f}, ' + \
                          f'Valid: {val_score:.4f}, Test: {test_score:.4f}'
                print(out_str)

                if out_path is not None:
                    with open(out_path, 'a') as f:
                        f.write(f'{out_str}\n')

        result = test(model, data, train_idx, valid_idx, test_idx)
        for i, (train_score, val_score, test_score) in enumerate(zip(*result)):
            out_str = f'Final evaluation. ' + \
                      f'Run: {run + 1:02d}, Labelset: {i:03d}, ' + \
                      f'Loss: {loss:.4f}, Train: {train_score:.4f}, ' + \
                      f'Valid: {val_score:.4f}, Test: {test_score:.4f}'
            if out_path is not None:
                with open(out_path, 'a') as f:
                    f.write(f'{out_str}\n')


if __name__ == "__main__":
    main()

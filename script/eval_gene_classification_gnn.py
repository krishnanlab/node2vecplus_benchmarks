import os
import argparse

import numpy as np
from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv
from torch_geometric.data import Data


DATA_DIR = "../data"
RESULT_DIR = "../result"
OUTPUT_DIR = f"{RESULT_DIR}/gene_classification_gnn"
NETWORK_DIR = f"{DATA_DIR}/networks/ppi"
LABEL_DIR = f"{DATA_DIR}/labels/gene_classification"

# check if output directory exist and create it not
if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

####DEFAULT PARAMETERS####
EVAL_STEPS = 1000
HPARAM_GCN_DIM = 128
HPARAM_GCN_LR = 0.01
HPARAM_SAGE_DIM = 64
HPARAM_SAGE_LR = 0.0005
HPARAM_NUM_LAYERS = 3
HPARAM_EPOCHS = 100000
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


def train(model, data, train_idx, optimizer, pos_weight):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer.zero_grad()
    out = model(data.x, data.adj)
    loss = criterion(out, data.y.to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


def score_func(y_true, y_pred):
    prior = y_true.sum() / y_true.size
    auprc = average_precision_score(y_true, y_pred)
    return np.log2(auprc / prior)


@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx):
    model.eval()
    y_pred_all = model(data.x, data.adj)

    score_lst_lst = []
    for idx in train_idx, valid_idx, test_idx:
        y_true = data.y[idx,i].cpu()
        y_pred = y_pred_all[idx,i].cpu()
        score_lst = [score_func(y_true, y_pred) for i in range(data.y.shape[1])]
        score_lst_lst.append(score_lst)

    return score_lst_lst


def main():
    parser = argparse.ArgumentParser(description="GNN")
    parser.add_argument('--network', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--nooutput', action='stor_true')
    parser.add_argument('--use_sage', action='store_true')
    args = parser.parse_args()
    print(args)

    network = args.network
    dataset = args.dataset
    nooutput = args.nooutput
    device = args.device
    use_sage = args.use_sage

    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    method = 'sage' if use_sage else 'gcn'

    network_fp = f"{NETWORK_DIR}/{network}.npz"
    label_fp = f"{LABEL_DIR}/{network}_{dataset}_label_split.npz"
    output_fp = f"{OUTPUT_DIR}/{network}_{dataset}_{method}.csv"
    
    adj_mat = np.load(network_fp)['data']
    y, train_idx, valid_idx, test_idx, label_ids, _ = np.load(label_fp).values()

    tot_num = train_idx.size + valid_idx.size + test_idx.size
    pos_weight = (tot_num - y.sum(axis=0)) / y.sum(axis=0)
    print("pos_weight = ", pos_weight)

    adj = torch.tensor(adj_mat).float() # dense adj
    y = torch.tensor(y)
    pos_weight = torch.tensor(pos_weight).to(device)
    
    if use_sage:  # GraphSAGE, use degree as feature
        x = torch.reshape(torch.tensor(adj_mat.sum(axis=0)).float(), (adj_mat.shape[0], -1))
        model = SAGE(x.shape[1], HPARAM_SAGE_DIM, y.shape[1], HPARAM_NUM_LAYERS).to(device)
        lr = HPARAM_SAGE_LR
    else:  # GCN, use constant feature
        x = torch.ones(adj_mat.shape[0], 1)
        model = GCN(x.shape[1], HPARAM_GCN_DIM, y.shape[1], HPARAM_NUM_LAYERS).to(device)
        lr = HPARAM_GCN_LR
    
    data = Data(x=x, y=y)
    data.adj = adj
    data = data.to(device)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long).to(device)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long).to(device)
    test_idx = torch.tensor(test_idx, dtype=torch.long).to(device)
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 1 + HPARAM_EPOCHS):
        loss = train(model, data, train_idx, optimizer, pos_weight)

        if epoch % EVAL_STEPS  == 0:
            results = test(model, data, train_idx, valid_idx, test_idx)
            train_score, valid_score, test_score = [sum(score_lst) / len(score_lst) 
                                                  for score_lst in results]
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ' + \
                  f'Train: {train_score:.4f}, ' + \
                  f'Valid: {valid_score:.4f}, Test: {test_score:.4f}')

    # final evaluation
    results = test(model, data, train_idx, valid_idx, test_idx)
    result_df = pd.DataFrame()
    result_df['Training score'], result_df['Validation score'], result_df['Testing score'] = results
    result_df['Task'] = list(label_ids)
    result_df['Dataset'], result_df['Network'], result_df['Method'] = dataset, network, method

    # save or print results
    if nooutput:
        print(result_df)
    else:
        result_df.to_csv(output_fp, index=False)


if __name__ == "__main__":
    main()


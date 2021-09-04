#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

python gnn.py --network_path ../../data/networks/ppi/STRING.npz --dataset_path ../../data/labels/gene_classification/STRING_KEGGBP_label_split.npz --epochs 100 --eval_steps 10


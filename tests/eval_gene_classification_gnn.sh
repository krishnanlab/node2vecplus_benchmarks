#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

homedir=$(dirname $(dirname $(realpath $0)))
echo homedir=$homedir

cd $homedir/script

python eval_gene_classification_gnn.py --network STRING --dataset GOBP --test
python eval_gene_classification_gnn.py --network STRING --dataset GOBP --use_sage --test

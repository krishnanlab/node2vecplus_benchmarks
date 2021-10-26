#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

homedir=$(dirname $(dirname $(realpath $0)))
echo homedir=$homedir

cd $homedir/script

python eval_gene_classification_n2v.py --network STRING --nooutput --p 1 --q 0.01
python eval_gene_classification_n2v.py --network STRING --nooutput --p 1 --q 0.01 --extend

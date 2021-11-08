#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

homedir=$(dirname $(dirname $(dirname $(realpath $0))))
echo homedir=$homedir

cd $homedir/script

python eval_product_full.py --q 0.1 --test --nooutput

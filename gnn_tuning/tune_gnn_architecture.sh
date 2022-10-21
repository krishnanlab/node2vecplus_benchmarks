#!/bin/bash --login

dim_opts=(16 32 64 128)
num_layers_opts=(2 3 4 5)

base_script="python eval_gene_classification_gnn.py --hp_tune --lr 0.01 --dropout 0.1 --weight_decay 0.00001"

for dim in ${dim_opts[@]}; do
    for num_layers in ${num_layers_opts[@]}; do
        script="${base_script} --dim ${dim} --num_layers ${num_layers}"

        sbatch -J tune_gnn_architecture job_template.sb "${script}"
        sbatch -J tune_gnn_architecture job_template.sb "${script} --residual"
        sbatch -J tune_gnn_architecture job_template.sb "${script} --use_sage"
        sbatch -J tune_gnn_architecture job_template.sb "${script} --use_sage --residual"
    done
done

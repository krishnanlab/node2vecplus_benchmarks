#!/bin/bash --login

gcn_dim=64
gcn_num_layers=5
gcn_residual=1

sage_dim=64
sage_num_layers=5
sage_residual=0

lr_opts=(0.0001 0.001 0.01 0.1)
dropout_opts=(0.0 0.1 0.3)
weight_decay_opts=(0.0001 0.00001 0.000001 0.0000001)

base_script="python eval_gene_classification_gnn.py --hp_tune"

function submit_job {
    model=$1

    if [ $model = gcn ]; then
        script="${base_script} --dim ${gcn_dim} --num_layers ${gcn_num_layers}"
        (( gcn_residual == 1 )) && script+=" --residual"
    else
        script="${base_script} --use_sage --dim ${sage_dim} --num_layers ${sage_num_layers}"
        (( sage_residual == 1 )) && script+=" --residual"
    fi
    script+=" --lr ${lr} --dropout ${dropout} --weight_decay ${weight_decay}"

    echo $script
    sbatch -J tune_gnn_params job_template.sb $script
}

for lr in ${lr_opts[@]}; do
    for dropout in ${dropout_opts[@]}; do
        for weight_decay in ${weight_decay_opts[@]}; do
            submit_job gcn
            submit_job sage
        done
    done
done

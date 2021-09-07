#!/bin/bash --login

cd slurm

# submit jobs to specified nodes, modify or remove specification if needed
sbatch -C amd20 eval_hcluster_all.sb
sbatch -C amd20 eval_gene_classification_n2v.sb
sbatch eval_gene_classification_gnn.sb


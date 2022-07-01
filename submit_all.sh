#!/bin/bash --login

cd slurm

# submit jobs to specified nodes, modify or remove specification if needed
sbatch -C amd20 eval_hcluster_all.sb
sbatch -C amd20 eval_gene_classification_n2vplus.sb
sbatch -C amd20 eval_gene_classification_n2v.sb
sbatch -C amd20 eval_gene_tissue_classification_n2vplus.sb
sbatch -C amd20 eval_gene_tissue_classification_n2v.sb
sbatch -C amd20 eval_realworld_networks.sb

# submit job with no specifications
sbatch eval_gene_classification_gnn.sb

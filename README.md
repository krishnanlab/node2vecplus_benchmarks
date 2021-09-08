# Node2vec+ Benchmarks

This repository contains data and scripts for reproducing evaluation results presented in 
[*Accurately modeling biased random walks on weighted graphs using node2vec+*](). 
Node2vec+ is implemented as an extension to [PecanPy](https://github.com/krishnanlab/PecanPy), 
a fast and memory efficient implementation of [node2vec](https://snap.stanford.edu/node2vec/). 

# Quick start

Follow the scripts below to execute full evaluation provaided in this repository. 
For more details, check out the sections below. 

* [Setup environment](#setting-up-environment)
* [Download PPIs](#downloading-ppis)
* [Evaluate](#evaluation)

***PROCEED WITH CAUTION: the full evaluation consumes significant amount of space and computational resources (via [SLURM](https://slurm.schedmd.com/overview.html))***

```bash
cd script/init_setup

# setup conda environment
sh setup_env.sh

# download ppis
sh download_ppis.sh

# submit evaluation jobs
cd ../../slurm
sh submit_all.sh
```

After all evaluation jobs are finished successfully, open the jupyter notebooks in `plot/` and generate evaluation plots. 

# Setting up environment

Requirements forthe conda environment `node2vecplus-bench`:

* [Python 3.8](https://www.python.org/downloads/release/python-3810/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [PecanPy](https://github.com/krishnanlab/PecanPy) (latest dev version)
* [PyTorch](https://pytorch.org/)
* [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Setup manually

Follow steps below to setup the environment for testing.

***Note: need to setup CUDA10.2 properly before installing pytorch geometirc***

```bash
# create and activate conda environment
conda create -n node2vecplus-bench python=3.8 pandas scikit-learn -y
conda activate node2vecplus-bench

# download and install development version of PecanPy
git clone https://github.com/krishnanlab/pecanpy.git
cd pecanpy
pip install -e .

# setup pytorch geometric (CUDA10.2)
conda install pytorch cudatoolkit=10.2 -c pytorch -y
conda install pytorch-geometric -c rusty1s -c conda-forge -y
```

## Setup using bash script

Alternatively, one can run the `setup_env.sh` script to setup the `node2vecplus-bench` conda environment:

```bash
cd script/init_setup
sh setup_env.sh

# to remove the node2vecplus-bench conda environment, run the following
sh clean_env.sh
```

Note that this script uses the following two lines to load GCC8.3 and CUDA10.2. 
If this is not compatible with your system, you need to modify them accordingly before executing the script.

```bash
module load GCC/8.3.0
module load CUDA/10
```

# Data

* Hierarchical cluster graphs
* Real world (small) networks
    * `BlogCatalog` (10,312 nodes, 333,983 edges)
    * `Wikipedia` (4,777 nodes, 92,406 edges)
* Protein-protein interaction networks (*need to download, see below*)
    * `STRING` (17,352 nodes, 3,640,737 edges)
    * `GIANT-TN` (25,825 nodes, 333,452,400 edges)
    * `GIANT-TN-c01` (25,689 nodes, 38,904,929 edges)

The hierarchical cluster graphs are constructed by taking RBF of point coulds generated in the Euclidean space, 
and hence each graph natually exhibits a hierarchical community structure (more info in the supplementary materials of the paper). 
Each network is assocaited with two tasks, cluster classification and level classification.

The BlogCatalog and Wikipedia networks, along with the associated node labels, are obtained from [SNAP-node2vec](https://snap.stanford.edu/node2vec/). 
The networks are processed by removing isolated nodes and converting to edge list tsv files.

## Downloading PPIs

Due to the file size limitation, the PPIs are not uploaded to this GitHub repository. 
Instead, they are pulled from two external data repositories 
[GenePlexus](https://zenodo.org/record/3352348/#.YTejK9NKhzU) and [GIANT](http://giant.princeton.edu/). 

Run the following script to pull and preprocess networks (takes 1~2 hours to complete)

***WARNING: takes up 21GB of space, proceed with caution!!!***

```bash
cd script/init_setup
sh download_ppis.sh
```

The labels for gene classificaitons are available under `data/labels/gene_classification/`, 
processed for each PPI network following [GenePlexus](https://academic.oup.com/bioinformatics/article/36/11/3457/5780279)
* `KEGGBP`
* `GOBP`
* `DisGeNet`

# Evaluation

This repository contains the following scripts for reproducing the evaluation results

* `eval_hcluster.py` - evaluate performnace of node2vec(+) using hierarchical cluster graphs
* `eval_realworld_networks.py` - evaluate performance of node2vec(+) using commonly benchmarked real-world datasets BlogCatalog and Wikipedia
* `eval_gene_classification_n2v.py` - evalute performance of node2vec(+) for gene classification tasks using PPI networks
* `eval_gene_classification_gnn.py` - evaluate performance of GNNs for gene classification tasks using PPI networks

Each one of the above scripts can be run from command line, e.g.

```bash
cd script

# example of evaluating K3L2 hierarchical cluster graph using node2vec with q=10
python evalu_hcluster.py --network K3L2 --q 10 --nooutput

# sample as above but using node2vec+
python evalu_hcluster.py --netwokr K3L2 --q 10 --nooutput --extend

# check other commandline keyward options 
python eval_hcluster.py --help
```

If `--nooutput` is not specified, then the evaluation results are saved to `result/` as `.csv`.

## Submitting evaluation jobs

Alternatively, one can submit evaluation jobs using

```bash
cd slurm

# submit all evaluations on hierarchical cluster graphs
sbatch eval_hcluster_all.sb

# submit all evaluations for BlogCatalog and Wikipedia
sbatch eval_realworld_networks.sb

# submit all evaluations for gene classifications using node2vec(+)
sbatch eval_gene_classification_n2v.sb

# submit all evaluations for gene classifications using GNNs
sbatch eval_gene_classification_gnn.sb
```

Or submitting all evaluations above by simply running

```bash
sh evaluate_all.sh
```

Note: depending on the your preference you can modify the nodes requirement in `evaluate_all.sh` for individual jobs script.


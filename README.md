# Node2vec+ Benchmarks

# Quick start

* setup environment
* download PPIs
* submit job scripts

# Setting up environment

First, setup the conda environment `node2vecplus-bench`. 

Requirements:
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
and it natually exhibits a hierarchical community structure (more info in the supplementary materials of the paper). 
Each network is assocaited with two tasks, cluster classification and level classification.

The BlogCatalog and Wikipedia networks, along with the associated node labels, are obtained from [SNAP-node2vec](https://snap.stanford.edu/node2vec/). 
The networks are processed by removing isolated nodes and converting to edge list tsv files.

## Downloading PPIs

Due to file size limitation, the PPIs are not uploaded to this GitHub repository. 
Instead, they are pulled from two external data repositories 
[GenePlexus](https://zenodo.org/record/3352348/#.YTejK9NKhzU) and [GIANT](http://giant.princeton.edu/). 

Run the following script to pull and preprocess networks (takes 1~2 hours to complete)

***WARNING: takes up 21GB of space, proceed with caution!!!***

```bash
cd script/init_setup
sh download_ppis.sh
```



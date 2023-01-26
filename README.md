# Node2vec+ Benchmarks [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7573612.svg)](https://doi.org/10.5281/zenodo.7573612)

This repository contains data and scripts for reproducing evaluation results presented in 
[*Accurately modeling biased random walks on weighted networks using node2vec+*](https://www.biorxiv.org/content/early/2022/08/15/2022.08.14.503926).
Node2vec+ is implemented as an extension to [PecanPy](https://github.com/krishnanlab/PecanPy), 
a fast and memory efficient implementation of [node2vec](https://snap.stanford.edu/node2vec/). 

## Overview

Follow the scripts below to execute full evaluation provaided in this repository. 
For more details, check out the sections below. 

* [Set up conda environment](#setting-up-environment)
* [Set up gene interaction network data](#download)
* [Evaluate](#evaluation)

***PROCEED WITH CAUTION: the full evaluation consumes significant amount of space and computational resources (via [SLURM](https://slurm.schedmd.com/overview.html))***

```bash
# Set up conda environment
source config.sh setup

# Download and set up gene interaction network data
source config.sh download_ppis

# Submit all evaluation jobs
sh submit_all.sh
```

After all evaluation jobs are finished successfully, open the jupyter notebooks in [`plot/`](plot) and generate evaluation plots.

## Setting up environment

We provide a simple script to set up the [conda](https://conda.io/projects/conda/en/latest/index.html) environemnt `node2vecplus-bench`:

```bash
source config.sh setup
```

To remove the environment, simply run

```bash
source config.sh cleanup
```

### Set up manually

Alternatively, user can set up the environment manually instead of using the `config.sh` script.
Additionally all the required dependencies can be found in [`requirements.txt`](requirements.txt).

* **Step1.** Set up node2vecpluc-bench conda environment with Python 3.8

    ```bash
    conda create -n node2vecplus-bench python=3.8 && conda activate node2vecplus-bench
    ```

* **Step2.** Set up [PyTorch](https://pytorch.org) related packages with CUDA 10.2 (checkout the PyTorch website for other CUDA/CPU installation options)

    ```bash
    conda install pytorch=1.9 torchvision cudatoolkit=10.2 -c pytorch -y
    pip install torch-geometric==2.0.0 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
    ```

* **Step3.** Install rest of the depencies for reproducing experiemnts

    ```bash
    pip install -r requirements.txt
    ```

## Data

* Hierarchical cluster graphs
* Standard benchmarking datasets
    * `BlogCatalog`
    * `Wikipedia`
* Human gene interaction networks (*need to download, see below*)
    * `STRING`
    * `HumanBase*`
    * `GTExCoExp*`

The hierarchical cluster graphs are constructed by taking RBF of point coulds generated in the Euclidean space, 
and hence each graph natually exhibits a hierarchical community structure (more info in the supplementary materials of the paper). 
Each network is assocaited with two tasks, cluster classification and level classification.

The BlogCatalog and Wikipedia networks, along with the associated node labels, are obtained from [SNAP-node2vec](https://snap.stanford.edu/node2vec/). 
The networks are processed by removing isolated nodes and converting to edge list tsv files.

### Gene interaction networks [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7007164.svg)](https://doi.org/10.5281/zenodo.7007164)

```bash
source config.sh download_ppis
```

#### Download

Under the root directory of the repository, download gene interaction networks from Zenodo

```bash
curl -o node2vecplus_bench_ppis.tar.gz https://zenodo.org/record/7007164/files/node2vecplus_bench_ppis.tar.gz
```

(Recommended) Although Zenodo provide a nice feature for versioning datasets with DOI, downloading could be a bit slow.
Thus, we provide an alternative download option from Dropbox.
The file should be in sync with the latest dataset version on Zenodo.

```bash
curl -L -o node2vecplus_bench_ppis.tar.gz https://www.dropbox.com/s/aettebq5lbgu1cu/node2vecplus_bench_ppis-v1.0.0.tar.gz?dl=1
```

#### Extract

After the zipped tar ball is downloaded, extract and place them under `data/networks` by

```bash
tar -xzvf node2vecplus_bench_ppis.tar.gz --transform 's/node2vecplus_bench_ppis/ppi/' --directory data/networks
```

## Evaluation

This repository contains the following scripts for reproducing the evaluation results

* [`eval_hcluster.py`](script/eval_hcluster.py) - evaluate performnace of node2vec(+) using hierarchical cluster graphs
* [`eval_realworld_networks.py`](script/eval_realworld_networks.py) - evaluate performance of node2vec(+) using commonly benchmarked real-world datasets BlogCatalog and Wikipedia
* [`eval_gene_classification_n2v.py`](script/eval_gene_classification_n2v.py) - evalute performance of node2vec(+) for gene classification tasks using gene interaction networks
* [`eval_gene_classification_gnn.py`](script/eval_gene_classification_gnn.py) - evaluate performance of GNNs for gene classification tasks using gene interaction networks

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

If `--nooutput` is not specified, then the evaluation results are saved to [`result/`](result) as `.csv`.

### Submitting evaluation jobs

Alternatively, one can submit evaluation jobs using

```bash
cd slurm

# submit all evaluations on hierarchical cluster graphs
sbatch eval_hcluster_all.sb

# submit all evaluations for BlogCatalog and Wikipedia
sbatch eval_realworld_networks.sb

# submit all evaluations for gene classifications using node2vec+
sbatch eval_gene_classification_n2vplus.sb

# submit all evaluations for gene classifications using node2vec
sbatch eval_gene_classification_n2v.sb

# submit all evaluations for gene classifications using GNNs
sbatch eval_gene_classification_gnn.sb

# submit all evaluations for tissue-specific gene classifications using node2vec+
sbatch eval_tissue_gene_classification_n2vplus.sb

# submit all evaluations for tissue-specific gene classifications using node2vec
sbatch eval_tissue_gene_classification_n2v.sb
```

Or submitting all evaluations above by simply running

```bash
sh submit_all.sh
```

Note: depending on the your preference you can modify the nodes requirement in [`submit_all.sh`](submit_all.sh) for individual jobs script.

#### Tuning GNNs

First, tune the architecture of GNN (hidden dimension, number of layers, residual connection)

```bash
cd gnn_tuning
sh tune_gnn_architecture.sb
```

Then, fix the best architecture and tune the rest of the training parameters (learning rate, dropout rate, weight decay)

```bash
cd gnn_tuning
sh tune_gnn_params.sb
```

To aggregate the gnn tuning results, use [`aggregate_tuning_results.py`](gnn_tuning/aggregate_tuning_results.py):

```bash
python gnn_tuning/aggregate_tuning_results.py
```

Finally, use the [GNN tuning notebook](plot/tune_gnn.ipynb) to analyze the results and find the optimal GNN configurations.

## Dev notes

Example test commands

```bash
python eval_gene_classification_n2v.py --gene_universe HBGTX --network HumanBaseTop-global --p 1 --q 1 --nooutput --test
```

### Setting up gene interaction network (from scratch)

* [STRING](https://doi.org/10.5281/zenodo.3352323)
* [HumanBase](script/get_humanbase/README.md)
* [GTExCoExp](script/get_gtexcoexp/README.md)

### Generating labeled data for gene classification

Install additional dev dependencies

```bash
pip install -r requirements-dev.txt
```

Once the network data are set up and placed under ``data/networks/ppi``, run

```bash
process_labels.py
```

### Update gene interaction network data on Zenodo

1. Make new dataset version on zenodo and upload corresponding file
1. Upload file to dropbox for alternative download option
1. Update README (Zenodo DOI, Zenodo link, Dropbox link)
1. Update ``config.sh`` Dropbox link

## Cite us

If you find this work useful, please consider citing our paper:

```bibtex
@article {liu2022node2vecplus,
	title = {Accurately modeling biased random walks on weighted networks using node2vec+},
	author = {Liu, Renming and Hirn, Matthew and Krishnan, Arjun},
	year = {2022},
	doi = {10.1101/2022.08.14.503926},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
	URL = {https://www.biorxiv.org/content/early/2022/08/15/2022.08.14.503926},
	eprint = {https://www.biorxiv.org/content/early/2022/08/15/2022.08.14.503926.full.pdf},
}
```

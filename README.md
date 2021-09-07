# Node2vec+ Benchmarks

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
```

Note that this script uses the following two lines to load GCC8.3 and CUDA10.2. 
If this is not compatible with your system, you need to modify them accordingly before executing the script.

```bash
module load GCC/8.3.0
module load CUDA/10
```


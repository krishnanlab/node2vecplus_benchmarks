# Node2vec+ Benchmarks


# Setting up environment

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


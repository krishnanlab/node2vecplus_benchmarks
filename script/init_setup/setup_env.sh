#!/bin/bash --login

source ~/.bashrc
module load GCC/8.3.0
module load CUDA/10

sh clean_env.sh

echo Setting up conda environment...

conda create -n node2vecplus-bench python=3.8 pandas scikit-learn -y
conda activate node2vecplus-bench

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $home_dir
echo home_dir=$home_dir
echo

echo Download and install development version of PecanPy
git clone https://github.com/krishnanlab/pecanpy.git
cd pecanpy
pip install -e .

echo Setting up Pytorch Geometric
conda install pytorch cudatoolkit=10.2 -c pytorch -y
conda install pytorch-geometric -c rusty1s -c conda-forge -y


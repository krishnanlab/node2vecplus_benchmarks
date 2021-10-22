#!/bin/bash --login

source ~/.bashrc
module load GCC/10.2.0 CUDA/11.1.1

sh clean_env.sh

echo Setting up conda environment...

conda create -n node2vecplus-bench python=3.8 -y
conda activate node2vecplus-bench

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $home_dir
echo home_dir=$home_dir
echo

echo Setting up Pytorch Geometric
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
conda install pyg -c pyg -c conda-forge -y
conda install numba pandas scikit-learn -y

echo Download and install development version of PecanPy
git clone https://github.com/krishnanlab/pecanpy.git
cd pecanpy
pip install -e .

conda clean --all -y

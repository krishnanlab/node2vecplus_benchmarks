#!/bin/bash --login

source ~/.bashrc
module load GCC/8.3.0 CUDA/10.2.89

sh clean_env.sh

echo Setting up conda environment...

conda create -n node2vecplus-bench python=3.8 -y
conda activate node2vecplus-bench

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
cd $home_dir
echo home_dir=$home_dir
echo

echo Setting up Pytorch Geometric and other dependencies
conda install pytorch=1.9 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge -y
conda install pandas scikit-learn -y

echo Installing PecanPy v2.0.2
pip install pecanpy==2.0.2

conda clean --all -y

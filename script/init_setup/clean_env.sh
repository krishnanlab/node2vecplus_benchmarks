#!/bin/bash --login

source ~/.bashrc

conda remove --name node2vecplus-bench --all -y
home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
rm -rf $home_dir/pecanpy



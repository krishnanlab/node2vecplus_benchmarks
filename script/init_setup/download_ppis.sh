#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

echo "Start processing PPI networks (the whole process should take about 1-2 hours)"

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/data/networks

if [ ! -d "ppi" ]; then
    mkdir ppi
else
    rm -rf ppi/*
fi
cd ppi

# pull STRING and GIANT-TN-c01 from GenePlexus data repository
echo -e "\nDownloading networks from GenePlexus..."
wget https://zenodo.org/record/3352348/files/Supervised-learning%20is%20an%20accurate%20method%20for%20network-based%20gene%20classification%20-%20Data.tar.gz

echo -e "\nUnpacking data..."
tar -xzf "Supervised-learning is an accurate method for network-based gene classification - Data.tar.gz" --strip-components=1

echo Moving GIANT-TN-c01
mv networks/GIANT-TN.edg ./GIANT-TN-c01.edg
echo Converting GIANT-TN-c01 to dense format
pecanpy --input GIANT-TN-c01.edg --output GIANT-TN-c01.npz --task todense

echo Moving STRING
mv networks/STRING.edg .
echo Converting STRING to dense format
pecanpy --input STRING.edg --output STRING.npz --task todense

rm -rf "Supervised-learning is an accurate method for network-based gene classification - Data.tar.gz" LICENSE.txt embeddings/ networks/ labels/

echo -e "\nDownloading full GIANT-TN network..."
wget http://giant.princeton.edu/static/networks/all_tissues.gz
gzip -d all_tissues.gz

echo Modifying edgelist by removing unwanted node class labels...
awk '{if (NF == 3) print $1"\t"$2"\t"$3; else print $1"\t"$2"\t"$4}' all_tissues > GIANT-TN.edg
echo Converting GIANT-TN to dense format
pecanpy --input GIANT-TN.edg --output GIANT-TN.npz --task todense

rm -rf all_tissues


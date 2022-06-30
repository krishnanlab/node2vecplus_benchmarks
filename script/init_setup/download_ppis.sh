#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

echo "Start processing PPI networks (the whole process should take at most half an hour)"

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/data/networks

mkdir -p ppi
cd ppi

# pull STRING from GenePlexus data repository
echo -e "\nDownloading networks from GenePlexus..."
wget https://zenodo.org/record/3352348/files/Supervised-learning%20is%20an%20accurate%20method%20for%20network-based%20gene%20classification%20-%20Data.tar.gz

echo -e "\nUnpacking data..."
tar -xzf "Supervised-learning is an accurate method for network-based gene classification - Data.tar.gz" --strip-components=1

echo Moving STRING
mv networks/STRING.edg .
echo Converting STRING to dense format
pecanpy --input STRING.edg --weighted --output STRING.npz --task todense

rm -rf "Supervised-learning is an accurate method for network-based gene classification - Data.tar.gz" LICENSE.txt embeddings/ networks/ labels/

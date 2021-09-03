#!/bin/bash --login

echo Start processing PPI networks

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/data/networks

if [ ! -d "ppi" ]; then
    mkdir ppi
fi
cd ppi

# pull STRING and GIANT-TN-c01 from GenePlexus data repository
echo -e "\nDownloading networks from GenePlexus..."
wget https://zenodo.org/record/3352348/files/Supervised-learning%20is%20an%20accurate%20method%20for%20network-based%20gene%20classification%20-%20Data.tar.gz

echo -e "\nUnpacking data..."
tar -xzf "Supervised-learning is an accurate method for network-based gene classification - Data.tar.gz" --strip-components=1
echo Moving GIANT-TN-c01
mv networks/GIANT-TN.edg ./GIANT-TN-c01.edg
echo Moving STRING
mv networks/STRING.edg .
rm -rf "Supervised-learning is an accurate method for network-based gene classification - Data.tar.gz" LICENSE.txt embeddings/ networks/ labels/

echo
echo Downloading full GIANT-TN network...
wget http://giant.princeton.edu/static/networks/all_tissues.gz
gzip -d all_tissues.gz
echo Modifying edgelist by removing unwanted node class labels...
awk '{if (NF == 3) print $1"\t"$2"\t"$3; else print $1"\t"$2"\t"$4}' all_tissues > GIANT-TN.edg
rm -rf all_tissues


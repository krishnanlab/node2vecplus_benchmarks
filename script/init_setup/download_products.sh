#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/data/networks

if [[ ! -d "product" ]]; then
    mkdir product
else
    rm -rf product/*
fi

mkdir product/downloads
cd product/downloads

echo WARNING: start downloading Amazon product data, this will take up to an hour

# download and unzip product metadata
wget http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
gzip -d metadata.json.gz

# download ratings data and convert to edge list (reviewer-product)
wget http://snap.stanford.edu/data/amazon/productGraph/item_dedup.csv
cat item_dedup.csv | awk -F "," '{print $1"\t"$2}' > item_dedup.edg

# process data to get the product graph with product categories
cd $home_dir/script/init_setup
product_data_dir=$home_dir/data/networks/product
edgelst_fp=$product_data_dir/downloads/item_dedup.edg
metadata_fp=$product_data_dir/downloads/metadata.json
graph_output_fp=$product_data_dir/Product.edg
label_output_fp=$home_dir/data/labels/Product.tsv

echo WARNING: start constructing the Amazon product co-review graph, this will take several horus

python process_products.py $edgelst_fp $metadata_fp --graph_output_fp $graph_output_fp \
    --label_output_fp $label_output_fp > $product_data_dir/category_info.txt

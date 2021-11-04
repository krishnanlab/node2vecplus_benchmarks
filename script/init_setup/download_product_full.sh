#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/data/networks

if [[ ! -d "product_full" ]]; then
    mkdir product_full
fi

mkdir product_full/downloads
cd product_full/downloads

echo WARNING: start downloading Amazon product data, this will take up to an hour

# download and unzip product metadata
wget http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
gzip -d metadata.json.gz

# download ratings data and convert to edge list (reviewer-product)
wget http://snap.stanford.edu/data/amazon/productGraph/item_dedup.csv
cat item_dedup.csv | awk -F "," '{print $1"\t"$2}' > item_dedup.edg

# process data to get the product graph with product categories
cd $home_dir/script/init_setup
product_data_dir=$home_dir/data/networks/product_full
edgelst_fp=$product_data_dir/downloads/item_dedup.edg
metadata_fp=$product_data_dir/downloads/metadata.json
graph_output_fp=$product_data_dir/ProductFull.edg
csr_output_fp=$product_data_dir/ProductFull.csr.npz
label_output_fp=$home_dir/data/labels/ProductFull.tsv

echo WARNING: start constructing the Amazon product co-review graph, this will take several horus

python process_product_full.py $edgelst_fp $metadata_fp --graph_output_fp $graph_output_fp \
    --label_output_fp $label_output_fp > $product_data_dir/category_info.txt

echo WARNING: start converting the ProductFull edge list to CSR, this will take ~1 hour
pecanpy --intput $graph_output_fp --output $csr_output_fp --weighted --task tocsr --workers 1

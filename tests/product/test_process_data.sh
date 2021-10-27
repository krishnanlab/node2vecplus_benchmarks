#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/script/init_setup

product_data_dir=$home_dir/tests/product
edgelst_fp=$product_data_dir/test_item_dedup.edg
metadata_fp=$product_data_dir/test_metadata.json
graph_output_fp=$product_data_dir/test_product.edg
label_output_fp=$product_data_dir/test_product.tsv

python process_products.py $edgelst_fp $metadata_fp --graph_output_fp $graph_output_fp \
    --label_output_fp $label_output_fp > $product_data_dir/category_info.txt

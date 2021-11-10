#!/bin/bash --login

source ~/.bashrc
conda activate node2vecplus-bench

for fn in review_product.edg product_categories.tsv; do
    if [[ -f $fn ]]; then
        echo $fn already exists, removing...
        rm -f $fn
    fi
done

home_dir=$(dirname $(dirname $(dirname $(realpath $0))))
echo home_dir=$home_dir
cd $home_dir/data/networks

if [[ ! -d "product_lite" ]]; then
    mkdir product_lite
fi

mkdir product_lite/downloads
cd product_lite/downloads

echo Cleaning up old reviews files
rm -rf reviews_*.json

echo WARNING: start downloading Amazon 5-core product review data, this will take couple minuts

# download 5-core product reviews from the top ten categories (exlucinding books)
product_id=1
for fn in \
reviews_Electronics_5.json \
reviews_Movies_and_TV_5.json \
reviews_CDs_and_Vinyl_5.json \
reviews_Clothing_Shoes_and_Jewelry_5.json \
reviews_Home_and_Kitchen_5.json \
reviews_Kindle_Store_5.json \
reviews_Sports_and_Outdoors_5.json \
reviews_Cell_Phones_and_Accessories_5.json \
reviews_Health_and_Personal_Care_5.json \
reviews_Toys_and_Games_5.json; do
echo $fn
    address="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/${fn}.gz"
    wget $address
    gunzip ${fn}.gz

    # extract first two columns: reviewer, product
    cat $fn | sed 's/"//g' | awk -F"[,: ]+" '{print $2"\t"$4}' > tmp.edg
    cat tmp.edg >> review_product.edg

    # write product label index
    cat tmp.edg | awk -F "\t" '{print $2}' | sort -u | awk -v a=$product_id '{print $1"\t"a}' >> product_categories.tsv
    (( product_id += 1 ))

    rm -f $fn tmp.edg
done

# move back to the script directory and process the product graph
cd $home_dir/script/init_setup
product_data_dir=$home_dir/data/networks/product_lite
input_edg_fp=$product_data_dir/downloads/review_product.edg
input_label_fp=$product_data_dir/downloads/product_categories.tsv
output_csr_fp=$product_data_dir/ProductLite.csr.npz
output_label_fp=$home_dir/data/labels/ProductLite.tsv

echo WARNING: start cosntructing the Amazon product co-review graph, this will take ~20 minuts


python process_product_lite.py --input_edg_fp $input_edg_fp --output_csr_fp $output_csr_fp \
    --input_label_fp $input_label_fp --output_label_fp $output_label_fp

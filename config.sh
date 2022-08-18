#!/bin/sh --login

trap "return 1" SIGINT

# Specify versions
ENV_NAME=node2vecplus-bench
PYTHON_VERSION=3.8
PYTORCH_VERSION=1.11.0
PYG_VERSION=2.0.0
CUDA_VERSION=10.2
PYG_WHL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu102.html"

# Check configuration option and execute
if [ -z $1 ]
then
    echo "ERROR: please provide option for configuration [setup,cleanup,download_ppis]"
    return 1
fi

case $1 in
    setup)
        setup
        ;;
    cleanup)
        cleanup
        ;;
    download_ppis)
        download_ppis
        ;;
    *)
        echo "ERROR: unknown option ${1}, please choose from [setup,cleanup,download_ppis]"
        return 1
        ;;
esac

# Set up conda environment
setup () {
    echo "Setting up conda environment ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    conda activate ${ENV_NAME} || return 1

    conda install pytorch=${PYTORCH_VERSION} torchvision cudatoolkit=${CUDA_VERSION} -c pytorch -y
    pip install torch-geometric==${PYG_VERSION} torch-scatter torch-sparse torch-cluster -f ${PYG_WHL_URL}

    pip install -r requirements.txt

    conda deactivate
    echo "Successfully install conda environment ${ENV_NAME}"
    echo "To activate this environment, use"
    echo -e "\n    \$ conda activate ${ENV_NAME}\n"
}

# Clean up conda environment
cleanup () {
    echo "Cleaning up conda environment ${ENV_NAME}"
    conda remove --name ${ENV_NAME} --all -y && conda clean --all -y
}

# Download gene interaction network data from Dropbox
download_ppis () {
    echo "Start downloading gene interaciton network data from Dropbox"
    curl -L -o node2vecplus_bench_ppis.tar.gz https://www.dropbox.com/s/aettebq5lbgu1cu/node2vecplus_bench_ppis-v1.0.0.tar.gz?dl=1

    echo "Finshed downloading, start extracting, this might take a while..."
    tar -xzvf node2vecplus_bench_ppis.tar.gz --transform 's/node2vecplus_bench_ppis/ppi/' --directory data/networks && rm -f node2vecplus_bench_ppis.tar.gz
}

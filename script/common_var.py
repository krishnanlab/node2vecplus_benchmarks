import pathlib

HOME_DIR = pathlib.Path(__file__).absolute().parents[1]
DATA_DIR = HOME_DIR / "data"
RESULT_DIR = HOME_DIR / "result"
NETWORK_DIR = DATA_DIR / "networks"

LABEL_PATH_DICT = {
    "DisGeNet": "/mnt/research/compbio/krishnanlab/data/disease-gene_annotations/disgenet/disgenet_disease-genes_prop.gsea-min10-max600-ovlppt7-jacpt5.nonred.gmt",
    "GOBP": "/mnt/research/compbio/krishnanlab/data/functional_annotations/go/go_bp-genes_exp-ec_prop_pos-slim.gsea-min10-max200-ovlppt7-jacpt5.nonred.gmt",
    "GOBP-tissue": "/mnt/research/compbio/krishnanlab/data/functional_annotations/go/go_tissue/GOBP-tissue-subset.gmt",
}
PUBMED_COUNT_PATH = "/mnt/research/compbio/krishnanlab/data/pubmed/gene2pubmed_human_gene-counts.txt"

REPETITION = 10

W2V_NUMWALKS = 10
W2V_WALKLENGTH = 80
W2V_WINDOW = 10
W2V_EPOCHS = 1

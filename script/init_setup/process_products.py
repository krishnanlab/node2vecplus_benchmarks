import argparse
import logging
import json
from time import time

import networkx as nx

from pecanpy.graph import SparseGraph


class timeit:
    """
    Warpper for timing function calls
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        def timed_func(*args, **kwargs):
            t = time()
            result = func(*args, **kwargs)
            t = time() - t

            t_hr = int(t // 3600)
            t_min = int(t % 3600 // 60)
            t_sec = t % 60
            t_str = f'{t_hr:02d}:{t_min:02d}:{t_sec:05.2f}'

            logging.info(f'Took {t_str} to {self.name}')

            return result

        return timed_func


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process the product-reviewer graph and convert into a '
                    'product-prodcut graph with co-review relations. The '
                    'largest connected component is used. The node labels '
                    'correspond to the top level product categories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('edglst_fp',
        help='Path to the product-review edge list file')

    parser.add_argument('metadata_fp', help='Path to the product metadata file')

    parser.add_argument('--graph_output_fp', default=None,
                        help='Output path for the product graph')

    parser.add_argument('--label_output_fp', default=None,
                        help='Output path for the labels')

    args = parser.parse_args()
    print(args)

    return args


@timeit('get product categories')
def get_product_categoeis(fp):
    """
    Extract product top category information and return the dictionary mapping
    from product ``asin`` to its corresponding top category.
    """
    product_category_dict = {}
    with open (fp, 'r') as f:
        for line in f:
            line = line.strip().replace('\'', '\"') # make compatible with JSON

            try:
                terms = json.loads(line)
                product = terms['asin']
                top_category = terms['categories'][0][0]
                product_category_dict[product] = top_category
            except Exception:
                continue

    return product_category_dict


@timeit('construct product-review graph')
def get_product_review_graph(fp):
    """
    Load and return the product-review edge list as a graph.
    """
    g = SparseGraph()
    g.read_edg(fp, False, False, False)
    return g


@timeit('construct product-product coreview graph')
def get_product_product_graph(g, product_category_dict):
    """
    Convert the product-review graph into a product product co-review graph.
    """
    g_product_raw = nx.Graph()
    n = len(g.IDlst)

    for i in range(n - 1):
        id1 = g.IDlst[i]
        if id1 not in product_category_dict:
            continue
        reviewer_set1 = set(g.data[i])

        for j in range(i + 1, n):
            id2 = g.IDlst[j]
            if id2 not in product_category_dict:
                continue
            reviewer_set2 = set(g.data[j])

            coreview = len(reviewer_set1 & reviewer_set2)

            if coreview > 0:
                g_product_raw.add_edge(id1, id2, weight=coreview)

    logging.info(f'Finished constructing the prodcut-prodcut graph, number '
                 f'of produts = {g_product_raw.number_of_nodes()}')

    lcc = max(nx.connected_components(g_product_raw), key=len)
    g_product_lcc = g_product_raw.subgraph(lcc)

    logging.info(f'Extracted largest connected component: number of nodes = '
                 f'{g_product_lcc.number_of_nodes()}, number of edges = '
                 f'{g_product_lcc.number_of_edges()}')

    return g_product_lcc


@timeit('save the product graph')
def save_graph(g, output_fp):
    """
    Save the product graph as an edge list file with coreview count.
    """
    if output_fp is None:
        logging.warning(f'Output disabled, did not save product graph')
    else:
        nx.write_edgelist(g, output_fp, data=['weight'])


@timeit('save the product category labels')
def save_label(nodes, category_dict, output_fp):
    """
    Save product category information as index.

    Product category information will be printed, along with the number of
    products per category.
    """
    if output_fp is None:
        logging.warning(f'Output disabled, did not save product graph')
        return

    category_count = {}
    category_idx_map = {}

    with open(output_fp, 'w') as f:
        for product in nodes:
            category = category_dict[product]

            if category not in category_count:
                category_count[category] = 0
                category_idx_map[category] = len(category_count)

            category_count[category] += 1
            category_idx = category_idx_map[category]

            f.write(f'{product}\t{category_idx}\n')

    print("Category index mapping:")
    for category, idx in category_idx_map.items():
        print(f'\t{idx:02d}: {category} (n = {category_count[category]})')


@timeit('run the full processing')
def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    g = get_product_review_graph(args.edglst_fp)
    product_category_dict = get_product_categoeis(args.metadata_fp)
    g_product = get_product_product_graph(g, product_category_dict)

    save_graph(g_product, args.graph_output_fp)
    save_label(g_product.nodes, product_category_dict, args.label_output_fp)


if __name__ == '__main__':
    main()

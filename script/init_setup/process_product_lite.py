"""
Preprocess the Amazon coreview graph.

Key steps:
- Load the product-reviewer graph (edge list), which is downloaded and
    processed by the download_product_5.sh sciprt
- Construct the product co-review graph where the edge weights between two
    products are the number of reviewers they share
- Extract the largest connected component from the co-review product graph and
    save as CSR to be used by PecanPy
- Save labels (same as the input but only for products in the LCC above)

Test command:
```bash
$ python process_products.py --input_edg_fp review_product.edg \
$     --output_csr_fp test_ProductLite.csr.npz \
$     --input_label_fp product_categories.tsv \
$     --output_label_fp test_ProductLite.tsv
```

"""

import argparse
import logging
import json
from time import time

from pecanpy.graph import SparseGraph


class timeit:
    """Warpper for timing function calls."""

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


class Graph:
    """A lite weight graph object."""

    def __init__(self):
        self.data = []
        self.IDlst = []
        self.IDmap = {}
        self._number_of_nodes = 0
        self._number_of_edges = 0

    @property
    def nodes(self):
        return self.IDlst.copy()

    @property
    def number_of_nodes(self):
        return self._number_of_nodes

    @property
    def number_of_edges(self):
        return self._number_of_edges

    def get_node_idx(self, node):
        if node not in self.IDmap:
            self.IDmap[node] = self.number_of_nodes
            self.IDlst.append(node)
            self.data.append({})
            self._number_of_nodes += 1

        return self.IDmap[node]

    def add_edge(self, node1, node2, weight):
        idx1 = self.get_node_idx(node1)
        idx2 = self.get_node_idx(node2)
        self.data[idx1][idx2] = self.data[idx2][idx1] = weight
        self._number_of_edges += 1

    def save(self, output_edg_fp, subgraph=None):
        """
        Save (sub)graph as an edge list with weights.

        Args:
            output_edg_fp: output edge list file path.
            subgraph: Optional, node indices to be output to the edge list.

        """
        if output_edg_fp is not None:
            subgraph = set(subgraph) if subgraph is not None else \
                       set(range(self.number_of_nodes))

            with open(output_edg_fp, 'w') as f:
                for idx1 in range(self.number_of_nodes):
                    if idx1 not in subgraph:
                        continue

                    node1 = self.IDlst[idx1]
                    for idx2, weight in self.data[idx1].items():
                        if idx2 not in subgraph:
                            continue

                        node2 = self.IDlst[idx2]
                        f.write(f'{node1}\t{node2}\t{weight}\n')

    @timeit('extract the largest connected component')
    def get_lcc(self):
        lcc = max(self.get_connected_components(), key=len)
        logging.info(f'Extracted largest connected component: '
                     f'number of nodes = {len(lcc)}')
        return sorted(lcc)

    def get_connected_components(self):
        """Find connected components via BFS search."""
        unvisited = set(range(self.number_of_nodes))
        components = []

        while unvisited:
            seed_node = next(iter(unvisited))
            next_level_nodes = [seed_node]
            component_membership = []

            while next_level_nodes:
                curr_level_nodes = next_level_nodes[:]
                next_level_nodes = []

                for node in curr_level_nodes:
                    if node in unvisited:
                        for nbr in self.data[node]:
                            if nbr in unvisited:
                                next_level_nodes.append(nbr)
                        component_membership.append(node)
                        unvisited.remove(node)
                        logging.debug(f'Number of unvisited = {len(unvisited)}')

            components.append(component_membership)

        return components


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process the product-reviewer graph and convert into a '
                    'product-prodcut graph with co-review relations. The '
                    'largest connected component is used. The node labels '
                    'correspond to the top level product categories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_edg_fp', required=True,
        help='Path to the product-review edge list file')

    parser.add_argument('--output_csr_fp', required=True,
        help='Output path for the product graph in CSR format')

    parser.add_argument('--input_label_fp', required=True,
        help='File path to the product labels')

    parser.add_argument('--output_label_fp', required=True,
        help='Output path to the product labels for the final product graph')

    parser.add_argument('--logging', default='info', help='Logging level')

    return parser.parse_args()


@timeit('get product categories')
def get_product_categories(fp):
    """Load product labels from a tsv file."""
    product_category_dict = {}
    with open (fp, 'r') as f:
        for line in f:
            product, category = line.strip().split('\t')
            product_category_dict[product] = int(category)

    return product_category_dict


@timeit('construct product-review graph')
def get_product_review_graph(fp):
    """Load and return the product-review edge list as a graph."""
    g = SparseGraph()
    g.read_edg(fp, False, False, False)
    return g


@timeit('construct product-product coreview graph')
def get_product_product_graph(g, product_category_dict):
    """Convert the product-review graph into a product co-review graph."""
    g_product = Graph()
    n = len(g.IDlst)

    for i in range(n - 1):
        id1 = g.IDlst[i]
        # only process prodcut, leave reviewers out
        if id1 not in product_category_dict:
            continue
        reviewer_set1 = set(g.data[i])

        # bfs for finding relevant products to reduce runtime
        product_set = set()
        for reviewer_idx in reviewer_set1:
            new_products = [j for j in g.data[reviewer_idx] if j > i]
            product_set.update(new_products)
        if not product_set:  # no new neighbor to add
            continue

        logging.debug(f'Working on {i + 1} of {n} products, '
                      f'{len(product_set)} new neighbors')

        for j in product_set:
            id2 = g.IDlst[j]
            # only process prodcut, leave reviewers out
            if id2 not in product_category_dict:
                continue
            reviewer_set2 = set(g.data[j])

            coreview = len(reviewer_set1 & reviewer_set2)
            if coreview > 0:
                g_product.add_edge(id1, id2, coreview)

    logging.info(f'Finished constructing the prodcut-prodcut graph, number '
                 f'of produts = {g_product.number_of_nodes}')

    return g_product


def save_csr(g_product, subgraph, subgraph_ids, output_csr_fp):
    """
    Save CSR graph as .csr.npz to be loaded directly by PecanPy.

    Args:
        g_product: lite graph object defined earlier in this script
        subgraph: node index list for the largest connected component
        subgraph_ids: node ids for the largest connected component
        output_csr_fp: output filepath for the CSR graph
    """
    subgraph_idxmap = {j: i for i, j in enumerate(subgraph)}
    data = []
    for i in subgraph_idxmap:
        data.append({})
        for j, w in g_product.data[i].items():
            if j in subgraph_idxmap:
                data[-1][subgraph_idxmap[j]] = w

    g = SparseGraph()
    g.data = data
    g.set_ids(subgraph_ids)
    g.to_csr()
    g.save(output_csr_fp)


def save_label(products, product_category_dict, output_label_fp):
    """
    Save product category information as index.

    Product category information will be printed, along with the number of
    products per category.

    Args:
        products: list of products in the final prodcut graph
        product_category_dict: full product-category mapping
        output_label_fp: output path to the final product mapping

    """
    with open(output_label_fp, 'w') as f:
        for product in sorted(products, key=product_category_dict.get):
            category = product_category_dict[product]
            f.write(f'{product}\t{category}\n')


@timeit('run the full processing')
def main():
    args = parse_args()
    logging_level = getattr(logging, args.logging.upper())
    logging.basicConfig(level=logging_level)
    logging.info(args)

    # load product review bipartite graph and the product categories
    g = get_product_review_graph(args.input_edg_fp)
    product_category_dict = get_product_categories(args.input_label_fp)

    # construct product co-review and extract the lagest connected component
    g_product = get_product_product_graph(g, product_category_dict)
    subgraph = g_product.get_lcc()
    subgraph_ids = [g_product.IDlst[i] for i in subgraph]

    # save largest connected component of the product graph
    save_csr(g_product, subgraph, subgraph_ids, args.output_csr_fp)  # as csr

    # save labels for products in the lagest connected component
    save_label(subgraph_ids, product_category_dict, args.output_label_fp)


if __name__ == '__main__':
    main()

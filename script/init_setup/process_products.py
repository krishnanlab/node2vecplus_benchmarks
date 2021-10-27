import argparse
import logging
import json
from time import time

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


class Graph:
    """
    A lite weight graph object.
    """
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

    def save(self, outpath, subgraph=None):
        """
        Save (sub)graph as an edge list with weights.

        Args:
            outpath: output edge list file path.
            subgraph: Optional, node indices to be output to the edge list.
        """
        if outpath is not None:
            subgraph = set(subgraph) if subgraph is not None else \
                       set(range(self.number_of_nodes))

            with open(outpath, 'w') as f:
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
                     f'number of nodes = {self.number_of_nodes}')
        return lcc

    def get_connected_components(self):
        """
        Find connected components via BFS search.
        """
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

            components.append(component_membership)

        return components


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
    logging.info(args)

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
    g_product_raw = Graph()
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
                g_product_raw.add_edge(id1, id2, coreview)

    logging.info(f'Finished constructing the prodcut-prodcut graph, number '
                 f'of produts = {g_product_raw.number_of_nodes}')

    #return g_product_lcc
    return g_product_raw


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

    print('#index\tcategory\tcaount')
    for category, idx in category_idx_map.items():
        print(f'{idx}\t{category}\t{category_count[category]}')


@timeit('run the full processing')
def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    g = get_product_review_graph(args.edglst_fp)
    product_category_dict = get_product_categoeis(args.metadata_fp)
    g_product = get_product_product_graph(g, product_category_dict)

    subgraph = g_product.get_lcc()
    subgraph_ids = [g_product.IDlst[i] for i in subgraph]
    g_product.save(args.graph_output_fp, subgraph)
    save_label(subgraph_ids, product_category_dict, args.label_output_fp)


if __name__ == '__main__':
    main()

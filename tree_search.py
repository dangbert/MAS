import networkx as nx
import matplotlib.pyplot as plt


def create_tree(depth):
    """
    Create binary tree of given depth.
    Nodes are named with ints incrementing from 1 up.
    """
    assert depth >= 1
    G = nx.Graph()
    # we can store whatever attributes we want (e.g. "address")
    G.add_node(1, address="")
    last_node = 1  # name of last node created

    for d in range(1, depth):
        prev_row_size = 2 ** (d - 1)
        prev_row_start = last_node - prev_row_size + 1

        prev_row = list(range(prev_row_start, last_node + 1))
        for n in prev_row:  # add 2 children to each node in previous row
            G.add_node(last_node + 1, address=G.nodes[n]["address"] + "L")
            G.add_node(last_node + 2, address=G.nodes[n]["address"] + "R")
            G.add_edge(n, last_node + 1)
            G.add_edge(n, last_node + 2)
            last_node += 2
            # nx.draw(G, with_labels=True, node_size=300)
            # plt.show()
    return G


def edit_distance(add1: str, add2: str) -> int:
    assert len(add1) == len(add2)
    return sum([int(a1 != a2) for a1, a2 in zip(add1, add2)])


if __name__ == "__main__":
    depth = 4
    tree = create_tree(4)
    # print(tree)
    print("\ntree:")
    print(tree.edges())
    print(tree.nodes())

    nx.draw(tree, with_labels=True, node_size=300)
    plt.show()

    first_leaf_node = tree.number_of_nodes() - 2 ** (depth - 1) + 1
    leaf_node_names = list(range(first_leaf_node, tree.number_of_nodes() + 1))

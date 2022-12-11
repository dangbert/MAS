import networkx as nx
import matplotlib.pyplot as plt


def create_tree(depth):
    """Create binary tree of given depth."""
    assert depth >= 1
    G = nx.Graph()

    G.add_node(1)
    last_node = 1
    # prev_start, prev_count = (1, 1)  # first node name, num nodes (in prev row)
    for d in range(1, depth):

        prev_row_size = 2 ** (d - 1)
        prev_row_start = last_node - prev_row_size + 1
        # total_nodes = 1 + 2**d

        # add 2 children to each node in previous row
        prev_row = list(range(prev_row_start, last_node + 1))
        # print(f"\n d={d}, prev row")
        # print(prev_row)
        # for n in prev_row:
        for n in prev_row:
            G.add_edge(n, last_node + 1)
            G.add_edge(n, last_node + 2)
            last_node += 2
            # nx.draw(G, with_labels=True, node_size=300)
            # plt.show()

        # prev_count = last_node - (prev_start + prev_count + 1)
        # prev_start = last_node
    return G


def edit_distance(add1: str, add2: str) -> int:
    assert len(add1) == len(add2)
    return sum([int(a1 != a2) for a1, a2 in zip(add1, add2)])


if __name__ == "__main__":
    tree = create_tree(4)
    # print(tree)
    print("\ntree:")
    print(tree.edges())
    print(tree.nodes())

    nx.draw(tree, with_labels=True, node_size=300)
    plt.show()

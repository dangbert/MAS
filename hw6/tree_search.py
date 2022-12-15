from copy import deepcopy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import List

STATS_TEMPLATE = {"vals": []}


class MCTS:
    tree: nx.Graph
    c: float
    B: int
    target_name: int

    def __init__(self, depth: int, c=2, B: float = 25, draw: bool = False):
        """
        Init class with a binary tree of given depth.
        Nodes are named with ints incrementing from 1 up.

        :param depth: depth of tree to create
        :param B: param for computing values of leaf ndoes
        :param c: hyperparam for UCB calculation
        """
        self.c = c
        self.B = B

        self.reset(depth, draw=draw)

    def reset(self, depth: int, draw: bool = False):
        assert depth >= 1
        tree = nx.Graph()
        # we can store whatever attributes we want (e.g. "address")
        tree.add_node(1, address="")
        last_node = 1  # name of last node created

        for d in range(1, depth):
            prev_row_size = 2 ** (d - 1)
            prev_row_start = last_node - prev_row_size + 1

            prev_row = list(range(prev_row_start, last_node + 1))
            for n in prev_row:  # add 2 children to each node in previous row
                tree.add_node(last_node + 1, address=tree.nodes[n]["address"] + "L")
                tree.add_node(last_node + 2, address=tree.nodes[n]["address"] + "R")
                tree.add_edge(n, last_node + 1)
                tree.add_edge(n, last_node + 2)
                last_node += 2
        if draw:
            nx.draw(tree, with_labels=True, node_size=300)
            plt.show()

        # now pick a target (leaf node) and assign values to all leaf nodes
        first_leaf_node = tree.number_of_nodes() - 2 ** (depth - 1) + 1
        leaf_node_names = list(range(first_leaf_node, tree.number_of_nodes() + 1))

        self.target_name = random.choice(leaf_node_names)
        dists = [
            MCTS.edit_distance(
                tree.nodes[n]["address"], tree.nodes[self.target_name]["address"]
            )
            for n in leaf_node_names
        ]
        dmax = max(dists)
        print(
            f"target node = #{self.target_name}, num leaf nodes = {len(leaf_node_names)}, max distance: {dmax}, min distance: {min(dists)}"
        )
        # compute values for each leaf node based on distance from target node
        for idx, n in enumerate(leaf_node_names):
            tree.nodes[n]["value"] = (
                self.B * math.pow(math.e, (-5 * dists[idx] / dmax)) + np.random.normal()
            )

        leaf_vals = [tree.nodes[n]["value"] for n in leaf_node_names]

        plt.xlabel("leaf node name")
        plt.ylabel("node value")
        plt.title(
            f"Distribution of Leaf Node Values (target node = {self.target_name})"
        )
        plt.plot(leaf_node_names, leaf_vals)
        if draw:
            plt.show()

        self.tree = tree

    def run(self):
        """
        RUN Monte Carlo Tree search algorithm.
        TODO: add budget params
        """
        cur_root = 1
        # init stats for node
        self.tree.nodes[cur_root]["stats"] = deepcopy(STATS_TEMPLATE)

    def expansion(self, root_node: int):
        """Explores a new child of root_node?"""
        children = self.get_child_names(root_node)
        unex_children = [c for c in children if "stats" not in self.tree.nodes[c]]

        if len(unex_children) == 0:
            return
        cur = random.choice(unex_children)

        self.tree.nodes[cur]["stats"] = deepcopy(STATS_TEMPLATE)
        self.tree.nodes[cur]["stats"]["vals"].append(self.simulate(cur))

    def selection(self, root_node: int) -> int:
        """
        Suggests which child of a given node should be returned.
        Selection is based on 2 things: how good are the stats, and how much a child node has been ignored.
        """
        pass

    def update(self, root_node: int):
        """"""
        pass

    def simulate(self, node_name: int) -> float:
        """Simulate random search from given node and return the leaf value it reaches."""
        child_nodes = self.get_child_names(node_name)
        if len(child_nodes) == 0:
            return self.tree.nodes[node_name]["value"]
        # continue random search
        return self.simulate(random.choice(child_nodes))

    @staticmethod
    def edit_distance(add1: str, add2: str) -> int:
        assert len(add1) == len(add2)
        return sum([int(a1 != a2) for a1, a2 in zip(add1, add2)])

    def get_child_names(self, node_name: int) -> List[int]:
        """Returns list of names of child nodes of the provided node e.g. [2, 3]"""
        node_names = [name for _, name in self.tree.edges(node_name)]
        return list(filter(lambda n: n > node_name, node_names))


if __name__ == "__main__":
    depth = 4
    mc = MCTS(depth, draw=True)
    res = mc.simulate(1)
    import pdb

    pdb.set_trace()

    exit(0)
    tree = mc.tree
    # print(tree)
    print("\ntree:")
    print(tree.edges())
    print(tree.nodes())

    nx.draw(tree, with_labels=True, node_size=300)
    plt.show()

    first_leaf_node = tree.number_of_nodes() - 2 ** (depth - 1) + 1
    leaf_node_names = list(range(first_leaf_node, tree.number_of_nodes() + 1))

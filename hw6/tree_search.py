from copy import deepcopy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import List, Union, Optional
import pdb

STATS_TEMPLATE = {"vals": []}

# max number of iterations starting at a given root node
MAX_ITERATIONS = 20

# max number of rollouts from a given "snowcap" leaf node
MAX_ROLLOUTS = 5

EXPANSION_PROB = 0.3


class MCTS:
    """
    Performs monte carlo tree search on a binary tree with varying values stored in the leaf nodes.
    Based on p. 185 in Textbook.
    """

    tree: nx.Graph
    c: float
    B: int
    target_name: int
    global_root: int  # name of tree's root node

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
        self.global_root = last_node

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
        # compute values for each leaf node based on distance from target node
        for idx, n in enumerate(leaf_node_names):
            tree.nodes[n]["value"] = (
                self.B * math.pow(math.e, (-5 * dists[idx] / dmax)) + np.random.normal()
            )

        leaf_vals = [tree.nodes[n]["value"] for n in leaf_node_names]
        target_value = tree.nodes[self.target_name]["value"]
        print(
            f"target node = #{self.target_name} (value {target_value:.3f}), num leaf nodes = {len(leaf_node_names)}, max distance: {dmax}, min distance: {min(dists)}"
        )

        plt.xlabel("leaf node name")
        plt.ylabel("node value")
        plt.title(
            f"Distribution of Leaf Node Values (target node = {self.target_name})"
        )
        plt.plot(leaf_node_names, leaf_vals)
        if draw:
            plt.show()

        self.tree = tree

    def run(self) -> List[int]:
        """
        Runs the Monte Carlo Tree search algorithm.
        Returns a trajectory (list of node names) from the root of the tree to a leaf node.
        Note that we identify nodes in our search (sub)tree by the presence of "stats".
        """
        trajectory = []
        cur_root = self.global_root

        # continue until cur_root is a leaf node
        while len(self.get_child_names(cur_root)) > 1:
            # print(f"\nreached root {cur_root}")
            trajectory.append(cur_root)

            # init stats for node
            self.tree.nodes[cur_root]["stats"] = deepcopy(STATS_TEMPLATE)

            for i in range(MAX_ITERATIONS):
                print(f"\nroot {cur_root}, iteration {i}")
                node = self.selection(cur_root)
                expand = random.random() <= EXPANSION_PROB or node is None
                if node is None:
                    # initial condition
                    expand = True
                    node = cur_root

                # on some iterations, the tree is expanded from the selected leaf node
                #   by adding one or more child nodes reached from the selected node via unexplored actions.
                if expand:
                    print(f"expanding from node {node}")
                    node = self.expansion(node)

                print(f"simulating from node {node}...")
                assert node is not None
                # simulate random searches from node until a leaf node is reached
                vals = [self.simulate(node) for _ in range(MAX_ROLLOUTS)]

                print("updating stats...")
                # update stats starting at node (backing up all the way to cur_root)
                self.update(node, vals, cur_root)

            # after computation budget expanded, make a final move to descend to child with highest number of simulations
            #   this can be based on the "action having the largest action value", or the most visited node (to avoid outliers).
            rel_children = [
                self.tree.nodes[n]
                for n in self.get_child_names(cur_root)
                if "stats" in self.tree.nodes[n]
            ]
            cur_root = max(
                rel_children,
                key=lambda cn: len(cn["stats"]["vals"]),
            )

    def selection(self, root_node: int) -> Optional[int]:
        """
        Starts at root_node, applies a 'tree policy' to select a "snowcap" leaf node.

        Suggests which child of a given node should be returned.
        Selection is based on 2 things: how good are the stats, and how much a child node has been ignored.
        """
        children = self.get_child_names(root_node)
        ex_children = [c for c in children if "stats" in self.tree.nodes[c]]
        if len(ex_children) == 0:
            return None
        return random.choice(ex_children)  # for now

        # TODO: figure out how to implement and run experiment
        # probably need to recurse?

    def expansion(self, node: int) -> Optional[int]:
        """Expands the search tree, from the given "snowcap" leaf node, identifying a new child to explore.
        If all children have been explored, returns node as backup.
        ^TODO: ensure selection() traverses recursively to depth of "search tree" so node is snowcap leaf
        """
        children = self.get_child_names(node)
        unex_children = [c for c in children if "stats" not in self.tree.nodes[c]]
        if len(unex_children) == 0:
            return node

        cur = random.choice(unex_children)
        self.tree.nodes[cur]["stats"] = deepcopy(STATS_TEMPLATE)
        return cur
        # self.update(cur, self.simulate(cur))

    def simulate(self, node_name: int) -> float:
        """Simulate random search from given node and return the leaf value it reaches."""
        print(f"in simulate node_name = {node_name}")
        assert type(node_name) == int
        child_nodes = self.get_child_names(node_name)
        if len(child_nodes) == 0:
            return self.tree.nodes[node_name]["value"]
        # continue random search
        return self.simulate(random.choice(child_nodes))

    def update(self, node: int, values: List[float], cur_root: int) -> None:
        """
        Updates the stats of a given node (and its parents as well) using a given simulation result.
        Stops updating when cur_root is reached.
        """
        print(f"updating node: {node}")
        try:
            self.tree.nodes[node]["stats"]["vals"].extend(values)
        except KeyError as err:
            import pdb

            pdb.set_trace()
            print(err)
        parent = self.get_parent_name(node)
        if parent is not None and parent != cur_root:
            # we'll just add the results to the parent as well
            self.update(parent, values, cur_root)

    def get_child_names(self, node_name: int) -> List[int]:
        """Returns list of names of child nodes of the provided node e.g. [2, 3]"""
        node_names = [name for _, name in self.tree.edges(node_name)]
        print("node_name = ")
        print(node_name)
        print("node_names = ")
        print(node_names)
        return list(filter(lambda n: n > node_name, node_names))

    def get_parent_name(self, node_name: int) -> Union[int, None]:
        """Returns name of parent node (or None) if there's no parent."""
        node_names = [name for _, name in self.tree.edges(node_name)]
        parents = list(filter(lambda n: n < node_name, node_names))
        if len(parents) == 0:
            return None
        assert len(parents) == 1
        return parents[0]

    @staticmethod
    def edit_distance(add1: str, add2: str) -> int:
        assert len(add1) == len(add2)
        return sum([int(a1 != a2) for a1, a2 in zip(add1, add2)])


if __name__ == "__main__":
    depth = 4
    mc = MCTS(depth, draw=True)
    res = mc.simulate(1)

    tra = mc.run()
    print(f"returned trajectory: ")
    print(tra)
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


def avg(arr: List) -> float:
    return sum(arr) / len(arr)

#!/usr/bin/env python3
import argparse
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import List, Union, Optional
import logging
import pdb

logger = logging.getLogger(__name__)

# we simply store a list of (payoff) values for each node
#   (where the length tracks the number of visits)
STATS_TEMPLATE = {"vals": []}

# max number of iterations starting at a given root node
MAX_ITERATIONS = 20

# max number of rollouts from a given "snowcap" leaf node
MAX_ROLLOUTS = 5

EXPANSION_PROB = 0.5


def avg(arr: List) -> float:
    return sum(arr) / len(arr)


class MCTS:
    """
    Performs monte carlo tree search on a binary tree with varying values stored in the leaf nodes.
    Based on p. 185 in Textbook.

    In this case, our search tree exists from the start and never increases in size
    (but stats get stored in each node).
    In a more typical implementation, you'd build a tree from scratch as you traverse the space of possible actions from the initial state (node).

    Here a "snowcap" leaf node is not necessarily a terminal node in self.tree, but a leaf node in the search subtree of self.tree
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
        :param B: param for computing values of leaf (terminal) nodes
        :param c: hyperparam for UCB calculation
        """
        self.c = c
        self.B = B

        self.reset(depth, draw=draw)

    def reset(self, depth: int, draw: bool = False):
        assert depth >= 1
        logger.info(f"creating tree with depth {depth}")
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
            plt.savefig("graph.pdf", dpi=400)
            print("wrote graph.pdf")

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

        plt.clf()
        plt.xlabel("leaf node name")
        plt.ylabel("node value")
        plt.title(
            f"Distribution of Leaf Node Values (target node = {self.target_name})"
        )
        plt.plot(leaf_node_names, leaf_vals)
        if draw:
            plt.show()
        plt.savefig("leaves.pdf", dpi=400)
        print("wrote leaves.pdf")

        self.tree = tree

    def run(self) -> List[int]:
        """
        Runs the Monte Carlo Tree search algorithm.
        Returns a trajectory (list of node names) from the root of the tree to a leaf node.
        Note that we identify nodes in our search (sub)tree by the presence of "stats".
        """
        cur_root = self.global_root
        trajectory = [cur_root]

        # continue until cur_root is a leaf (terminal) node
        while len(self.get_child_names(cur_root)) > 1:
            # print(f"\nreached root {cur_root}")

            if "stats" not in self.tree.nodes[cur_root]:
                # init stats for node, mark as part of snowcap
                self.tree.nodes[cur_root]["stats"] = deepcopy(STATS_TEMPLATE)
                self.tree.nodes[cur_root]["snowcap"] = True

            for i in range(MAX_ITERATIONS):
                logger.debug(
                    f"root {cur_root}, iteration {i} (cur_root = {cur_root})\n"
                )
                node = self.selection(cur_root)
                expand = random.random() <= EXPANSION_PROB or node == cur_root
                if node is None:
                    # initial condition
                    expand = True
                    node = cur_root

                # on some iterations, the tree is expanded from the selected leaf node
                #   by adding one or more child nodes reached from the selected node via unexplored actions.
                if expand:
                    logger.debug(f"expanding from node {node}\n")
                    node = self.expansion(node)

                logger.debug(f"simulating from node {node}...\n")
                assert node is not None
                # simulate random searches from node until a leaf node is reached
                vals = [self.simulate(node) for _ in range(MAX_ROLLOUTS)]

                logger.debug("updating stats...")
                # update stats starting at node (backing up all the way to cur_root)
                self.update(node, vals, cur_root)

            # after computation budget expanded, make a final move to descend to a child.
            #   this can be based on the "action having the largest action value", or the most visited node (to avoid outliers).
            # TODO: could we use UCB here?
            rel_children = [
                n
                for n in self.get_child_names(cur_root)
                if "stats" in self.tree.nodes[n]
            ]
            nodes = self.tree.nodes
            cur_root = max(
                rel_children,
                key=lambda c: nodes[c]["value"] if "value" in nodes[c]
                # else len(nodes[c]["stats"]["vals"]),
                else avg(nodes[c]["stats"]["vals"]),
            )
            trajectory.append(cur_root)
            logger.info(
                f"moving to new root {cur_root} ({len(rel_children)} children considered), trajectory = {trajectory}"
            )
            print()
        return trajectory

    def selection(self, node: int) -> int:
        """
        Starts at root_node, applies a 'tree policy' to select a "snowcap" leaf node.

        Suggests which child of a given node should be returned.
        Selection is based on 2 things: how good are the stats, and how much a child node has been ignored.
        The UCB formula weights these factors.
        """
        children = self.get_child_names(node)
        # find relevant children (those that have been visited before)
        rel_children = [c for c in children if "snowcap" in self.tree.nodes[c]]
        if len(rel_children) == 0:
            return node  # if no children have been visited, then select root_node

        parent_visits = len(self.tree.nodes[node]["stats"]["vals"])
        ucb_scores = {}
        for c in rel_children:
            stats = self.tree.nodes[c]["stats"]
            ucb_scores[c] = avg(stats["vals"]) + self.c * math.sqrt(
                math.log(parent_visits) / len(stats["vals"])
            )

        return self.selection(max(rel_children, key=lambda c: ucb_scores[c]))

    def expansion(self, node: int) -> Optional[int]:
        """
        Expands the search tree, from the given "snowcap" leaf node, randomly picking a new child to explore.
        If all children have been explored, returns node as fallback.
        """
        children = self.get_child_names(node)
        unex_children = [c for c in children if "snowcap" not in self.tree.nodes[c]]
        if len(unex_children) == 0:
            logger.warn(f"unable to expand node {node} (children also in snowcap)")
            return node

        cur = random.choice(unex_children)
        self.tree.nodes[cur]["stats"] = deepcopy(STATS_TEMPLATE)
        return cur

    def simulate(self, node: int) -> float:
        """Simulate random search from given node and return the (terminal) leaf value it reaches."""
        # logger.debug(f"in simulate node_name = {node}")
        assert type(node) == int
        child_nodes = self.get_child_names(node)
        if len(child_nodes) == 0:
            return self.tree.nodes[node]["value"]
        # continue random search
        return self.simulate(random.choice(child_nodes))

    def update(self, node: int, values: List[float], cur_root: int) -> None:
        """
        Updates the stats of a given node (and its parents as well) using a given simulation result.
        Stops updating when cur_root is reached.
        :param: values list of values to append to stats
        """
        logger.debug(
            f"updating node: {node} with ({len(values)}) values (avg {avg(values):.3f})"
        )
        self.tree.nodes[node]["stats"]["vals"].extend(values)
        if node == cur_root:
            return
        parent = self.get_parent_name(node)
        if parent is not None:
            # update the results to the parent as well
            self.update(parent, values, cur_root)

    def get_child_names(self, node: int) -> List[int]:
        """Returns list of names of child nodes of the provided node e.g. [2, 3]"""
        node_names = [name for _, name in self.tree.edges(node)]
        return list(filter(lambda n: n > node, node_names))

    def get_parent_name(self, node: int) -> Union[int, None]:
        """Returns name of parent node (or None) if there's no parent."""
        node_names = [name for _, name in self.tree.edges(node)]
        parents = list(filter(lambda n: n < node, node_names))
        if len(parents) == 0:
            return None
        assert len(parents) == 1
        return parents[0]

    @staticmethod
    def edit_distance(add1: str, add2: str) -> int:
        assert len(add1) == len(add2)
        return sum([int(a1 != a2) for a1, a2 in zip(add1, add2)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tree search.")
    parser.add_argument("--depth", type=int, help="depth of tree to create", default=20)
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    FORMAT = "[%(levelname)5s][%(filename)s:%(lineno)s - %(funcName)15s()] %(message)s"
    logger.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(sh)

    # logging.basicConfig(format=FORMAT, level=log_level)

    mc = MCTS(args.depth, draw=False)
    # res = mc.simulate(1)

    tra = mc.run()
    print(f"returned trajectory: ")
    print(tra)
    value = mc.tree.nodes[tra[-1]]["value"]
    target_value = mc.tree.nodes[mc.target_name]["value"]
    dist = MCTS.edit_distance(
        mc.tree.nodes[tra[-1]]["address"], mc.tree.nodes[mc.target_name]["address"]
    )
    print(
        f"value of trajectory terminal node: {value:.2f} ({100 * value/target_value:.2f}% of target_value {target_value:.2f}), dist = {dist}"
    )

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

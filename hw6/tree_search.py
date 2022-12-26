#!/usr/bin/env python3
import argparse
from collections import defaultdict
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import sys
from typing import List, Union, Optional
import logging
import pdb

logger = logging.getLogger(__name__)

# we simply store a list of (payoff) values for each node
#   (where the length tracks the number of visits)
STATS_TEMPLATE = {"vals": []}

# max number of iterations starting at a given root node
MAX_ITERATIONS = 100

# max number of rollouts from a given "snowcap" leaf node
MAX_ROLLOUTS = 5

EXPANSION_PROB = 0.05


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

    def __init__(self, depth: int, c: float = 2, B: float = 5, draw: bool = False):
        """
        Init class with a binary tree of given depth.
        Nodes are named with ints incrementing from 1 up.

        :param depth: depth of tree to create
        :param B: param for computing values of leaf (terminal) nodes
        :param c: hyperparam for UCB calculation (higher value encourages more exploration)
        """
        self.c = c
        self.B = B

        self.reset(depth, draw=draw)

    def reset(self, depth: int, draw: bool = False, save: bool = True):
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
        if draw or save:
            nx.draw(tree, with_labels=True, node_size=300)
            if save:
                plt.savefig("graph.pdf", dpi=400)
                print("wrote graph.pdf")
            if draw:
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
        logger.info(
            f"target node = #{self.target_name} (value {target_value:.3f}), num leaf nodes = {len(leaf_node_names)}, max distance: {dmax}, min distance: {min(dists)}"
        )

        if draw or save:
            plt.clf()
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(11, 8.5)
            fig.tight_layout(pad=6.0)
            ax[0].set_xlabel("leaf node name")
            ax[0].set_ylabel("node value")
            ax[0].set_title(
                f"Distribution of Leaf Node Values (target node = {self.target_name})"
            )
            ax[0].plot(leaf_node_names, leaf_vals)

            ax[1].set_xlabel("leaf node name")
            ax[1].set_ylabel("Edit Distance from Target")
            ax[1].set_title(
                f"Distribution of Leaf Node Distances (target node = {self.target_name})"
            )
            ax[1].plot(leaf_node_names, dists)
            if draw:
                plt.show()
            if save:
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

        optimal_tra = self.get_path(self.target_name)  # for debugging
        # continue until cur_root is a leaf (terminal) node
        while len(self.get_child_names(cur_root)) > 1:
            # print(f"\nreached root {cur_root}")

            if "snowcap" not in self.tree.nodes[cur_root]:
                # init stats etc
                self.expansion(cur_root)

            # expand all children
            for c in self.get_child_names(cur_root):
                if "snowcap" not in self.tree.nodes[c]:
                    self.expansion(c)

            all_selected = defaultdict(int)
            for i in range(MAX_ITERATIONS):
                logger.debug(
                    f"root {cur_root}, iteration {i} (cur_root = {cur_root})\n"
                )
                node = self.selection(cur_root)
                all_selected[node] += 1
                expand = random.random() <= EXPANSION_PROB

                # on some iterations, the tree is expanded from the selected leaf node
                #   by adding one or more child nodes reached from the selected node via unexplored actions.
                if expand:
                    # TOOD: if root has only one snowcap children, ...
                    # node = self.expansion(node)
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
            rel_children = [
                n
                for n in self.get_child_names(cur_root)
                if "snowcap" in self.tree.nodes[n]
            ]
            if len(rel_children) < 2:
                logger.warning(f"only {len(rel_children)} options for final move")
            nodes = self.tree.nodes

            child_stats = {}
            for c in rel_children:
                child_stats[c] = {
                    "avg": round(avg(nodes[c]["stats"]["vals"]), 3),
                    "len": len(nodes[c]["stats"]["vals"]),
                    "value": nodes[c]["value"] if "value" in nodes[c] else None,
                }

            new_root = max(
                rel_children,
                key=lambda c: nodes[c]["value"]
                if "value" in nodes[c]
                else len(nodes[c]["stats"]["vals"]),
                # else avg(nodes[c]["stats"]["vals"]),
            )
            logger.info("all_selected was:")
            logger.info(all_selected)
            logger.info("child_stats=")
            logger.info(child_stats)
            if new_root not in optimal_tra:
                logger.info("child_stats=")
                logger.info(child_stats)
                logger.info("optimal_tra=")
                logger.info(optimal_tra)
                # pdb.set_trace()
                # TODO: draw current snowcap (subtree)
            cur_root = new_root
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
            # pdb.set_trace()
            return node  # if no children have been expanded, then select node

        parent_visits = len(self.tree.nodes[node]["stats"]["vals"])
        ucb_scores = {}
        random.shuffle(rel_children)  # in case of tie with UCB scores
        for c in rel_children:
            stats = self.tree.nodes[c]["stats"]
            if len(stats["vals"]) == 0:
                # expanded children with 0 visits will be targeted first
                ucb_scores[c] = sys.maxsize
            else:
                ucb_scores[c] = avg(stats["vals"]) + self.c * math.sqrt(
                    math.log(parent_visits) / len(stats["vals"])
                )

        selected = self.selection(max(rel_children, key=lambda c: ucb_scores[c]))

        if node == 1:
            print(ucb_scores)
            print(selected)
        return selected

    def expansion(self, node: int) -> Optional[int]:
        """
        Expands the search tree, from the given "snowcap" leaf node, randomly picking a new child to explore.
        If all children have been explored, returns node as fallback.
        """
        logger.debug(f"expanding from node {node}\n")
        if "snowcap" not in self.tree.nodes[node]:
            cur = node
        else:
            children = self.get_child_names(node)
            unex_children = [c for c in children if "snowcap" not in self.tree.nodes[c]]
            if len(unex_children) == 0:
                # logger.warning(f"unable to expand node {node} (0 non-snowcap children)")
                return node
            cur = random.choice(unex_children)

        logger.debug(f"expanding node {cur}\n")
        self.tree.nodes[cur]["stats"] = deepcopy(STATS_TEMPLATE)
        self.tree.nodes[cur]["snowcap"] = True
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

    def get_path(self, node: int) -> List[int]:
        """Get the trajectory (path) to a given node."""
        parent = self.get_parent_name(node)
        if parent is not None:
            return self.get_path(parent) + [node]
        return [node]

    @staticmethod
    def edit_distance(add1: str, add2: str) -> int:
        assert len(add1) == len(add2)
        return sum([int(a1 != a2) for a1, a2 in zip(add1, add2)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tree search.")
    parser.add_argument("--depth", type=int, help="depth of tree to create", default=20)
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug logging"
    )
    parser.add_argument("-c", type=float, help="hyperparam for UCB", default=1.5)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    FORMAT = "[%(levelname)7s][%(filename)s:%(lineno)s - %(funcName)15s()] %(message)s"
    logger.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(sh)

    # logging.basicConfig(format=FORMAT, level=log_level)

    mc = MCTS(args.depth, draw=False, c=args.c)
    # res = mc.simulate(1)

    tra = mc.run()
    print(f"returned trajectory: ")
    print(tra)
    value = mc.tree.nodes[tra[-1]]["value"]
    target_value = mc.tree.nodes[mc.target_name]["value"]
    dist = MCTS.edit_distance(
        mc.tree.nodes[tra[-1]]["address"], mc.tree.nodes[mc.target_name]["address"]
    )
    print(f"optimal trajectory =\n{mc.get_path(mc.target_name)}")
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

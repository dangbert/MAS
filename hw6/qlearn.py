#!/usr/bin/env python3
from copy import deepcopy
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Any
from copy import deepcopy
import random
from typing import List, Optional, Union, MutableSet
import pdb


class Spot(Enum):
    EMPTY = 0
    WALL = 1
    SNAKES = 2
    TREASURE = 3


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


Loc = Tuple[int, int]

World = Any  # TODO


class QLearn:
    world: World
    qtable: pd.DataFrame

    def __init__(self):
        self.init_world()
        self.init_q_table()

    def init_world(self) -> None:
        self.world = np.zeros((9, 9), dtype=int)
        self.world[1, 2:7] = Spot.WALL.value
        self.world[1:5, 6] = Spot.WALL.value
        self.world[7, 1:5] = Spot.WALL.value
        self.world[6:5] = Spot.SNAKES.value
        self.world[-1, -1] = Spot.TREASURE.value

    def init_q_table(self) -> None:
        # table = pd.DataFrame({"s_0": [], "s_1": [], "a": [], "r": [], "q": []})
        table = pd.DataFrame({"s": [], "a": [], "r": [], "q": []})
        self.world.shape
        rows, cols = self.world.shape

        for row in range(rows):
            for col in range(cols):
                s = (row, col)
                for a in self.get_possible_actions(s):
                    r = -1
                    q = np.random.normal()
                    if self.world[s] == Spot.TREASURE.value:
                        # terminal state
                        q, r = 0, 50
                    elif self.world[s] == Spot.SNAKES.value:
                        r = -50
                    next_s = self.apply_action(s, a)

                    cur_row = {"s": [s], "a": [a.value], "r": [r], "q": [q]}
                    # table = table.append(cur_row, ignore_index=True)
                    # table = pd.concat([table, pd.DataFrame(cur_row)], ignore_index=True)
                    table = pd.concat([table, pd.DataFrame(cur_row)], ignore_index=True)
        self.qtable = table
        return table

    def run_episode(
        self,
        s0: Optional[Loc] = None,
        alpha: float = 0.9,
        gamma: float = 2 / 3,
        epsilon: float = 0.08,
    ):
        """
        Run one episode of qlearning.

        :param s0 (optional) initial state to start at. If not provided random state is selected.
        :param alpha learning rate
        :param gamma discount factor of future rewards
        :param epsilon value for epsilon-greedy selection
            (i.e. the prob. of selecting a random action instead of the greedy best action)

        References:
            p. 131 in "Reinforcement Learning" 2nd Edition by Sutton.
        """
        assert 0.0 <= epsilon <= 1.0

        # get random initial (non-terminal) state e.g. (0,0)
        while s0 is None:
            s0 = (
                random.randint(0, self.world.shape[0] - 1),
                random.randint(0, self.world.shape[1] - 1),
            )
            if self.world[s0] != Spot.TREASURE.value:
                break

        count = 0
        rewards = []
        s = s0
        while True:
            # choose action from policy derived by qtable

            # get rows for this state, select row with action to perform
            cur_rows = self.qtable.loc[self.qtable["s"] == s]
            if random.random() > epsilon:
                cur_idx = cur_rows["q"].idxmax()  # select greedy action
            else:
                cur_idx = random.choice(cur_rows.index)  # select random action
            cur_row = self.qtable.iloc[cur_idx]

            next_s = self.apply_action(s, Action(cur_row["a"]))
            next_rows = self.qtable.loc[self.qtable["s"] == next_s]
            next_row = self.qtable.iloc[next_rows["q"].idxmax()]

            # update q value for (s,a) in place
            self.qtable.at[cur_idx, "q"] = cur_row["q"] + alpha * (
                cur_row["r"] + gamma * next_row["q"] - cur_row["q"]
            )
            rewards.append(cur_row["r"])
            s = next_s
            count += 1
            if self.world[s] == Spot.TREASURE.value:
                # TODO: how is reward for termianl state considered?
                break
        print(f"terminated episode after {count} actions")

    def apply_action(self, s: Loc, a: Union[Action, float]) -> Loc:
        """
        Returns location of new state after action.
        Next state may be the same as the previous state (e.g. bumpted into wall).
        """
        a = Action(a)  # ensure type(a) == Action
        if a == Action.UP:
            new_s = (s[0] - 1, s[1])
        elif a == Action.DOWN:
            new_s = (s[0] + 1, s[1])
        elif a == Action.LEFT:
            new_s = (s[0], s[1] - 1)
        elif a == Action.RIGHT:
            new_s = (s[0], s[1] + 1)
        else:
            raise NotImplementedError(f"unhandled action {a}")

        if (new_s[0] < 0 or new_s[0] >= self.world.shape[0]) or (
            new_s[1] < 0 or new_s[1] >= self.world.shape[1]
        ):
            new_s = s  # bumped into grid boundary so no state change
        if self.world[new_s] == Spot.WALL.value:
            new_s = s  # bumped into wall so no state change
        return new_s

    def get_possible_actions(self, s: Loc) -> MutableSet[Action]:
        """Return set of actions that are valid in state s."""
        return set(Action)

        actions = []
        for a in set(Action):
            try:
                self.apply_action(s, a)
                actions.append(a)
            except IndexError:
                continue
        return actions


if __name__ == "__main__":
    sim = QLearn()
    sim.run_episode()

    import pdb

    pdb.set_trace()
    exit(0)

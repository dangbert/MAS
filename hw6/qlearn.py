from copy import deepcopy
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Any
from copy import deepcopy
import random


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


def apply_action(a: Action, s: Loc, world: World) -> Loc:
    """
    Returns location of new state after action.
    Raises IndexError if action isn't allowed in current state.
    """
    new_s = None
    if a == Action.UP:
        new_s = (s[0] - 1, s[1])
    elif a == Action.DOWN:
        new_s = (s[0] + 1, s[1])
    elif a == Action.LEFT:
        new_s = (s[0], s[1] - 1)
    elif a == Action.RIGHT:
        new_s = (s[0], s[1] + 1)
    else:
        raise ValueError(f"invalid action {a}")

    if (new_s[0] < 0 or new_s[0] >= world.shape[0]) or (
        new_s[1] < 0 or new_s[1] >= world.shape[1]
    ):
        raise IndexError(f"action {a.value} invalid in state {s}")


def create_world() -> World:
    world = np.zeros((9, 9), dtype=int)
    world[1, 2:7] = Spot.WALL.value
    world[1:5, 6] = Spot.WALL.value
    world[7, 1:5] = Spot.WALL.value
    world[6:5] = Spot.SNAKES.value
    world[-1, -1] = Spot.TREASURE.value
    return world


def init_q_table(world) -> pd.DataFrame:
    # table = pd.DataFrame({"s_0": [], "s_1": [], "a": [], "r": [], "q": []})
    table = pd.DataFrame({"s": [], "a": [], "r": [], "q": []})
    world.shape
    rows, cols = world.shape

    for row in range(rows):
        for col in range(cols):
            s = (row, col)
            # get a random (valid) action for this state
            while True:
                a = random.choice(list(set(Action)))
                try:
                    apply_action(a, s, world)
                    break
                except IndexError:
                    continue

            # TODO consider optimal initial values for r and q
            r = np.random.normal()
            q = np.random.normal()
            cur_row = {"s": [s], "a": [a.value], "r": [r], "q": [q]}
            # table = table.append(cur_row, ignore_index=True)
            # table = pd.concat([table, pd.DataFrame(cur_row)], ignore_index=True)
            table = pd.concat([table, pd.DataFrame(cur_row)], ignore_index=True)

    return table


if __name__ == "__main__":
    world = create_world()
    table = init_q_table(world)

    import pdb

    pdb.set_trace()
    exit(0)

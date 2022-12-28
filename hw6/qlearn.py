#!/usr/bin/env python3
"""
Implements Q-Learning in a (static) grid world environment.
The textbook p. "Reinforcement Learning" 2nd Edition by Sutton was referenced for this assignment.
"""
from enum import Enum
import numpy as np
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import pdb
import random
from typing import List, Tuple, Any, Optional, Union, MutableSet, Dict


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


# for visualizing actions
ACTION_MAP = {
    Action.UP: "^",
    Action.DOWN: "v",
    Action.LEFT: "<",
    Action.RIGHT: ">",
}

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
        self.world[1:6, 6] = Spot.WALL.value
        self.world[7, 1:5] = Spot.WALL.value
        self.world[6, 5] = Spot.SNAKES.value
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
                    q = np.random.normal(scale=0.1)
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
        epsilon: float = 0.10,
        max_steps: Optional[int] = None,
        save_dir: str = "",
    ) -> Tuple[List, List]:
        """
        Run one episode of qlearning (play grid world until termination).

        :param s0 (optional) initial state to start at. If not provided random state is selected.
        :param alpha learning rate (set to 0.0 to disable learning)
        :param gamma discount factor of future rewards
        :param epsilon value for epsilon-greedy action selection (0.0 for pure greedy selection, 1.0 for pure random selection).
            (i.e. the prob. of selecting a random action instead of the greedy best action)

        :param max_steps stop simulation after this many steps if terminal state not reached.

        References:
            p. 131 in "Reinforcement Learning" 2nd Edition by Sutton.
        """
        assert 0.0 <= epsilon <= 1.0

        # get random initial (non-terminal) state e.g. (0,0)
        while s0 is None or self.world[s0] == Spot.TREASURE.value:
            s0 = (
                random.randint(0, self.world.shape[0] - 1),
                random.randint(0, self.world.shape[1] - 1),
            )

        rewards = []
        history = []
        s = s0
        while True:
            if max_steps is not None and len(rewards) >= max_steps:
                break
            # choose action from policy derived by qtable

            # get rows for this state, select row with action to perform
            cur_rows = self.qtable.loc[self.qtable["s"] == s]
            if random.random() < epsilon:
                cur_idx = random.choice(cur_rows.index)  # select random action
            else:
                cur_idx = cur_rows["q"].idxmax()  # select greedy action
            cur_row = self.qtable.iloc[cur_idx]

            if save_dir:
                if len(rewards) > 100:
                    pdb.set_trace()
                self.visualize_qtable(
                    save_path=os.path.join(
                        save_dir, f"step{str(len(rewards)).zfill(4)}.png"
                    ),
                    player_state=s,
                    title=f"s={s}, a={ACTION_MAP[Action(cur_row['a'])]}, r={cur_row['r']:.3f}, q={cur_row['q']:.3f}",
                )

            rewards.append(cur_row["r"])
            if self.world[s] == Spot.TREASURE.value:
                history.append(
                    {"s": s, "a": cur_row["a"], "r": cur_row["r"], "next_s": None}
                )
                break

            next_s = self.apply_action(s, Action(cur_row["a"]))
            next_rows = self.qtable.loc[self.qtable["s"] == next_s]
            next_row = self.qtable.iloc[next_rows["q"].idxmax()]

            # update q value for (s,a) in place
            self.qtable.at[cur_idx, "q"] = cur_row["q"] + alpha * (
                cur_row["r"] + gamma * next_row["q"] - cur_row["q"]
            )

            history.append(
                {"s": s, "a": cur_row["a"], "r": next_row["r"], "next_s": next_s}
            )
            s = next_s
        # print(f"terminated episode after {count} actions")
        return rewards, history

    def experience_replay(
        self,
        exp: Dict,
        alpha: float = 0.9,
        gamma: float = 2 / 3,
    ):
        """
        Learn by replaying a single given experience.

        :param exp the experience to replay
            (E.g. an entry of the history list returned by run_episode()

        References:
            p. 440 textbook
            https://youtu.be/Bcuj2fTH4_4?t=278
            https://youtu.be/0bt0SjbS3xc?t=272
        """
        assert type(exp) == dict
        if Spot(self.world[exp["s"]]) == Spot.TREASURE:
            return
        # self.qtable.loc[(self.qtable["s"] == exp['s'] & self.qtable['a'] == exp['a'])]
        cur_row = self.qtable[(self.qtable["s"] == exp["s"])].loc[
            self.qtable["a"] == exp["a"]
        ]
        assert len(cur_row) == 1
        cur_idx = cur_row.index[0]
        next_rows = self.qtable.loc[self.qtable["s"] == exp["next_s"]]
        next_row = self.qtable.iloc[next_rows["q"].idxmax()]

        # qlearn
        # self.qtable.at[cur_idx, "q"] = cur_row["q"] + alpha * (
        #    cur_row["r"] + gamma * next_row["q"] - cur_row["q"]
        # )
        self.qtable.at[cur_idx, "q"] = cur_row["q"] + alpha * (
            exp["r"] + gamma * next_row["q"] - cur_row["q"]
        )

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

    def visualize_qtable(
        self, title: str = "", player_state: Optional[Loc] = None, save_path: str = ""
    ):
        """Draw the world with greedy action for each state overlayed."""
        SPOT_ALPHA = 0.75

        # Create a figure and axis
        fig, ax = plt.subplots()
        if title:
            ax.set_title(title)

        # Set axis limits
        num_rows = self.world.shape[0]
        num_cols = self.world.shape[1]
        ax.set_xlim([-1, num_cols])
        ax.set_ylim([-1, num_rows])
        # ax.set_aspect("equal")
        # ax.invert_yaxis()

        arrows = []
        for r in range(self.world.shape[0]):
            arrows.append([])
            for c in range(self.world.shape[1]):
                s = (r, c)
                row_idx = self.qtable.loc[self.qtable["s"] == s]["q"].idxmax()
                a = Action(self.qtable.iloc[row_idx]["a"])

                arrows[-1].append(ACTION_MAP[a])
                arrow = ACTION_MAP[a]
                spot = Spot(self.world[s])

                if spot == Spot.TREASURE:
                    # ax.scatter(
                    #    c + 0.5,
                    #    num_rows - r - 1 - 0.5,
                    #    marker="*",
                    #    s=200,
                    #    color="green",
                    #    alpha=spot_alpha,
                    # )
                    ax.add_patch(
                        plt.Rectangle(
                            (c + 0.5, num_rows - r - 1 - 0.75),
                            0.5,
                            0.5,
                            facecolor="green",
                            alpha=SPOT_ALPHA,
                        )
                    )
                elif spot == Spot.SNAKES:
                    ax.add_patch(
                        plt.Rectangle(
                            (c + 0.5, num_rows - r - 1 - 0.75),
                            0.5,
                            0.5,
                            facecolor="red",
                            alpha=SPOT_ALPHA,
                        )
                    )
                elif spot == Spot.WALL:
                    ax.add_patch(
                        plt.Rectangle(
                            (c + 0.5, num_rows - r - 1 - 0.75),
                            0.5,
                            0.5,
                            facecolor="blue",
                            alpha=SPOT_ALPHA,
                        )
                    )

                if arrow in [">", "<", "^", "v"] and spot in {Spot.EMPTY, Spot.SNAKES}:
                    ax.arrow(
                        c + 0.5,
                        num_rows - r - 1 - 0.5,
                        {">": 0.5, "<": -0.5, "^": 0, "v": 0}[arrow],
                        {">": 0, "<": 0, "^": 0.5, "v": -0.5}[arrow],
                        head_width=0.2,
                        head_length=0.1,
                        fc="k",
                        ec="k",
                        length_includes_head=True,
                    )

                if player_state is not None and s == player_state:
                    ax.scatter(
                        c + 0.5,
                        num_rows - r - 1 - 0.5,
                        marker="*",
                        s=200,
                        color="yellow",
                        alpha=SPOT_ALPHA,
                    )

        # Remove the axis labels and ticks
        # ax.set_xticks([])
        # ax.set_yticks([])

        ax.set_xticks(np.arange(0, num_cols, 1))
        ax.set_yticks(np.arange(0, num_rows, 1))
        ax.grid(True, which="major", alpha=0.6)
        # ax.set_axis_off()

        if save_path:
            plt.savefig(save_path, dpi=400)
            plt.close(fig)
        else:
            plt.show()


# def fitness(sim: QLearn):


def strategy2a(display=print, save_dir=""):
    """Experiment with qlearning (direct updates)."""
    ITERATIONS = 1000
    # ITERATIONS = 5

    sim = QLearn()
    # world = ql.create_world()
    print(f"initial Q table:")
    display(sim.qtable)
    sim.visualize_qtable(title="Initial Random Q table (Showing Greedy Actions)")

    s0 = (0, 0)
    rewards = []
    for _ in range(ITERATIONS):
        rewards.append(sim.run_episode()[0])
        # rewards.append(sim.run_episode(s0=s0))

    plt.title(
        f"Total Rewards Across {ITERATIONS} Learning Episodes (With Random Initial States)"
    )
    plt.plot(range(0, len(rewards)), [sum(r) for r in rewards])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    print(f"new Q table:")
    display(sim.qtable)
    sim.visualize_qtable(title="Q Table Post Learning")

    # save_dir = "tmp"
    if save_dir:
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        print(f"writing to {save_dir}")

    ESPSILON = 0.05
    step_rewards = sim.run_episode(
        s0=s0, epsilon=ESPSILON, alpha=0.0, save_dir=save_dir
    )[0]
    plt.title(
        f"Reward Across Each Step of Single Episode (With Epsilon={ESPSILON:.3f} Greedy Selection)"
    )

    plt.plot(range(0, len(step_rewards)), step_rewards)
    plt.xlabel("Time Step")
    plt.ylabel("Reward")


def strategy2b():
    """Experiment with qlearning with experience replay buffer."""
    MAX_BUFFER = 10000
    MAX_STEPS = 10000

    sim = QLearn()
    sim.visualize_qtable(title=f"Q Table After {0} Replay Steps")

    # build buffer of experiences
    buffer = []
    while len(buffer) < MAX_BUFFER:
        # we use a high epsilon for experience collection to encourage trying a diverse set of actions
        buffer.extend(sim.run_episode(alpha=0, epsilon=0.75, max_steps=1000)[1])
    buffer = buffer[:MAX_BUFFER]

    print(f"collected {len(buffer)} experiences to sample from.")
    # exps = random.choices(buffer, k=1000)
    for _ in range(MAX_STEPS):
        sim.experience_replay(random.choice(buffer))
    sim.visualize_qtable(title=f"Q Table After {MAX_STEPS} Replay Steps")


if __name__ == "__main__":
    sim = QLearn()
    # sim.visualize_qtable()
    # sim.run_episode()

    strategy2a(save_dir="")
    # strategy2b()
    # build buffer of experiences
    # _, buffer = sim.run_episode(alpha=0, max_steps=50)
    # print(f"collected {len(buffer)} experiences to sample from.")
    # sim.replay_experience(buffer[0])

    pdb.set_trace()
    exit(0)

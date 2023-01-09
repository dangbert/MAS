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
from typing import List, Tuple, Any, Optional, Union, MutableSet, Dict, Callable


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
                    next_s = self.apply_action(s, a)
                    # the reward for (s,a) comes from next_s
                    if self.world[next_s] == Spot.TREASURE.value:
                        # terminal state
                        q, r = 0, 50
                    elif self.world[next_s] == Spot.SNAKES.value:
                        r = -50

                    cur_row = {
                        "s": [s],
                        "a": [a.value],
                        "r": [r],
                        "q": [q],
                    }
                    # table = table.append(cur_row, ignore_index=True)
                    # table = pd.concat([table, pd.DataFrame(cur_row)], ignore_index=True)
                    table = pd.concat([table, pd.DataFrame(cur_row)], ignore_index=True)
        self.qtable = table
        return table

    def get_initial_state(self):
        """Get random initial (non-terminal) state e.g. (1,2)"""
        s0 = None
        while s0 is None or Spot(self.world[s0]) != Spot.EMPTY:
            s0 = (
                random.randint(0, self.world.shape[0] - 1),
                random.randint(0, self.world.shape[1] - 1),
            )
        return s0

    def run_episode(
        self,
        s0: Optional[Loc] = None,
        alpha: float = 0.9,
        gamma: float = 2 / 3,
        sarsa: bool = False,
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
        :param sarsa toggle SARSA learnign algorithm (instead of Q-learning default)

        :param max_steps stop simulation after this many steps if terminal state not reached.
        :param save_dir optional path to directoy to save visualization of each time step

        References:
            p. 131 in "Reinforcement Learning" 2nd Edition by Sutton.
        """
        assert 0.0 <= epsilon <= 1.0

        s0 = self.get_initial_state() if s0 is None else s0

        def epsilon_greedy(rows: pd.DataFrame, epsilon: float) -> int:
            """Do epsilon greedy action selection, returning the index of the selected row (from a subset of the qtable)."""
            if random.random() < epsilon:
                return random.choice(rows.index)  # select random action
            return rows["q"].idxmax()  # select greedy action

        rewards = []
        history = []
        s = s0
        while True:
            if max_steps is not None and len(rewards) >= max_steps:
                break
            if self.world[s] == Spot.TREASURE.value:
                break

            # get rows for this state, select row with action to perform
            cur_rows = self.qtable.loc[self.qtable["s"] == s]
            cur_idx = epsilon_greedy(cur_rows, epsilon)
            cur_row = self.qtable.iloc[cur_idx]

            # choose action from policy derived by qtable
            next_s = self.apply_action(s, Action(cur_row["a"]))
            next_rows = self.qtable.loc[self.qtable["s"] == next_s]
            next_row = self.qtable.iloc[next_rows["q"].idxmax()]
            if sarsa:
                next_row = self.qtable.iloc[epsilon_greedy(next_rows, epsilon)]

            # update q value for (s,a) in place
            self.qtable.at[cur_idx, "q"] = cur_row["q"] + alpha * (
                cur_row["r"] + gamma * next_row["q"] - cur_row["q"]
            )

            rewards.append(cur_row["r"])
            history.append(
                {
                    "s": s,
                    "a": cur_row["a"],
                    "r": next_row["r"],
                    "next_s": next_s,
                    "terminal": self.world[next_s] == Spot.TREASURE.value,
                }
            )
            if save_dir:
                self.visualize_qtable(
                    save_path=os.path.join(
                        save_dir, f"step{str(len(rewards)).zfill(4)}.png"
                    ),
                    player_state=s,
                    title=f"s={s}, a={ACTION_MAP[Action(cur_row['a'])]}, r={cur_row['r']:.3f}, q={cur_row['q']:.3f}",
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
        Next state may be the same as the previous state (e.g. bumped into wall).
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

        # hide axis labels (but keep grid visible)
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=400)
            plt.close(fig)
        else:
            plt.show()


def direct_updates(
    sarsa: bool = False,
    epsilon: float = 0.1,
    iterations: int = 1000,
    show_initial: bool = True,
    show_final: bool = False,
    display: Callable = print,
    save_dir: str = "",
    csv_path: str = "",
) -> QLearn:
    """
    Experiment with qlearning (direct updates).
    Set sarsa=True to experiment with SARSA instead of q-learning.
    Set epsilon=0.0 for greedification.
    """
    sim = QLearn()
    if show_initial:
        print(f"initial Q table:")
        display(sim.qtable)
        sim.visualize_qtable(title="Initial Random Q table (Showing Greedy Actions)")

    s0 = (0, 0)
    rewards = []
    for _ in range(iterations):
        rewards.append(sim.run_episode(sarsa=sarsa, epsilon=epsilon)[0])
        # rewards.append(sim.run_episode(s0=s0))

    plt.title(
        f"Total Rewards Across {iterations} Learning Episodes (With Random Initial States)"
    )
    plt.plot(range(0, len(rewards)), [sum(r) for r in rewards])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    print(f"Q table (post learning):")
    display(sim.qtable)
    sim.visualize_qtable(title="Q Table Post Learning (Showing Greedy Actions)")

    # save_dir = "tmp"
    if save_dir:
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        print(f"writing to {save_dir}")

    if show_final:
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
    if csv_path:
        sim.qtable.to_csv(csv_path)
        print(f"wrote {csv_path}")
    return sim


def strategy2b(max_buffer: int = 5000, max_replays: int = 10000):
    """Experiment with qlearning with experience replay buffer."""
    sim = QLearn()
    sim.visualize_qtable(title=f"Q Table After {0} Replay Steps")

    # build buffer of experiences
    buffer = []
    while len(buffer) < max_buffer:
        # we use a high epsilon for experience collection to encourage trying a diverse set of actions
        history = sim.run_episode(alpha=0, epsilon=0.75, max_steps=1000)[1]
        # store experiences closer to the episode termination
        history = history[-200:]
        buffer.extend(history)
    buffer = buffer[:max_buffer]

    print(f"collected {len(buffer)} experiences to sample from.")
    # exps = random.choices(buffer, k=1000)
    for _ in range(max_replays):
        sim.experience_replay(random.choice(buffer))
    sim.visualize_qtable(title=f"Q Table After {max_replays} Replay Steps")


def strategy3(
    max_buffer: int = 10000, training_steps: int = 20000, save_dir: str = "dqn_output"
):
    """
    Using a DQN.
    see dqn.py for reference links.
    """
    from dqn import DQN, DQAgent
    import torch

    # gamma = 0.99
    gamma = 2 / 3
    batch_size = 32
    start_e, end_e = 1.0, 0.1
    epsilon = start_e
    C = 500  # how often to update target_net

    sim = DQAgent()

    buffer = []  # experience buffer
    stats = {"step": [], "fitness": []}

    model_path = os.path.join(save_dir, "model.pth")
    stats_path = os.path.join(save_dir, "training.pdf")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print(f"will save to '{save_dir}'")

    def plot_stats(stats, save_path: str = ""):
        plt.clf()
        plt.title(f"Average Reward of Policy Throughout Training")
        plt.plot(stats["step"], stats["fitness"])
        plt.xlabel("Training Step")
        plt.ylabel("Average Reward (per move)")
        if save_path:
            print("wrote")
            plt.savefig(save_path)
        else:
            plt.show()

    # maps (x,y) positions to Q values for each possible action
    net = DQN(2, len(Action))
    target_net = DQN(2, len(Action))
    for i in range(0, training_steps):
        if i % C == 0:
            # update target_net (on first iteration and every C iterations after)
            target_net.load_state_dict(net.state_dict())

        # anneal e from start_e -> end_e
        #   (initialy using a high epsilon for experience collection to encourage trying a diverse set of actions)
        epsilon = start_e - (i / training_steps) * (start_e - end_e)
        # print(f"epsilon = {epsilon:.3f}")

        if i % int(training_steps / 50) == 0:
            stats["step"].append(i)
            stats["fitness"].append(sim.measure(net))
            plot_stats(stats, save_path=stats_path)
            torch.save(net.state_dict(), model_path)
            print(
                f"training step {i}/{training_steps}: avg_reward = {stats['fitness'][-1]:.3f}"
            )

        # (conditionally) update memory buffer with newer experiences
        if i % 1000 == 0:
            buffer = buffer[1000:]
        while len(buffer) < max_buffer:
            history = sim.run_episode(net, epsilon=epsilon, max_steps=2000)[1]
            buffer.extend(history)
        buffer = buffer[:max_buffer]

        # get batch of experiences
        batch_raw = np.random.choice(buffer, batch_size, replace=False)
        int_keys = ["terminal", "a"]
        batch = {
            # e.g. for k='s' create tensor where each row is one state (e.g. dim 32x2)
            k: torch.tensor(np.array([b[k] for b in batch_raw], dtype=np.float32)).to(
                net.device
            )
            for k in set(batch_raw[0].keys()) - set(int_keys)
        }
        # 1 indicates a terminal state, 0 otherwise
        for k in int_keys:  # int data types
            batch[k] = torch.tensor(
                np.array([int(b[k]) for b in batch_raw], dtype=np.int64)
            ).to(net.device)

        # compute target q values
        target_qs = target_net(batch["next_s"]).to(net.device)
        # max q value for each batch
        # max_target_qs = target_qs.max(dim=1, keepdim=True)[0].to(net.device)
        max_target_qs = target_qs.max(dim=1)[0].to(net.device)

        # compute q learning targets (note for terminal states the future reward is zeroed)
        # TODO: understand this better given targets are for (s,a) pair...
        targets = (batch["r"] + gamma * (1.0 - batch["terminal"]) * max_target_qs).to(
            net.device
        )
        predicted_qs = net(batch["s"]).to(net.device)
        # get predicted q value of action taken in each episode
        action_qs = torch.gather(
            input=predicted_qs, dim=1, index=batch["a"].unsqueeze(-1)
        ).to(net.device)

        net.optimizer.zero_grad()
        # TODO: understand how loss works better
        loss = net.loss(action_qs, targets)
        loss.backward()
        net.optimizer.step()

        i += 1


if __name__ == "__main__":
    # sim = QLearn()
    # sim.visualize_qtable()
    # sim.run_episode()

    # direct_updates()
    # strategy2b()

    strategy3(max_buffer=1000)

    pdb.set_trace()
    exit(0)

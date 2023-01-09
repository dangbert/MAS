#!/usr/bin/env python3
import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import random
from typing import Any, Optional, Tuple, List
import pandas as pd
from pathlib import Path

import numpy as np
import qlearn
from qlearn import Spot, Action, ACTION_MAP, Loc
import matplotlib.pyplot as plt
import json


class DQN(nn.Module):
    stack: nn.Sequential
    device: str
    loss: Any
    optim: optim.Optimizer

    def __init__(self, input_len: int, output_len: int, lr: float = 0.0025):
        """
        Note for LM project, there are 8 IRs so input_len could be 8.
        Inspired by:
            phil's video: https://www.youtube.com/watch?v=wc-FxNENg9U
                https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py

            brthor's video: https://www.youtube.com/watch?v=wc-FxNENg9U
                https://github.com/fsan/dqn_-brthor/blob/main/main.py


        parmas:
            lr: learning rate
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stack = nn.Sequential(
            nn.Linear(input_len, 30),
            nn.Sigmoid(),
            nn.Linear(30, 15),
            nn.Sigmoid(),
            nn.Linear(15, output_len),
            # nn.Softmax(dim=output_len),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # mean sequared error loss
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        """
        Defines the forward computation at every call.
        Note: This shouldn't be called directly.
        """
        # TODO: perhaps normalize IR inputs first (remove -infinity?
        actions = self.stack(state)
        return actions


class DQAgent(qlearn.QLearn):
    # net: DQN

    def __init__(self):
        super().__init__()
        # self.net = net

    def run_episode(
        self,
        net: DQN,
        s0: Optional[Loc] = None,
        epsilon: float = 0.10,
        max_steps: Optional[int] = None,
        save_dir: str = "",
    ) -> Tuple[List, List]:
        """
        Runs an epsiode to completion or until max_steps have passed.
        Doesn't do any learning, just makes actions based on the policy defined by self.net and epsilon
        """
        assert 0.0 <= epsilon <= 1.0
        s0 = self.get_initial_state() if s0 is None else s0

        device = net.device
        rewards = []
        history = []
        s = s0
        while True:
            if max_steps is not None and len(rewards) >= max_steps:
                break
            if self.world[s] == Spot.TREASURE.value:
                break

            qvals = net(torch.tensor(s, dtype=torch.float32))
            # epsilon-greedy action selection
            if random.random() < epsilon:
                a = random.choice(list(Action))
            else:
                a = Action(torch.argmax(qvals).item())
            q = qvals[a.value].item()

            r = self.get_reward(s, a)
            next_s = self.apply_action(s, a)

            rewards.append(r)
            history.append(
                {
                    "s": s,
                    "a": a.value,
                    "r": r,
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
                    title=f"s={s}, a={ACTION_MAP[Action(a)]}, r={r:.3f}, q={q:.3f}, (epsilon = {epsilon}",
                )
            s = next_s
        # print(f"terminated episode after {count} actions")
        return rewards, history

    def measure(
        self, net: DQN, trials=10, epsilon=0.05, max_steps=250, save_dir: str = ""
    ):

        """Evaluate the DQN by having it run several trials of episodes and report the average cumulative reward."""
        rewards = []
        raw_rewards = []
        for n in range(trials):
            cur_dir = ""
            if save_dir:
                cur_dir = os.path.join(save_dir, f"trial{n}")
                if not os.path.isdir(cur_dir):
                    os.makedirs(cur_dir)

            cur_rewards = self.run_episode(
                net, epsilon=epsilon, max_steps=max_steps, save_dir=cur_dir
            )[0]
            total_reward = sum(cur_rewards)
            rewards.append(total_reward)
            raw_rewards.append(cur_rewards)
        return avg(rewards)

    def get_reward(self, s: Loc, a: Action):
        """Using self.qtable just to get the reward associated with (s,a) pairs."""
        rows = self.qtable[(self.qtable["s"] == s) & (self.qtable["a"] == a.value)]
        assert len(rows) == 1
        return rows.loc[rows.index[0]]["r"]

    def visualize(self, net: DQN, **kwargs):
        # build self.qtable from net and visualize
        self.init_q_table(net=net)
        self.visualize_qtable(**kwargs)


def avg(arr):
    return sum(arr) / len(arr)


# def reload(dir: str):
#    pass


def run(
    save_dir: str,
    lr: float = 0.0025,
    max_buffer: int = 50000,
    training_steps: int = int(1e9),
    start_e: float = 1.0,
    end_e: float = 0.1,
    num_measures: int = 100,
    C: int = 10000,  # how often to update target_net
    replay_only: bool = False,
    replay_draw: bool = False,  # draw measured episodes
):
    """
    Using a DQN.
    see dqn.py for reference links.
    """
    from dqn import DQN, DQAgent
    import torch

    gamma = 0.99
    # gamma = 2 / 3
    batch_size = 64
    epsilon = start_e

    sim = DQAgent()

    buffer = []  # experience buffer

    model_path = os.path.join(save_dir, "model.pth")
    graph_path = os.path.join(save_dir, "training.pdf")
    stats_path = os.path.join(save_dir, "stats.json")
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
            plt.savefig(save_path)
        else:
            plt.show()

    # maps (x,y) positions to Q values for each possible action
    net = DQN(2, len(Action), lr=lr)
    print(f"device = {net.device}")
    target_net = DQN(2, len(Action), lr=lr)
    stats = {"step": [], "fitness": []}

    start_step = 0
    reloaded = False
    if os.path.exists(model_path):
        print(f"reloaded network from '{model_path}'")
        net.load_state_dict(torch.load(model_path))
        reloaded = True
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            start_step = int(stats["step"][-1]) + 1
            print(f"reloaded stats from '{stats_path}' (start_step = {start_step})")

    if replay_only:
        assert reloaded, f"failed to find model to reload: '{model_path}'"
        print(f"replaying data from '{save_dir}'")
        sim.visualize(
            net, save_path=os.path.join(save_dir, f"qtable_step{start_step-1}.pdf")
        )
        trials = 25
        if replay_draw:
            trials = 10
            score = sim.measure(net, trials=trials, save_dir=save_dir)
        else:
            score = sim.measure(net, trials=trials)
        print(f"score = {score} (averaged over {trials} trials)")
        print("replay done!")
        exit(0)

    must_init = True
    for i in range(start_step, training_steps):
        if must_init or i % C == 0:
            print("updating target_net")
            # update target_net (on first iteration and every C iterations after)
            target_net.load_state_dict(net.state_dict())
            must_init = False

        # anneal e from start_e -> end_e
        #   (initialy using a high epsilon for experience collection to encourage trying a diverse set of actions)
        epsilon = start_e - (i / training_steps) * (start_e - end_e)
        # print(f"epsilon = {epsilon:.3f}")

        if i % int(training_steps / num_measures) == 0:
            stats["step"].append(i)
            stats["fitness"].append(sim.measure(net, trials=25))
            plot_stats(stats, save_path=graph_path)
            torch.save(net.state_dict(), model_path)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

            print(
                f"training step {i}/{training_steps} ({(100 * i / training_steps):.2f}%): avg_reward = {stats['fitness'][-1]:.3f}, epsilon={epsilon:.3f}, lr={lr:.4f}"
            )

        # (conditionally) update memory buffer with newer experiences
        if i % 5000 == 0:
            buffer = buffer[5000:]
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
        targets = (
            (batch["r"] + gamma * (1.0 - batch["terminal"]) * max_target_qs)
            .to(net.device)
            .unsqueeze(-1)
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


def main():
    # strategy3(
    #    max_buffer=50000,
    #    training_steps=int(5e6),
    #    start_e=1.0,
    #    num_measures=1500,
    #    save_dir="tmp",
    # )

    parser = argparse.ArgumentParser(description="Run DQN on gridworld")
    basedir = "dqn_experiments"
    parser.add_argument("dir", type=str, help="directory to save/load model and stats")
    parser.add_argument(
        "-se", "--start_e", default=1.0, type=float, help="start epsilon"
    )
    parser.add_argument("-ee", "--end_e", default=0.1, type=float, help="end epsilon")
    parser.add_argument(
        "-lr", "--learning_rate", default=0.0025, type=float, help="learning rate"
    )
    parser.add_argument(
        "--steps",
        default=int(3e6),
        type=int,
        help="number of training steps to run for",
    )
    parser.add_argument(
        "-b",
        "--buffer_size",
        default=50000,
        type=int,
        help="size of memory buffer",
    )
    parser.add_argument(
        "-m",
        "--measures",
        default=500,
        type=int,
        help="number of times to measure performance throughout training also also save network to disk ",
    )
    parser.add_argument(
        "-r",
        "--replay",
        action="store_true",
        help="replay / reload stats instead of continuing experiment by default",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="draw (measured) episodes during replay",
    )
    args = parser.parse_args()

    # prefer storing experiments within basedir unless user provides absolute path
    if Path(basedir) not in Path(args.dir).parents and not os.path.isabs(args.dir):
        args.dir = os.path.join(basedir, args.dir)

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    print(f"args = {args}\n")
    run(
        args.dir,
        lr=args.learning_rate,
        max_buffer=args.buffer_size,
        training_steps=args.steps,
        start_e=args.start_e,
        end_e=args.end_e,
        num_measures=args.measures,
        replay_only=args.replay,
        replay_draw=args.draw,
    )


if __name__ == "__main__":
    main()

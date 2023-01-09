#!/usr/bin/env python3
import os
import torch
from torch import nn
import torch.optim as optim
import random
from typing import Any, Optional, Tuple, List
import pandas as pd

import qlearn
from qlearn import Spot, Action, ACTION_MAP, Loc
import pdb


class DQN(nn.Module):
    stack: nn.Sequential
    device: str
    loss: Any
    optim: optim.Optimizer

    def __init__(self, input_len: int, output_len: int, lr: float = 25e-5):
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
        # save_dir: str = "",
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

            # epsilon-greedy action selection
            if random.random() < epsilon:
                a = random.choice(list(Action))
            else:
                qvals = net(torch.tensor(s, dtype=torch.float32))
                a = Action(torch.argmax(qvals).item())

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
            s = next_s
        # print(f"terminated episode after {count} actions")
        return rewards, history

    def measure(self, net: DQN, trials=10, epsilon=0.05, max_steps=250):
        """Evaluate the DQN by having it run several trials of episodes and report the average cumulative reward."""
        rewards = []
        for _ in range(trials):
            rewards.extend(
                self.run_episode(net, epsilon=epsilon, max_steps=max_steps)[0]
            )
        return sum(rewards) / len(rewards)

    def get_reward(self, s: Loc, a: Action):
        """Using self.qtable just to get the reward associated with (s,a) pairs."""
        rows = self.qtable[(self.qtable["s"] == s) & (self.qtable["a"] == a.value)]
        assert len(rows) == 1
        return rows.loc[rows.index[0]]["r"]

    # def visualize(self, net: DQN):
    #    # TODO: build q table from net (implement DQN.to_qtable())
    #    # super.visalize_qtable()


def main():
    pass


if __name__ == "__main__":
    main()

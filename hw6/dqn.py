#!/usr/bin/env python3
import torch
from torch import nn
import torch.optim as optim
from typing import Any


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


def main():
    pass


if __name__ == "__main__":
    main()

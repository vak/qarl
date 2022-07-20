import torch

from .abstract_oracle import AbstractOracle


class Oracle(AbstractOracle):
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def assess_the_guess(self, environment: torch.Tensor, guess: torch.Tensor) -> torch.Tensor:
        diff = guess - environment
        result = torch.clone(guess) * 0
        result[diff < -self.eps] = -1
        result[diff > self.eps] = 1
        return result

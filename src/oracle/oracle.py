import numpy as np

from .abstract_oracle import AbstractOracle


class Oracle(AbstractOracle):
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def assess_the_guess(self, environment: np.ndarray, guess: np.ndarray) -> np.ndarray:
        guess = np.array(guess)
        diff = guess - environment
        result = np.zeros(guess.shape)
        result[diff < -self.eps] = -1
        result[diff > self.eps] = 1
        return result

from typing import Generator

import numpy as np

from .abstact_enviroment import AbstractEnvironment


class GeneratorEnvironment(AbstractEnvironment):
    def __init__(self, number_generator: Generator[float, None, None], eps: float = 1e-7):
        self.generator = number_generator
        self.eps = eps
        self.value = next(self.generator)

    def assess_the_guess(self, guess: np.ndarray) -> np.ndarray:
        guess = np.array(guess)
        diff = guess - self.value
        self.value = next(self.generator)
        result = np.zeros(guess.shape)
        result[diff < -self.eps] = -1
        result[diff > self.eps] = 1
        return result

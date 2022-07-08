from abc import ABC, abstractmethod
from typing import List, TypedDict

import numpy as np


class GameState(TypedDict):
    agent_state: np.ndarray
    agent_step: np.ndarray
    environment_state: np.ndarray


class AbstractGame(ABC):
    @abstractmethod
    def run_iteration(self) -> GameState:
        pass

    def run_n_iterations(self, n=1) -> List[GameState]:
        states = [
            self.run_iteration()
            for _ in range(n)
        ]
        return states

    def __iter__(self):
        return self

    def __next__(self):
        return self.run_iteration()

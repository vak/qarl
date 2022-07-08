from abc import abstractmethod, ABC

import numpy as np


class AbstractOracle(ABC):
    @abstractmethod
    def assess_the_guess(self, environment: np.ndarray, guess: np.ndarray) -> np.ndarray:
        pass

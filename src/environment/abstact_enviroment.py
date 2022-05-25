from abc import abstractmethod, ABC

import numpy as np


class AbstractEnvironment(ABC):
    @abstractmethod
    def assess_the_guess(self, guess: np.ndarray) -> np.ndarray:
        pass

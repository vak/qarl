from abc import abstractmethod, ABC

import numpy as np


class AbstractAgent(ABC):
    @abstractmethod
    def react(self, feedback: np.ndarray | float) -> np.ndarray | float:
        pass

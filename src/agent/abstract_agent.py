from abc import abstractmethod, ABC

import numpy as np


class AbstractAgent(ABC):
    @abstractmethod
    def adapt(self, feedback: np.ndarray | float) -> np.ndarray | float:
        pass

    @abstractmethod
    def get_current_state(self) -> np.ndarray | float:
        pass

    @abstractmethod
    def get_current_step(self) -> np.ndarray | float:
        pass

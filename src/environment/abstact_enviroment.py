from abc import abstractmethod, ABC

import numpy as np


class AbstractEnvironment(ABC):
    @abstractmethod
    def get_next_state(self) -> np.ndarray | float:
        pass

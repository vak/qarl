from abc import ABC, abstractmethod

import numpy as np
import torch


class AbstractEnvironment(ABC):
    @abstractmethod
    def get_next_state(self) -> np.ndarray | torch.Tensor | float:
        pass

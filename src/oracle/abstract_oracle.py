from abc import ABC, abstractmethod

import numpy as np
import torch


class AbstractOracle(ABC):
    @abstractmethod
    def assess_the_guess(self, environment: np.ndarray | torch.Tensor | float,
                         guess: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
        pass

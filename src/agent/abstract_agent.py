from abc import ABC, abstractmethod

import numpy as np
import torch


class AbstractAgent(ABC):
    @abstractmethod
    def adapt(self, feedback: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
        pass

    @abstractmethod
    def get_current_state(self) -> np.ndarray | torch.Tensor | float:
        pass

    @abstractmethod
    def get_current_step(self) -> np.ndarray | torch.Tensor | float:
        pass

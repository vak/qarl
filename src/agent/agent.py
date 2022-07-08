import numpy as np

from .abstract_agent import AbstractAgent


# TODO: Add value function and policy
# TODO: Add apply_action function
class Agent(AbstractAgent):
    def __init__(self, acceleration: np.ndarray, braking: np.ndarray, low_quantile: float = 0.1,
                 high_quantile: float = 0.9, quantile: np.ndarray | None = None, low_initial_value: float = -1e3,
                 high_initial_value: float = 1e3, initial_value: np.ndarray | None = None, low_initial_step: float = -5,
                 high_initial_step: float = 5, initial_step: np.ndarray | None = None, seed: int = 0,
                 min_step: float = 0):
        super().__init__()
        assert acceleration.shape == braking.shape
        self.acceleration = acceleration
        self.braking = braking
        self.min_step = min_step

        if initial_value is None:
            np.random.seed(seed)
            self.current_value = np.random.uniform(low_initial_value, high_initial_value, len(acceleration))
        else:
            assert acceleration.shape == initial_value.shape
            self.current_value = initial_value

        if quantile is None:
            self.quantile = np.linspace(low_quantile, high_quantile, len(acceleration))
        else:
            assert acceleration.shape == quantile.shape
            self.quantile = quantile

        if initial_step is None:
            np.random.seed(seed)
            self.step = np.random.uniform(low_initial_step, high_initial_step, len(acceleration))
        else:
            assert acceleration.shape == initial_step.shape
            self.step = initial_step

    def adapt(self, feedback: np.ndarray) -> np.ndarray:
        feedback_is_the_same = (self.step * feedback) < 0

        # E step
        self.step *= np.where(feedback_is_the_same, self.acceleration, self.braking)
        min_step_mask = np.abs(self.step) < self.min_step
        self.step[min_step_mask] = np.sign(self.step[min_step_mask]) * self.min_step

        # M step
        self.current_value += np.abs(self.step) * (-feedback + 2.0 * self.quantile - 1.0)

        return self.current_value

    def get_current_state(self) -> np.ndarray | float:
        return self.current_value.copy()

    def get_current_step(self) -> np.ndarray | float:
        return self.step.copy()

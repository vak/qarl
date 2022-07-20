import torch

from .abstract_agent import AbstractAgent


# TODO: Add value function and policy
# TODO: Add apply_action function
class Agent(AbstractAgent):
    def __init__(self, acceleration: torch.Tensor, braking: torch.Tensor, device=torch.device('cpu'),
                 low_initial_value: float = -1e3, high_initial_value: float = 1e3,
                 initial_value: torch.Tensor | None = None, low_initial_step: float = -5, high_initial_step: float = 5,
                 initial_step: torch.Tensor | None = None, seed: int = 0, min_step: float = 0):
        super().__init__()
        self.n = len(acceleration)
        self.device = device
        assert acceleration.shape == braking.shape
        self.acceleration = acceleration.to(self.device)
        self.braking = braking.to(self.device)
        self.min_step = min_step

        torch.manual_seed(seed)
        self.current_value = self._initial_or_random(
            low_initial_value, high_initial_value, initial_value)
        self.step = self._initial_or_random(
            low_initial_step, high_initial_step, initial_step)

    def _initial_or_random(self, low: float, high: float, initial: torch.Tensor | None = None):
        if initial is None:
            data = (high - low) * torch.rand(self.n) + low
        else:
            assert len(initial) == self.n
            data = initial
        return data.to(self.device)

    def adapt(self, feedback: torch.Tensor) -> torch.Tensor:
        feedback_is_the_same = (self.step * feedback) < 0
        # E step
        self.step *= torch.where(feedback_is_the_same,
                                 self.acceleration, self.braking)
        min_step_mask = torch.abs(self.step) < self.min_step
        self.step[min_step_mask] = torch.sign(
            self.step[min_step_mask]) * self.min_step
        # M step
        self.current_value += self.step
        return self.current_value

    def get_current_state(self) -> torch.Tensor:
        return torch.clone(self.current_value)

    def get_current_step(self) -> torch.Tensor:
        return torch.clone(self.step)

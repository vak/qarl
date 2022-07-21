from typing import Generator, Tuple

import numpy as np
from gym import Env, spaces
from gym.core import ObsType
from gym.envs.registration import register
from matplotlib import pyplot as plt

from src.generator import gaussian, sin_gaussian


class SkippyEnv(Env):
    metadata = {'render.modes': ['human'], 'render_fps': 30}

    def __init__(self, generator: Generator[float, None, None] = gaussian(sigma=0), eps=1e-5):
        super(SkippyEnv, self).__init__()
        self.generator = generator
        self.eps = eps
        self.action_space = spaces.Box(
            low=-1e7, high=1e7, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-1e7, high=1e7, shape=(1,), dtype=np.float64)

        self.current_value = next(self.generator)
        self.last_diff = 1e7
        self.last_guess = 1e7
        self.plot_length = 200

        self.fig, self.axs = plt.subplots(2)
        self.fig.canvas.set_window_title('SkippyEnv')
        self.diff_plot, = self.axs[0].plot(
            [0], [0], label='Prediction - Conceived')
        self.conceived_plot, = self.axs[1].plot([0], [0], label='Conceived')
        self.prediction_plot, = self.axs[1].plot([0], [0], label='Prediction')
        plt.ion()
        plt.show()

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        self.current_value = next(self.generator)
        diff = action.sum() - self.current_value
        self.last_diff = diff
        self.last_guess = action.sum()
        done = bool(abs(diff) < self.eps)
        observation = -1 if diff < -self.eps else 1
        if done:
            observation = 0
        reward = 1 if self.last_diff > diff else 0
        info = {}
        return np.array(observation).reshape((1,)), reward, done, info

    def reset(self, **kwargs):
        self.generator = self.generator
        self.current_value = next(self.generator)
        self.last_diff = 1e7
        return np.array([-1])

    def render(self, mode="human"):
        diff_stats = np.append(self.diff_plot.get_ydata(), np.abs(
            self.last_diff))[-self.plot_length:]
        self.diff_plot.set_ydata(diff_stats)
        self.diff_plot.set_xdata(list(range(len(diff_stats))))

        prediction_stats = np.append(
            self.prediction_plot.get_ydata(), self.last_guess)[-self.plot_length:]
        self.prediction_plot.set_ydata(prediction_stats)
        self.prediction_plot.set_xdata(list(range(len(prediction_stats))))

        conceived_stats = np.append(
            self.conceived_plot.get_ydata(), self.current_value)[-self.plot_length:]
        self.conceived_plot.set_ydata(conceived_stats)
        self.conceived_plot.set_xdata(list(range(len(conceived_stats))))

        self.axs[0].set_ylim((0, np.max(diff_stats[-self.plot_length // 10:])))
        self.axs[0].set_xlim((0, self.plot_length))
        self.axs[1].set_ylim((
            min(np.min(
                prediction_stats[-self.plot_length // 10:]), np.min(conceived_stats)),
            max(np.max(
                prediction_stats[-self.plot_length // 10:]), np.max(conceived_stats))
        ))
        self.axs[1].set_xlim((0, self.plot_length))
        self.axs[0].legend(loc='upper right')
        self.axs[1].legend(loc='upper right')

        self.fig.canvas.flush_events()
        self.fig.canvas.draw()


register(
    id='skippy/env',
    entry_point='src.open_ai_gym.environment:SkippyEnv',
    max_episode_steps=10000,
)

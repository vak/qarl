from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.game import GameState


def show_game_state(states: List[GameState]):
    plt.figure(figsize=(16, 14))
    plt.subplot(2, 1, 1)
    plt.title(r'Value, $x_t$')

    environment_state = np.array([state['environment_state'] for state in states])
    plt.plot(environment_state, 'b', label='Conceived value')
    plt.ylim((
        np.min(environment_state) - 3,
        np.max(environment_state) + 3
    ))

    agent_state = np.array([state['agent_state'] for state in states])
    plt.plot(agent_state, 'r--', label='Estimated value')

    agent_step = np.array([state['agent_step'] for state in states])
    plt.subplot(2, 1, 2)
    plt.title(r'Step, $\delta_t$')
    plt.plot(agent_step)

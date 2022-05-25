import numpy as np
from matplotlib import pyplot as plt

from src.agent import Agent
from src.environment import GeneratorEnvironment


# TODO: Create abstract class
class Game:
    def __init__(self, agent: Agent, environment: GeneratorEnvironment):
        self.agent = agent
        self.environment = environment
        self.last_guess = agent.current_value
        self.stats = {
            "agent_value": [],
            "agent_step": [],
            "generator_value": []
        }

    def show_stats(self, skip_first=0, agents_to_show=32):
        # TODO: Dedicated callback for statistics
        plt.figure(figsize=(16, 14))
        plt.subplot(2, 1, 1)
        plt.title(r'Value, $x_t$')
        plt.plot(self.stats["generator_value"][skip_first:], 'b', label='Conceived value')
        agent_value = np.array(self.stats["agent_value"])[skip_first:, :agents_to_show]
        plt.ylim((
            np.min(self.stats["generator_value"]) - 3,
            np.max(self.stats["generator_value"]) + 3
        ))
        plt.plot(agent_value, 'r--', label='Estimated value')

        agent_step = np.array(self.stats["agent_step"])[skip_first:, :agents_to_show]
        plt.subplot(2, 1, 2)
        plt.title(r'Step, $\delta_t$')
        plt.plot(agent_step)

    def play_game(self, max_age: int = 300) -> np.ndarray:
        self.stats["agent_value"].clear()
        self.stats["agent_step"].clear()
        self.stats["generator_value"].clear()
        for age in range(max_age):
            feedback = self.environment.assess_the_guess(self.last_guess)
            self.last_guess = self.agent.react(feedback)
            self.stats["agent_value"].append(self.agent.current_value.copy())
            self.stats["agent_step"].append(self.agent.step.copy())
            self.stats["generator_value"].append(self.environment.value)
        return self.agent.current_value

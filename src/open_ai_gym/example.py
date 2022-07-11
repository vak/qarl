import gym
import numpy as np

from src.agent import Agent


def policy(agent: Agent, observation: np.ndarray):
    agent.adapt(observation)
    value = agent.current_value
    return value.reshape((1,))


env = gym.make('skippy/env')
agent = Agent(
    acceleration=np.ones(1) * 1.7897121877252560,
    braking=np.ones(1) * -0.0974316650691893,
    min_step=1e-1
)
is_done = False
observation = env.reset()
for step in range(1000):
    env.render()
    action = policy(agent, observation)
    observation, reward, done, info = env.step(action)
    if done and not is_done:
        is_done = True
        print(f"Done. Step {step}")

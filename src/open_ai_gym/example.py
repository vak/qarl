import gym
import numpy as np
import torch

from src.agent import Agent


def policy(agent: Agent, observation: np.ndarray):
    agent.adapt(observation)
    value = agent.current_value
    return value.reshape((1,))


env = gym.make('skippy/env')
k = 4
m = 0.1
a = 2
b = - 1 / (a + m) ** k
print(a, b)
agent = Agent(
    acceleration=torch.ones(1) * a,
    braking=torch.ones(1) * b,
    initial_step=torch.ones(1) * 4,
    min_step=1e-10
)
is_done = False
observation = env.reset()
for step in range(10000):
    env.render()
    action = policy(agent, observation)
    observation, reward, done, info = env.step(action)
    if done and not is_done:
        is_done = True
        print(f"Done. Step {step}")

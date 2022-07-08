from src.agent import AbstractAgent
from src.environment import AbstractEnvironment
from src.game.abstract_game import AbstractGame, GameState
from src.oracle.abstract_oracle import AbstractOracle


class Game(AbstractGame):
    def __init__(self, agent: AbstractAgent, environment: AbstractEnvironment, oracle: AbstractOracle):
        self.agent = agent
        self.environment = environment
        self.oracle = oracle

    def run_iteration(self) -> GameState:
        environment_state = self.environment.get_next_state()
        agent_state = self.agent.get_current_state()
        oracle_feedback = self.oracle.assess_the_guess(environment_state, agent_state)
        self.agent.adapt(oracle_feedback)
        return {
            'agent_state': self.agent.get_current_state(),
            'agent_step': self.agent.get_current_step(),
            'environment_state': environment_state,
        }

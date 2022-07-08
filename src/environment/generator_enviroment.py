from typing import Generator

from .abstact_enviroment import AbstractEnvironment


class GeneratorEnvironment(AbstractEnvironment):
    def __init__(self, number_generator: Generator[float, None, None]):
        self.generator = number_generator
        self.value = next(self.generator)

    def get_next_state(self) -> float:
        self.value = next(self.generator)
        return self.value

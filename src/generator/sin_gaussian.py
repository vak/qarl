from typing import Generator

import numpy as np


def sin_gaussian(step: float = np.pi / 32, mean: float = 0, sigma: float = 1) -> Generator[float, None, None]:
    x = 0
    while True:
        x += step
        yield np.sin(x) * sigma * 2 + np.random.normal(mean, sigma)

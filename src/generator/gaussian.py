from typing import Generator

import numpy as np


def gaussian(mean: float = 0, sigma: float = 1) -> Generator[float, None, None]:
    while True:
        yield np.random.normal(mean, sigma)

from typing import Generator

import numpy as np


def sin(step: float = np.pi / 10) -> Generator[float, None, None]:
    x = 0
    while True:
        x += step
        yield np.sin(x)

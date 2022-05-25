from typing import Generator

import numpy as np


def gaussian_mixture(mean: np.ndarray | None = None, sigma: np.ndarray | None = None, weight: np.ndarray | None = None) \
        -> Generator[float, None, None]:
    if weight is None:
        weight = [0.5, 0.5]
    if sigma is None:
        sigma = [1] * len(weight)
    if mean is None:
        mean = np.linspace(-5, 5, len(sigma))
    while True:
        yield np.random.choice(
            [
                np.random.normal(m, s)
                for m, s in zip(mean, sigma)
            ],
            p=weight
        )

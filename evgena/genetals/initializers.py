import numpy as np

from .core import InitializerBase
from typing import Sequence


class RandomBitStringInit(InitializerBase):
    def __init__(self, individual_size: int) -> None:
        super(RandomBitStringInit, self).__init__()

        self._individual_size = individual_size

    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        return np.random.choice(a=[False, True], size=(population_size, self._individual_size))


class RandomStdInit(InitializerBase):
    def __init__(self, individual_shape: Sequence[int], sigma: float = 1, mu: np.ndarray = 0):
        super(RandomStdInit, self).__init__()

        self._individual_shape = individual_shape
        self._sigma = sigma
        self._mu = mu

    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        return self._mu + (self._sigma * np.random.random((population_size,) + tuple(self._individual_shape)))

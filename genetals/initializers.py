from typing import Sequence, Callable

import numpy as np

from .core import InitializerBase


class RandomBitStringInit(InitializerBase):
    def __init__(self, individual_size: int) -> None:
        super(RandomBitStringInit, self).__init__()

        self._individual_size = individual_size

    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        return np.random.choice(a=[False, True], size=(population_size, self._individual_size))


class RandomStdInit(InitializerBase):
    def __init__(self,
        individual_shape: Sequence[int],
        sigma: np.ndarray = 1, mu: np.ndarray = 0
    ):
        super(RandomStdInit, self).__init__()
        
        self._individual_shape = individual_shape
        self._sigma = sigma
        self._mu = mu
    
    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        sigma = kwargs.get('sigma', self._sigma)
        mu = kwargs.get('mu', self._mu)
        
        individuals = np.random.standard_normal((population_size,) + tuple(self._individual_shape))
        
        return mu + (sigma * individuals)


class RandomUniformInit(InitializerBase):
    def __init(self,
        individual_shape: Sequence[int],
        mu: np.ndarray = 0, semi_range: np.ndarray = 1
    ):
        super(RandomUniformInit, self).__init__()
        
        self._individual_shape = individual_shape
        self._mu = mu
        self._semi_range = semi_range
        
    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        mu = kwargs.get('mu', self._mu)
        semi_range = kwargs.get('semi_range', self._semi_range)
        
        individuals = np.random.random_sample((population_size,) + tuple(self._individual_shape))
        
        return self.mu + (self.semi_range * (2 * individuals - 1))

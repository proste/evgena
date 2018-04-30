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
        
        genes = np.random.standard_normal((population_size,) + tuple(self._individual_shape))
        
        return mu + (sigma * genes)


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
        
        genes = np.random.random_sample((population_size,) + tuple(self._individual_shape))
        
        return self.mu + (self.semi_range * (2 * genes - 1))

class MultivariateRandomInit(InitializerBase):
    @classmethod
    def normal(cls,
        individual_shape: Sequence[int],
        sigmas: np.ndarray = 1, mus: np.ndarray = 0
    ) -> 'MultivariateRandomInit':
        return cls(individual_shape, np.random.standard_normal, sigmas, mus)
    
    @classmethod
    def uniform(cls,
        individual_shape: Sequence[int],
        scales: np.ndarray = 1, mus: np.ndarray = 0
    ) -> 'MultivariateRandomInit':
        return cls(
            individual_shape, (lambda shape: 2 * np.random.random_sample(shape) - 1), scales, mus
    )
    
    def __init__(self,
        individual_shape: Sequence[int],
        random_generator: Callable[[Sequence[int]], np.ndarray],
        scales: np.ndarray = 1, mus: np.ndarray = 0,
        
    ):
        super(MultivariateRandomInit, self).__init__()
        
        self._individual_shape = individual_shape
        self._scales = np.asarray(scales)
        self._mus = np.asarray(mus)
        self._random_generator = random_generator

    def _broadcast_momentum(self, population_size, momentum) -> np.ndarray:
        momentum = np.asarray(momentum)
        
        if momentum.ndim > 0:  # not scalar
            tile_count = (population_size + (len(momentum) - 1)) // len(momentum)
            momentum = np.tile(momentum, tile_count)[:population_size]
            momentum = momentum.reshape(
                [-1] + [1] * (len(self._individual_shape) - (momentum.ndim - 1)) + list(momentum.shape[1:])
            )
            
        return momentum
    
    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        scales = kwargs.get('scales', self._scales)
        mus = kwargs.get('mus', self._mus)
        
        scales = self._broadcast_momentum(population_size, scales)
        mus = self._broadcast_momentum(population_size, mus)
        
        genes = self._random_generator((population_size,) + tuple(self._individual_shape))
        
        return mus + (scales * genes)

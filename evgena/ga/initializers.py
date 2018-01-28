import numpy as np

from .core import InitializerBase


class RandomBitStringInit(InitializerBase):
    def __init__(self, individual_size: int) -> None:
        super().__init__()

        self._individual_size = individual_size

    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        return np.random.choice(a=[False, True], size=(population_size, self._individual_size))

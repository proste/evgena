import numpy as np

from .core import ObjectiveFncBase


class AllOneObjective(ObjectiveFncBase):
    def __call__(self, individuals: np.ndarray) -> np.ndarray:
        return individuals.sum(axis=1)

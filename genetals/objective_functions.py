import numpy as np

from .core import ObjectiveFncBase


class AllOneObjective(ObjectiveFncBase):
    def __call__(self, genes: np.ndarray) -> np.ndarray:
        return genes.sum(axis=1)

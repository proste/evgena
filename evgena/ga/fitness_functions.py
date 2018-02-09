import numpy as np

from .core import FitnessFncBase


class NormalizingFitness(FitnessFncBase):
    def __call__(self, individuals: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        return objectives / np.sum(objectives)

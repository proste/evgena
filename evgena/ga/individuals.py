# TODO consider numpy array individuum and how to implement it in DEAP
from typing import NewType
import numpy as np

AnyIndividuals = NewType('AnyIndividual', np.ndarray)

Strings = NewType('Strings', AnyIndividuals)

from typing import NewType
import numpy as np


# --- individuals ---
SomeIndividuals = np.ndarray
AnyIndividuals = np.ndarray

BitStrings = NewType('Strings', SomeIndividuals)

# --- objectives ---
SomeObjectives = np.ndarray
AnyObjectives = np.ndarray


# --- fitnesses ---

SomeFitnesses = np.ndarray
AnyFitnesses = np.ndarray

SimpleFloatFitnesses = NewType('SimpleFloatFitnesses', SomeFitnesses)

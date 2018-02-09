import numpy as np

from .core import GeneticAlgorithm, CallbackBase


class BestReport(CallbackBase):
    def __call__(self, ga: GeneticAlgorithm) -> None:
        offspring = ga.capture(-1)

        print('{gen_i:3d}: {ind}'.format(
            gen_i=ga.current_generation,
            ind=offspring.individuals[offspring.fitnesses.argmax()]
        ))

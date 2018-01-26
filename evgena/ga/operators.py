from .individuals import Strings
from .core import Population, GeneticAlgorithm, OperatorBase
from typing import Generic

import numpy as np


class OnePointXover(OperatorBase[Strings, None, None]):
    def __init__(self, parents_in):
        super(OnePointXover, self).__init__(parents_in)

    def _operation(
            self, ga, *input_populations
    ):
        parents = parents_pop.individuals
        offspring = parents.copy()

        for p_i in range(0, len(parents), 2):
            edge = np.random.randint(parents.shape[1])

            offspring[p_i, edge:, ...] = parents[p_i + 1, edge:, ...]
            offspring[p_i + 1, edge:, ...] = parents[p_i, edge:, ...]

        return Population(offspring, )

# TODO maybe override [] in population to [] in individuals (maybe not - shape and dtype would not make sense)
# TODO maybe operator as a class (with overridable call)
def one_point_xover(
        ga: GeneticAlgorithm[Strings, None, None],
        parents_pop: Population[Strings, None, None]
) -> Population[Strings, None, None]:
    # TODO vectorize
    # TODO check calling convention


def domain_mutation(
        ga: GeneticAlgorithm[Strings, None, None],
        parents_pop: Population[Strings, None, None]
) -> Population[Strings, None, None]:
    parents = parents_pop.individuals
    offspring = parents.copy()

    for p_i in range(len(parents)):
        if ()
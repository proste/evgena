from .core import Population, GeneticAlgorithm, OperatorBase
from typing import Tuple

import numpy as np


class RouletteWheelSelection(OperatorBase):
    def __init__(self, unfiltered_op: OperatorBase):
        super(RouletteWheelSelection, self).__init__(unfiltered_op)

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        unfiltered = input_populations[0]

        choice = np.random.choice(np.arange(unfiltered.size), size=unfiltered.size, p=unfiltered.fitnesses)

        return Population(unfiltered.individuals[choice, ...], ga)


class OnePointXover(OperatorBase):
    def __init__(self, xover_prob: float, parents_op: OperatorBase):
        super(OnePointXover, self).__init__(parents_op)

        self.xover_prob = xover_prob

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        parents = input_populations[0].individuals
        offspring = parents.copy()

        for p_i in range(0, len(parents), 2):
            edge = np.random.randint(parents.shape[1])

            offspring[p_i, edge:, ...] = parents[p_i + 1, edge:, ...]
            offspring[p_i + 1, edge:, ...] = parents[p_i, edge:, ...]

        return Population(offspring, ga)


class FlipMutation(OperatorBase):
    def __init__(self, mut_prob: float, gene_mut_prob: float, original_op: OperatorBase):
        super(FlipMutation, self).__init__(original_op)

        self._mut_prob = mut_prob
        self._gene_mut_prob = gene_mut_prob

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        originals = input_populations[0].individuals
        mutated = originals.copy()

        # TODO try to vectorize - maybe some bit mask?
        for p_i in range(len(originals)):
            if np.random.random() < self._mut_prob:
                mask = np.random.choice(
                    a=[False, True], size=originals.shape[1:], p=[1-self._gene_mut_prob, self._gene_mut_prob]
                )

                mutated[p_i, mask, ...] = np.invert(originals[p_i, mask, ...])

        return Population(mutated, ga)


class Elitism(OperatorBase):
    def __init__(self, elite_proportion: float, original_op: OperatorBase, evolved_op: OperatorBase):
        super(Elitism, self).__init__(original_op, evolved_op)

        self._elite_proportion = elite_proportion

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population) -> Population:
        elite_size = int(ga.population_size * self._elite_proportion)
        original = input_populations[0]
        evolved = input_populations[1]
        result = np.empty_like(original.individuals)

        # assign all but elite_size with random evolved individuals
        result[elite_size:] = evolved.individuals[
            np.random.choice(np.arange(evolved.size), evolved.size - elite_size, replace=False)
        ]

        # assign elite_size individuals with best of original individuals
        result[:elite_size] = original.individuals[original.fitnesses.argsort()[-elite_size:]]

        return Population(result, ga)

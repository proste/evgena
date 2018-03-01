import numpy as np

from .core import Population, GeneticAlgorithm, OperatorBase
from typing import Sequence, Union


class RouletteWheelSelection(OperatorBase):
    def __init__(self, unfiltered_op: OperatorBase):
        super(RouletteWheelSelection, self).__init__(unfiltered_op)

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        unfiltered = input_populations[0]

        choice = np.random.choice(
            np.arange(unfiltered.size), size=unfiltered.size,
            p=(unfiltered.fitnesses / unfiltered.fitness.sum)
        )

        return Population(unfiltered.individuals[choice, ...], ga)


class TournamentSelection(OperatorBase):
    def __init__(self, unfiltered_op: OperatorBase, match_size: int):
        super(TournamentSelection, self).__init__(unfiltered_op)

        self._match_size = match_size

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        unfiltered = input_populations[0]

        match_board = np.empty((unfiltered.size, self._match_size), dtype=np.int)
        for i in range(unfiltered.size):
            match_board[i] = np.random.choice(unfiltered.size, size=self._match_size)

        scores = unfiltered.fitnesses[match_board]
        chosen = match_board[range(unfiltered.size), scores.argmax(axis=-1)]

        return Population(unfiltered.individuals[chosen], ga)


class OnePointXover(OperatorBase):
    def __init__(self, parents_op: OperatorBase, xover_prob: float):
        super(OnePointXover, self).__init__(parents_op)

        self.xover_prob = xover_prob

    # TODO maybe change
    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        parents = input_populations[0].individuals
        offspring = parents.copy()

        for p_i in range(0, len(parents), 2):
            edge = np.random.randint(parents.shape[1])

            offspring[p_i, edge:, ...] = parents[p_i + 1, edge:, ...]
            offspring[p_i + 1, edge:, ...] = parents[p_i, edge:, ...]

        return Population(offspring, ga)


class FlipMutation(OperatorBase):
    def __init__(self, original_op: OperatorBase, mut_prob: float, gene_mut_prob: float):
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


class BiasedMutation(OperatorBase):
    def __init__(
            self, original_op, individual_mut_ratio=0.1, gene_mut_ratio=0.05,
            sigma=1, mu=0, l_bound: float = None, u_bound: float = None
    ):
        super(BiasedMutation, self).__init__(original_op)

        self._individual_mut_ratio = individual_mut_ratio
        self._gene_mut_ratio = gene_mut_ratio
        self._sigma = sigma
        self._mu = mu
        self._l_bound = l_bound
        self._u_bound = u_bound

    def _operation(self, ga, *input_populations):
        population = input_populations[0]
        individuals = population.individuals

        mutant_count = int(population.size * self._individual_mut_ratio)
        individual_size = np.prod(individuals.shape[1:])
        gene_count = int(individual_size * self._gene_mut_ratio)

        mutant_index = np.random.choice(population.size, size=mutant_count, replace=False)

        gene_index = np.empty(shape=(mutant_count, gene_count), dtype=np.int)
        for row in range(mutant_count):
            gene_index[row] = np.random.choice(individual_size, size=gene_count, replace=False)

        flat_index = (gene_index + np.expand_dims(mutant_index * individual_size, -1)).ravel()
        shift = self._sigma * np.random.standard_normal(flat_index.size) + self._mu

        mutated = individuals.copy()
        mutated.ravel()[flat_index] += shift
        np.clip(mutated, self._l_bound, self._u_bound, out=mutated)

        return Population(mutated, ga)


class TwoPointXover(OperatorBase):
    def __init__(self, parents_op: OperatorBase, xover_prob: float, axis: Union[int, Sequence[int]] = None):
        super(TwoPointXover, self).__init__(parents_op)

        self._xover_prob = xover_prob
        self._axis = axis if (not isinstance(axis, int)) else (axis,)

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population) -> Population:
        parents = input_populations[0].individuals
        offspring = parents.copy()
        individual_shape = offspring.shape[1:]

        for p_i in range(0, len(parents), 2):
            hyper_cube_bounds = [
                slice(0, u_bound) if ((self._axis is not None) and (i not in self._axis))
                else slice(*(np.sort(np.random.randint(0, u_bound, size=2))))
                for i, u_bound in enumerate(individual_shape)
            ]

            offspring[[p_i] + hyper_cube_bounds] = parents[[p_i + 1] + hyper_cube_bounds]
            offspring[[p_i + 1] + hyper_cube_bounds] = parents[[p_i] + hyper_cube_bounds]

        return Population(offspring, ga)


class Elitism(OperatorBase):
    def __init__(self, original_op: OperatorBase, evolved_op: OperatorBase, elite_proportion: float):
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

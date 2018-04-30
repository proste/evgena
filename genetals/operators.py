import numpy as np

from .core import Population, GeneticAlgorithm, OperatorBase
from typing import Sequence, Union


class RouletteWheelSelection(OperatorBase):
    def __init__(self, unfiltered_op: OperatorBase):
        super(RouletteWheelSelection, self).__init__(unfiltered_op)

    def _operation(self, ga: GeneticAlgorithm, unfiltered: Population):
        choice = np.random.choice(
            np.arange(unfiltered.size), size=unfiltered.size,
            p=(unfiltered.fitnesses / unfiltered.fitnesses.sum)
        )

        return Population(unfiltered.genes[choice, ...], ga)


class TournamentSelection(OperatorBase):
    def __init__(self, unfiltered_op: OperatorBase, match_size: int):
        super(TournamentSelection, self).__init__(unfiltered_op)

        self._match_size = match_size

    def _operation(self, ga: GeneticAlgorithm, unfiltered: Population):
        match_board = np.empty((unfiltered.size, self._match_size), dtype=np.int)
        for i in range(unfiltered.size):
            match_board[i] = np.random.choice(unfiltered.size, size=self._match_size)

        scores = unfiltered.fitnesses[match_board]
        chosen = match_board[range(unfiltered.size), scores.argmax(axis=-1)]

        return Population(unfiltered.genes[chosen], ga)


class OnePointXover(OperatorBase):
    def __init__(self, parents_op: OperatorBase, xover_prob: float):
        super(OnePointXover, self).__init__(parents_op)

        self.xover_prob = xover_prob

    # TODO maybe change
    def _operation(self, ga: GeneticAlgorithm, parents: Population):
        offspring_genes = parents.genes.copy()

        for p_i in range(0, len(parents), 2):
            edge = np.random.randint(parents.genes.shape[1])

            offspring_genes[p_i, edge:, ...] = parents.genes[p_i + 1, edge:, ...]
            offspring_genes[p_i + 1, edge:, ...] = parents.genes[p_i, edge:, ...]

        return Population(offspring_genes, ga)


class FlipMutation(OperatorBase):
    def __init__(self, original_op: OperatorBase, mut_prob: float, gene_mut_prob: float):
        super(FlipMutation, self).__init__(original_op)

        self._mut_prob = mut_prob
        self._gene_mut_prob = gene_mut_prob

    def _operation(self, ga: GeneticAlgorithm, original: Population):
        mutated_genes = originals.copy()

        # TODO try to vectorize - maybe some bit mask?
        for p_i in range(original.size):
            if np.random.random() < self._mut_prob:
                mask = np.random.choice(
                    a=[False, True], size=originals.genes.shape[1:],
                    p=[(1 - self._gene_mut_prob), self._gene_mut_prob]
                )

                mutated_genes[p_i, mask, ...] = np.invert(originals[p_i, mask, ...])

        return Population(mutated_genes, ga)


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

    def _operation(self, ga: GeneticAlgorithm, original: Population):
        mutant_count = int(original.size * self._individual_mut_ratio)
        individual_size = np.prod(original.genes.shape[1:])
        gene_count = int(individual_size * self._gene_mut_ratio)

        mutant_index = np.random.choice(original.size, size=mutant_count, replace=False)

        gene_index = np.empty(shape=(mutant_count, gene_count), dtype=np.int)
        for row in range(mutant_count):
            gene_index[row] = np.random.choice(individual_size, size=gene_count, replace=False)

        flat_index = (gene_index + np.expand_dims(mutant_index * individual_size, -1)).ravel()
        shift = self._sigma * np.random.standard_normal(flat_index.size) + self._mu

        mutated_genes = original.genes.copy()
        mutated_genes.ravel()[flat_index] += shift
        np.clip(mutated_genes, self._l_bound, self._u_bound, out=mutated_genes)

        return Population(mutated_genes, ga)


class TwoPointXover(OperatorBase):
    def __init__(self, parents_op: OperatorBase, xover_prob: float, axis: Union[int, Sequence[int]] = None):
        super(TwoPointXover, self).__init__(parents_op)

        self._xover_prob = xover_prob
        self._axis = axis if (not isinstance(axis, int)) else (axis,)

    def _operation(self, ga: GeneticAlgorithm, parents: Population) -> Population:
        offspring_genes = parents.genes.copy()
        individual_shape = offspring_genes.shape[1:]

        for p_i in range(0, parents.size, 2):
            hyper_cube_bounds = [
                slice(0, u_bound) if ((self._axis is not None) and (i not in self._axis))
                else slice(*(np.sort(np.random.randint(0, u_bound, size=2))))
                for i, u_bound in enumerate(individual_shape)
            ]

            offspring_genes[[p_i] + hyper_cube_bounds] = parents.genes[[p_i + 1] + hyper_cube_bounds]
            offspring_genes[[p_i + 1] + hyper_cube_bounds] = parents.genes[[p_i] + hyper_cube_bounds]

        return Population(offspring_genes, ga)


class ShuffleOperator(OperatorBase):
    def __init__(self, input_op: OperatorBase):
        super(ShuffleOperator, self).__init__(input_op)

    def _operation(self, ga: GeneticAlgorithm, unshuffled: Population):
        shuffled_genes = np.random.permutation(unshuffled.genes)

        return Population(shuffled_genes, ga)


class NSGAOperator(OperatorBase):
    def __init__(self, parents_op: OperatorBase, offspring_op: OperatorBase):
        super(NSGAOperator, self).__init__(parents_op, offspring_op)

    def _operation(self, ga: GeneticAlgorithm, parents: Population, offspring: Population):
        # merge parents with offspring
        genes = np.concatenate((parents.genes, offspring.genes))
        fitnesses = np.concatenate((parents.fitnesses, offspring.fitnesses))
        pool_size = len(genes)
        pop_size = parents.size

        fitnesses_count = fitnesses.shape[1]
        fitnesses_rank_to_index = np.argsort(fitnesses, axis=0)
        fitnesses_index_to_rank = np.argsort(fitnesses_rank_to_index, axis=0)

        # find dominating and dominated
        subs_indices = [None] * (pool_size)  # indices of individuals one is dominating
        doms_count = np.empty(shape=pool_size, dtype=np.int)  # number of individuals one is dominated by

        for i_i in range(pool_size):
            # iteratively build dominating and dominated
            i_rank = fitnesses_index_to_rank[i_i, 0]
            subs = fitnesses_rank_to_index[:i_rank, 0]
            doms = fitnesses_rank_to_index[i_rank + 1:, 0]
            for f_i in range(1, fitnesses_count):
                i_rank = fitnesses_index_to_rank[i_i, f_i]
                subs = np.intersect1d(subs, fitnesses_rank_to_index[:i_rank, f_i], assume_unique=True)
                doms = np.intersect1d(doms, fitnesses_rank_to_index[i_rank + 1:, f_i], assume_unique=True)

            # assign dominating and dominated
            subs_indices[i_i] = subs
            doms_count[i_i] = len(doms)

        # build non-dominated fronts
        fronts = []  # list of fronts (indices of individuals per front)
        choice_size = 0  # number of individuals chosen so far
        while True:
            # fetch non-dominated solutions
            front_indices, *_ = np.where(doms_count == 0)

            # keep front indices and update count of individuals chosen so far
            fronts.append(front_indices)
            choice_size += len(front_indices)

            if choice_size >= pop_size:
                break

            # invalidate individuals for future loop executions
            doms_count[front_indices] = -1

            # remove current front
            for i_i in range(len(front_indices)):
                doms_count[subs_indices[front_indices[i_i]]] -= 1

        # secondary sorting
        if choice_size > pop_size:
            # crowding-distance
            last_front = fronts[-1]

            normalised_distances = np.empty((len(last_front), fitnesses_count))
            for f_i in range(fitnesses_count):
                front_rank_to_index = np.argsort(fitnesses_index_to_rank[last_front, f_i])
                sorted_front = last_front[front_rank_to_index]

                dim_distances = fitnesses[sorted_front[2:], f_i] - fitnesses[sorted_front[:-2], f_i]
                dim_range = fitnesses[sorted_front[-1], f_i] - fitnesses[sorted_front[0], f_i]
                normalised_distances[front_rank_to_index[1:-1], f_i] = dim_distances / dim_range
                normalised_distances[front_rank_to_index[[0, -1]], f_i] = np.inf

            averaged_distances = np.average(normalised_distances, axis=1)

            choice = np.concatenate(
                fronts[:-1] + [last_front[np.argsort(averaged_distances)[(choice_size - pop_size):]]])
        else:
            choice = np.concatenate(fronts)

        return Population(genes[choice], ga)


class Elitism(OperatorBase):
    def __init__(self, parents_op: OperatorBase, offspring_op: OperatorBase, elite_proportion: float):
        super(Elitism, self).__init__(parents_op, offspring_op)

        self._elite_proportion = elite_proportion

    def _operation(self, ga: GeneticAlgorithm, parents: Population, offspring: Population) -> Population:
        elite_size = int(ga.population_size * self._elite_proportion)
        genes = np.empty_like(parents.genes)

        # assign all but elite_size with random evolved individuals
        genes[elite_size:] = offspring.genes[
            np.random.choice(np.arange(offspring.size), offspring.size - elite_size, replace=False)
        ]

        # assign elite_size individuals with best of original individuals
        genes[:elite_size] = parents.genes[parents.fitnesses.argsort()[-elite_size:]]

        return Population(genes, ga)

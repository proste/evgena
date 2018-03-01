from genetals.core import *
from genetals.operators import *
from genetals.callbacks import BestReport

import numpy as np
import matplotlib.pyplot as plt


class BestImgReport(CallbackBase):
    def __init__(self):
        super(BestImgReport, self).__init__()

        self._fig, self._ax = plt.subplots(1, 1)

    def __call__(self, ga: GeneticAlgorithm) -> None:
        offspring = ga.capture(-1)

        self._ax.imshow(offspring.individuals[offspring.fitnesses.argmax()])

        self._fig.canvas.draw()


class RandomFloat(InitializerBase):
    def __init__(self, shape):
        super(InitializerBase, self).__init__()

        self._shape = shape

    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        return np.random.rand(population_size, *self._shape)


class StdMutation(OperatorBase):
    def __init__(self, original_op, mut_prob, gene_mut_prob, sigma):
        super(StdMutation, self).__init__(original_op)

        self._mut_prob = mut_prob
        self._gene_mut_prob = gene_mut_prob
        self._sigma = sigma

    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population):
        originals = input_populations[0].individuals
        mutated = originals.copy()
        noise = np.random.randn(*mutated.shape) * self._sigma

        for p_i in range(len(originals)):
            if np.random.random() < self._mut_prob:
                mask = np.random.choice(
                    a=[0, 1], size=originals.shape[1:], p=[1 - self._gene_mut_prob, self._gene_mut_prob]
                )

                mutated[p_i, mask] += noise[p_i, mask]
                mutated[mutated < 0] = 0
                mutated[mutated > 1] = 1

        return Population(mutated, ga)


class SumObjective(ObjectiveFncBase):
    def __call__(self, individuals: np.ndarray) -> np.ndarray:
        return individuals.sum(axis=tuple([i for i in range(1, individuals.ndim)]))


class NormalizingFitness(FitnessFncBase):
    def __call__(self, individuals: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        return objectives / np.sum(objectives)


builder = OperatorGraph()

selection = TournamentSelection(builder.init_op, 4)
# selection = RouletteWheelSelection(builder.init_op)
xover = TwoPointXover(selection, 0.8)
mutation = StdMutation(xover, 0.1, 0.05, 0.5)
elitism = Elitism(builder.init_op, mutation, 0.1)

ga = GeneticAlgorithm(
    RandomFloat((4, 4, 1)), builder.build_graph(), SumObjective(), NormalizingFitness(), callbacks=[BestReport()]
)

ga.run()

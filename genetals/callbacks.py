import numpy as np

from .core import GeneticAlgorithm, CallbackBase
import matplotlib.pyplot as plt


class GAStatus(CallbackBase):
    def __init__(self, fig: plt.Figure):
        super(GAStatus, self).__init__()

        self._fig = fig

    def __call__(self, ga: GeneticAlgorithm) -> None:
        self._fig.canvas.set_window_title('Current generation: {}'.format(ga.current_generation))
        self._fig.canvas.draw()


class BestReport(CallbackBase):
    def __call__(self, ga: GeneticAlgorithm) -> None:
        offspring = ga.capture(-1)

        print('{gen_i:3d}: {ind}'.format(
            gen_i=ga.current_generation,
            ind=offspring.individuals[offspring.fitnesses.argmax()]
        ))


class BestImgReport(CallbackBase):
    def __init__(self, ax: plt.Axes = None):
        super(BestImgReport, self).__init__()

        if ax is None:
            self._fig, self._ax = plt.subplots(1, 1)
        else:
            self._fig, self._ax = ax.figure, ax

    def __call__(self, ga: GeneticAlgorithm) -> None:
        offspring = ga.capture(-1)
        best_i = offspring.fitnesses.argmax()

        self._ax.imshow(offspring.individuals[best_i], cmap='gray')


class HistoryReport(CallbackBase):
    def __init__(self, fitness_axes: plt.Axes = None, *objective_axes: plt.Axes):
        super(HistoryReport, self).__init__()

        if fitness_axes is None:
            _, self._fitness_axes = plt.subplots(1, 1)
        else:
            self._fitness_axes = fitness_axes

        self._objective_axes = objective_axes if (len(objective_axes) > 0) else None

    def __call__(self, ga: GeneticAlgorithm):
        self._fitness_axes.clear()
        self._fitness_axes.plot(ga.fitness_history)
        self._fitness_axes.set_title(
            'Current best fitness: {:.6f}'.format(ga.fitness_history[-1])
        )

        if self._objective_axes is not None:
            for column in range(ga.objectives_history.shape[1]):
                curr_axes = self._objective_axes[column if (len(self._objective_axes) > 1) else 0]

                curr_axes.clear()
                curr_axes.plot(ga.objectives_history[:, column])


class PolishedHistoryReport(CallbackBase):
    def __init__(self, quantile: float = None, fitness_axes: plt.Axes = None, *objective_axes: plt.Axes):
        super(HistoryReport, self).__init__()

        self._quantile_fitness = None
        self._quantile_objective = None
        self._quantile = quantile

        if fitness_axes is None:
            _, self._fitness_axes = plt.subplots(1, 1)
        else:
            self._fitness_axes = fitness_axes

        self._objective_axes = objective_axes if (len(objective_axes) > 0) else None

    def __call__(self, ga: GeneticAlgorithm):
        if self._quantile is not None:
            if ga.current_generation == 0:
                self._quantile_fitness = np.empty(shape=ga.generation_cap, dtype=np.float)

                if self._objective_axes is not None:
                    if (
                        (len(self._objective_axes) > 1) and
                        (len(self._objective_axes != ga.objectives_history.shape[1]))
                    ):
                        raise ValueError('Objectives count does not match objective axes provided in constructor')

                    self._quantile_objective = np.empty(
                        shape=(ga.generation_cap,) + ga.objectives_history.shape[1:],
                        dtype=ga.objectives_history.dtype
                    )

            offspring = ga.capture(-1)
            quantile_i = np.argpartition(offspring.fitnesses, int(self._quantile * offspring.size + 0.5))

            self._quantile_fitness[ga.current_generation] = offspring.fitnesses[quantile_i]

            if self._objective_axes is not None:
                self._quantile_objective[ga.current_generation] = offspring.objectives[quantile_i]

        self._fitness_axes.clear()
        self._fitness_axes.plot(ga.fitness_history)
        if self._quantile is not None:
            plt.fill_between(
                range(ga.current_generation), ga.fitness_history,
                self._quantile_fitness[:ga.current_generation], alpha=0.5
            )

        if self._objective_axes is not None:
            for column in range(ga.objectives_history.shape[1]):
                curr_axes = self._objective_axes[column if (len(self._objective_axes) > 1) else 0]

                curr_axes.clear()
                curr_axes.plot(ga.objectives_history[:, column])
                curr_axes.fill_between(
                    range(ga.current_generation), ga.objectives_history[:, column],
                    self._quantile_objective[:ga.current_generation, column], alpha=0.5
                )

import numpy as np

from typing import List, Callable, Tuple
from abc import ABC, abstractmethod


class FitnessFncBase(ABC):
    @abstractmethod
    def __call__(self, individuals: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ObjectiveFncBase(ABC):
    @abstractmethod
    def __call__(self, individuals: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EarlyStoppingBase(ABC):
    @abstractmethod
    def __call__(self, ga: 'GeneticAlgorithm') -> bool:
        raise NotImplementedError


class CallbackBase(ABC):
    @abstractmethod
    def __call__(self, ga: 'GeneticAlgorithm') -> None:
        raise NotImplementedError


class InitializerBase(ABC):
    @abstractmethod
    def __call__(self, population_size: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class OperatorBase:
    @property
    def op_id(self) -> int:
        return self._op_id

    def __init__(self, *input_ops: 'OperatorBase', graph_builder: 'OperatorGraph' = None):
        if len(input_ops) == 0:  # dummy operation
            self._graph_builder = graph_builder
            self._op_id = -1
            self._input_ids = None
        else:
            self._graph_builder = input_ops[0]._graph_builder

            for input_op in input_ops:
                if self._graph_builder is not input_op._graph_builder:
                    raise ValueError('Operations do not belong to one graph')

            self._op_id = self._graph_builder._add_operator(self)
            self._input_ids = [input_op.op_id for input_op in input_ops]

            for input_id in self._input_ids:
                if input_id >= self._op_id:
                    raise ValueError(
                        'Input (id: {0!r}) will be undefined at the time of processing this operation (id: {1!r})'
                        .format(input_id, self._op_id)
                    )

    @abstractmethod
    def _operation(
            self, ga: 'GeneticAlgorithm', *input_populations: 'Population') -> 'Population':
        raise NotImplementedError

    def __call__(self, ga: 'GeneticAlgorithm') -> 'Population':
        return self._operation(ga, *(ga.capture(input_id) for input_id in self._input_ids))


class Population:
    """Immutable population of individuals (with fitnesses and objectives)"""

    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def individuals(self) -> np.ndarray:
        return self._individuals

    @property
    def objectives(self) -> np.ndarray:
        self._evaluate_objective()
        return self._objective

    @property
    def fitnesses(self) -> np.ndarray:
        self._evaluate_fitness()
        return self._fitness

    def __init__(self, individuals: np.ndarray, ga: 'GeneticAlgorithm'):
        # define individuals and make them read only
        if individuals.flags['OWNDATA']:
            self._individuals = individuals
        else:
            self._individuals = individuals.copy()

        self._individuals.flags['WRITEABLE'] = False

        # define ga to which this pop belong
        self._ga = ga

        # initialize objective and fitness value tables
        self._objective = None
        self._fitness = None

    def _evaluate_objective(self) -> None:
        if self._objective is None:
            self._objective = self._ga.objective_fnc(self._individuals)
            self._objective.flags['WRITEABLE'] = False

    def _evaluate_fitness(self) -> None:
        if self._fitness is None:
            self._evaluate_objective()

            self._fitness = self._ga.fitness_fnc(self._individuals, self._objective)
            self._fitness.flags['WRITEABLE'] = False


class OperatorGraph:
    @property
    def init_op(self) -> OperatorBase:
        return self._init_op

    def __init__(self):
        self._is_built = False
        self._operators = []
        self._init_op = OperatorBase(graph_builder=self)

    def _add_operator(self, operator: OperatorBase) -> int:
        if self._is_built:
            raise ValueError(
                'Operator addition forbidden on built (finalized) instance of {!r}'.format(self.__class__.__name__)
            )

        new_op_id = len(self._operators)
        self._operators.append(operator)

        return new_op_id

    def _build_graph(self) -> List[OperatorBase]:
        self._is_built = True

        return self._operators


# TODO add individual as some type - ie.
# TODO configuration load dump mechanism
# TODO some basic set of operators and tests
class GeneticAlgorithm:
    @property
    def objective_fnc(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._objective_fnc

    @property
    def fitness_fnc(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._fitness_fnc

    def operator(self, operator_id: int) -> OperatorBase:
        return self._operators[operator_id]

    def capture(self, operator_id: int) -> Population:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide captures')

        return self._captures[operator_id]

    @property
    def current_generation(self) -> int:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide current_generation')

        return self._curr_generation

    @property
    def population_size(self) -> int:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide population_size')

        return self._population_size

    @property
    def generation_cap(self) -> int:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide generation_cap')

        return self._generation_cap

    @property
    def objectives_history(self) -> np.ndarray:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide objectives_history')

        return self._best_objectives[:self._curr_generation + 1]

    @property
    def fitness_history(self) -> np.ndarray:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide fitness_history')

        return self._best_fitnesses[:self._curr_generation + 1]

    def __init__(
            self, initialization: InitializerBase, operator_graph: OperatorGraph,
            objective_fnc: ObjectiveFncBase, fitness_fnc: FitnessFncBase = None,
            early_stopping: EarlyStoppingBase = None, callbacks: List[CallbackBase] = None
    ):
        # GA's volatile fields (valid only when GA is running)
        self._is_running = False
        self._captures = None
        self._curr_generation = None
        self._population_size = None
        self._generation_cap = None
        self._best_fitnesses = None
        self._best_objectives = None

        # GA persistent fields
        self._initialization = initialization
        self._objective_fnc = objective_fnc
        self._fitness_fnc = fitness_fnc if (fitness_fnc is not None) else (lambda _, obj: obj)
        self._early_stopping = early_stopping
        self._operators = operator_graph._build_graph()

        self._callbacks = callbacks if (callbacks is not None) else []

    def run(
            self, population_size: int = 32, generation_cap: int = 64, *args, **kwargs
    ) -> Tuple[Population, np.ndarray, np.ndarray]:
        # init volatile fields
        self._is_running = True
        self._captures: List[Population] = [None] * len(self._operators)
        self._population_size = population_size
        self._generation_cap = generation_cap

        # run initialization with optional params
        init_individuals = self._initialization(self.population_size, *args, **kwargs)
        init_pop = Population(init_individuals, self)
        self._captures[-1] = init_pop

        # initialize journals
        self._best_fitnesses = np.empty(generation_cap, np.float)
        self._best_objectives = np.empty((generation_cap,) + init_pop.objectives.shape[1:], init_pop.objectives.dtype)

        # loop over generations
        for self._curr_generation in range(self.generation_cap):
            # handle early stopping
            if (self._early_stopping is not None) and self._early_stopping(self):
                break

            # loop over each operator
            for op in self._operators:
                self._captures[op.op_id] = op(self)

            # write to journals
            best_i = self._captures[-1].fitnesses.argmax()
            self._best_fitnesses[self._curr_generation] = self._captures[-1].fitnesses[best_i]
            self._best_objectives[self._curr_generation] = self._captures[-1].objectives[best_i]

            # handle callbacks
            for callback in self._callbacks:
                callback(self)

            # clear all captures but last
            self._captures[:-1] = [None] * (len(self._captures) - 1)

        # clean up
        self._is_running = False
        result, fitnesses, objectives = self._captures[-1], self._best_fitnesses, self._best_objectives
        self._captures = self._best_fitnesses = self._best_objectives = None

        return result, fitnesses, objectives

from os import makedirs
from datetime import datetime
from typing import List, Callable, Tuple
from abc import ABC, abstractmethod

import numpy as np


class FitnessFncBase(ABC):
    @abstractmethod
    def __call__(self, genes: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ObjectiveFncBase(ABC):
    @abstractmethod
    def __call__(self, genes: np.ndarray) -> np.ndarray:
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
        return len(self._genes)

    @property
    def genes(self) -> np.ndarray:
        return self._genes

    @property
    def objectives(self) -> np.ndarray:
        self._evaluate_objectives()
        return self._objectives

    @property
    def fitnesses(self) -> np.ndarray:
        self._evaluate_fitnesses()
        return self._fitnesses

    def __init__(self, genes: np.ndarray, ga: 'GeneticAlgorithm'):
        # define genes and make them read only
        if genes.flags['OWNDATA']:
            self._genes = genes
        else:
            self._genes = genes.copy()

        self._genes.flags['WRITEABLE'] = False

        # define ga to which this pop belongs
        self._ga = ga

        # initialize objective and fitness value tables
        self._objectives = None
        self._fitnesses = None

    def _evaluate_objectives(self) -> None:
        if self._objectives is None:
            self._objectives = self._ga.objective_fnc(self._genes)
            self._objectives.flags['WRITEABLE'] = False

    def _evaluate_fitnesses(self) -> None:
        if self._fitnesses is None:
            self._evaluate_objectives()

            self._fitnesses = self._ga.fitness_fnc(self._genes, self._objectives)
            self._fitnesses.flags['WRITEABLE'] = False


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

        return self._objectives_history[:self._curr_generation + 1]

    @property
    def fitness_history(self) -> np.ndarray:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide fitness_history')

        return self._fitnesses_history[:self._curr_generation + 1]

    def __init__(
        self, initializer: InitializerBase, operator_graph: OperatorGraph,
        objective_fnc: ObjectiveFncBase, fitness_fnc: FitnessFncBase = None,
        early_stopping: EarlyStoppingBase = None, callbacks: List[CallbackBase] = None,
        results_dir=None
    ):
        # GA's volatile fields (valid only when GA is running)
        self._is_running = False
        self._captures = None
        self._curr_generation = None
        self._population_size = None
        self._generation_cap = None
        self._fitnesses_history = None
        self._objectives_history = None

        # GA persistent fields
        self._initializer = initializer
        self._objective_fnc = objective_fnc
        self._fitness_fnc = fitness_fnc if (fitness_fnc is not None) else (lambda _, obj: obj)
        self._early_stopping = early_stopping
        self._operators = operator_graph._build_graph()
        self._callbacks = callbacks if (callbacks is not None) else []
        self._results_dir = results_dir
        
        if self._results_dir is not None:
            makedirs(self._results_dir, exist_ok=True)

    def _run_epoch(self) -> bool:
        # handle early stopping
        if (self._early_stopping is not None) and self._early_stopping(self):
            return False

        # loop over each operator
        for op in self._operators:
            self._captures[op.op_id] = op(self)

        # write to journals
        self._fitnesses_history[self._curr_generation] = self._captures[-1].fitnesses
        self._objectives_history[self._curr_generation] = self._captures[-1].objectives

        # handle callbacks
        for callback in self._callbacks:
            callback(self)

        # clear all captures but last
        self._captures[:-1] = [None] * (len(self._captures) - 1)
        
        return True
    
    def _save_state(self):
        if self._results_dir is not None:
            np.savez(
                '{}/{}.npz'.format(self._results_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S')),
                genes=self._captures[-1].genes,
                objectives=self._objectives_history[:self.current_generation + 1],
                fitnesses=self._fitnesses_history[:self.current_generation + 1]
            )
        
    def run(
        self, population_size: int = 32, generation_cap: int = 64, *args, **kwargs
    ) -> Tuple[Population, np.ndarray, np.ndarray]:
        # init volatile fields
        self._is_running = True
        
        self._population_size = population_size
        self._generation_cap = generation_cap

        #init captures
        self._captures: List[Population] = [None] * len(self._operators)

        # run initialization with optional params
        init_genes = self._initializer(self.population_size, *args, **kwargs)
        init_pop = Population(init_genes, self)
        self._captures[-1] = init_pop

        # initialize journals
        self._fitnesses_history = np.empty((generation_cap,) + init_pop.fitnesses.shape, np.float)
        self._objectives_history = np.empty((generation_cap,) + init_pop.objectives.shape, init_pop.objectives.dtype)

        # loop over generations
        self._curr_generation = 0
        while self._curr_generation < self._generation_cap and self._run_epoch():
            self._curr_generation += 1

        # save results
        self._save_state()

        # clean up
        self._is_running = False
        
        return self._captures[-1], self._fitnesses_history, self._objectives_history

    def resume(self, generation_cap: int = 64) -> Tuple[Population, np.ndarray, np.ndarray]:
        # init volatile fields
        self._is_running = True
        
        self._generation_cap += generation_cap
        
        # initialize journals
        self._fitnesses_history = np.concatenate((
            self._fitnesses_history,
            np.empty((generation_cap,) + self._fitnesses_history.shape[1:], np.float)
        ))
        self._objectives_history = np.concatenate((
            self._objectives_history,
            np.empty((generation_cap,) + self._objectives_history.shape[1:], np.float)
        ))
        
        # loop over generations
        while self._curr_generation < self._generation_cap and self._run_epoch():
            self._curr_generation += 1
        
        # save results
        self._save_state()
        
        # clean up
        self._is_running = False
        
        return self._captures[-1], self._fitnesses_history, self._objectives_history

from abc import abstractmethod
from numpy import ndarray
from typing import List, Callable


class Population:
    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def individuals(self) -> ndarray:
        return self._individuals

    @property
    def objectives(self) -> ndarray:
        self._evaluate_objective()
        return self._objective

    @property
    def fitnesses(self) -> ndarray:
        self._evaluate_fitness()
        return self._fitness

    def __init__(
            self, individuals: ndarray,
            ga: GeneticAlgorithm
    ):
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
        # TODO maybe add index for even lazier fitness evaluation and objective value evaluation
        if self._fitness is None:
            self._evaluate_objective()

            self._fitness = self._ga.fitness_fnc(self._individuals, self._objective)
            self._fitness.flags['WRITEABLE'] = False

    def __deepcopy__(self):
        NotImplemented


class OperatorBase:
    @property
    def op_id(self) -> int:
        return self._op_id

    def __init__(self, *input_ops: 'OperatorBase', graph_builder: OperatorGraphBuilder = None):
        if len(input_ops) == 0:  # dummy operation
            self._graph_builder = graph_builder
            self._op_id = -1
            self._input_ids = None
        else:
            self._graph_builder = graph_builder if (graph_builder is not None) else input_ops[0]._graph_builder

            for input_op in input_ops:
                if self._graph_builder is not input_op._graph_builder:
                    raise ValueError('Operations do not belong to one graph')

            self._op_id = self._graph_builder.add_operator(self)
            self._input_ids = (input_op.op_id for input_op in input_ops)

            for input_id in self._input_ids:
                if input_id >= self._op_id:
                    raise ValueError(
                        'Input (id: {0!r}) will be undefined at the time of processing this operation (id: {1!r})'
                        .format(input_id, self._op_id)
                    )

    @abstractmethod
    def _operation(self, ga: GeneticAlgorithm, *input_populations: Population) -> Population:
        raise NotImplementedError('Call of not implemented abstract method.')

    def __call__(self, ga: GeneticAlgorithm) -> Population:
        return self._operation(ga, *(ga.capture(input_id) for input_id in self._input_ids))


class OperatorGraphBuilder:
    @property
    def init_op(self) -> OperatorBase:
        return self._init_op

    def __init__(self):
        self._is_built = False
        self._operators = []
        self._init_op = OperatorBase(graph_builder=self)

    def add_operator(self, operator: OperatorBase) -> int:
        if self._is_built:
            raise ValueError(
                'Operator addition forbidden on built (finalized) instance of {!r}'.format(self.__class__.__name__)
            )

        new_op_id = len(self._operators)
        self._operators.append(operator)

        return new_op_id

    def build_graph(self) -> List[OperatorBase]:
        self._is_built = True

        return self._operators


# TODO add individual as some type - ie.
# TODO configuration load dump mechanism
# TODO some basic set of operators and tests
class GeneticAlgorithm:
    @property
    def objective_fnc(self) -> Callable[[ndarray], ndarray]:
        return self._objective_fnc

    @property
    def fitness_fnc(self) -> Callable[[ndarray, ndarray], ndarray]:
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

    def __init__(
            self, initialization: Callable[[int, ...], ndarray], operators: List[OperatorBase],
            objective_fnc: Callable[[ndarray], ndarray], fitness_fnc: Callable[[ndarray, ndarray], ndarray] = None,
            early_stopping: Callable[['GeneticAlgorithm'], bool] = None,
            callbacks: List[Callable[['GeneticAlgorithm'], None]] = None,
            population_size: int = 128, generation_cap: int = 1024
    ):
        self.population_size = population_size
        self.generation_cap = generation_cap

        self._initialization = initialization
        self._objective_fnc = objective_fnc
        self._fitness_fnc = fitness_fnc if (fitness_fnc is not None) else (lambda _, obj: obj)
        self._early_stopping = early_stopping
        self._operators = operators

        self._callbacks = callbacks

        self._is_running = False
        self._captures = None
        self._curr_generation = None

    def run(self, *args, **kwargs) -> Population:
        # init capture list
        self._is_running = True
        self._captures = [None] * len(self._operators)

        # run initialization with optional params
        init_individuals = self._initialization(self.population_size, *args, **kwargs)
        self._captures[-1] = [Population(init_individuals, self)]

        # loop over generations
        for self._curr_generation in range(self.generation_cap):
            # handle early stopping
            if (self._early_stopping is not None) and self._early_stopping(self):
                break

            # loop over each operator
            for op in self._operators:
                self._captures[op.op_id] = op(self)

            for callback in (self._callbacks if (self._callbacks is not None) else []):
                callback(self)

            # clear all captures but last
            self._captures[:-1] = [None] * (len(self._captures) - 1)

        # clean up
        self._is_running = False
        result = self._captures[-1]
        self._captures = None

        return result  # TODO resolve type check

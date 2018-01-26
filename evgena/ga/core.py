from typing import TypeVar, Callable, Generic, List, NewType
from abc import ABC, abstractmethod

IndividualsTag = TypeVar('IndividualsTag')
ObjectivesTag = TypeVar('ObjectivesTag')
FitnessesTag = TypeVar('FitnessesTag')


ObjectiveFnc = NewType('ObjectiveFnc', Callable[[IndividualsTag], ObjectivesTag])
FitnessFnc = NewType('FitnessFnc', Callable[[IndividualsTag, ObjectivesTag], FitnessesTag])

Initializer = NewType('Initializer', Callable[[int, ...], IndividualsTag])


class Population(Generic[IndividualsTag, ObjectivesTag, FitnessesTag]):
    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def individuals(self) -> IndividualsTag:
        return self._individuals

    @property
    def objectives(self) -> ObjectivesTag:
        self._evaluate_objective()
        return self._objective

    @property
    def fitnesses(self) -> FitnessesTag:
        self._evaluate_fitness()
        return self._fitness

    def __init__(self, individuals: IndividualsTag, ga: 'GeneticAlgorithm') -> None:
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

    def _evaluate_objective(self):
        if self._objective is None:
            self._objective = self._ga.objective_fnc(self._individuals)
            self._objective.flags['WRITEABLE'] = False

    def _evaluate_fitness(self):
        # TODO maybe add index for even lazier fitness evaluation and objective value evaluation
        if self._fitness is None:
            self._evaluate_objective()

            self._fitness = self._ga.fitness_fnc(self._individuals, self._objective)
            self._fitness.flags['WRITEABLE'] = False

    def __deepcopy__(self):
        NotImplemented


class OperatorBase(ABC, Generic[IndividualsTag, ObjectivesTag, FitnessesTag]):
    @property
    def op_id(self) -> int:
        return self._op_id

    def __init__(
            self, *input_ops: 'OperatorBase[IndividualsTag, ObjectivesTag, FitnessesTag]'
    ) -> None:
        self._input_ids = (input_op.op_id for input_op in input_ops)

        for input_i, input_id in enumerate(self._input_ids):
            if input_id is None:
                raise ValueError(
                    '{0:d}-th input will be undefined at the time of processing currently constructed operation'
                    .format(input_i)
                )

        self._op_id = None

    def register(self, op_id) -> None:
        if self._op_id is not None:
            raise ValueError('Operator already registered with id: {0!r}, failed to register with: {1!r}'
                             .format(self._op_id, op_id))

        self._op_id = op_id

    @abstractmethod
    def _operation(
            self, ga: 'GeneticAlgorithm[IndividualsTag, ObjectivesTag, FitnessesTag]',
            *input_populations: Population[IndividualsTag, ObjectivesTag, FitnessesTag]
    ) -> Population[IndividualsTag, ObjectivesTag, FitnessesTag]:
        ...

    def __call__(
            self, ga: 'GeneticAlgorithm[IndividualsTag, ObjectivesTag, FitnessesTag]'
    ) -> Population[IndividualsTag, ObjectivesTag, FitnessesTag]:
        return self._operation(ga, *(ga.capture(input_id) for input_id in self._input_ids))


class OperatorGraphBuilder(Generic[Operator[IndividualsTag, ObjectivesTag, FitnessesTag]]):
    @property
    def init_op(self) -> Operator[IndividualsTag, ObjectivesTag, FitnessesTag]:
        return self._init_op

    def __init__(self):
        self._operators = []
        self._init_op = Operator(None, -1)

    def add_operator(self, operation, *input_ops, **op_config) -> Operator[IndividualsTag, ObjectivesTag, FitnessesTag]:
        if self._operators is None:
            raise ValueError(
                'Operator addition forbidden on builded (finalized) instance of {!r}'.format(self.__class__.__name__)
            )

        new_op = Operator(operation, len(self._operators), *input_ops, **op_config)
        self._operators.append(new_op)

        return new_op

    def build(self) -> List[Operator[IndividualsTag, ObjectivesTag, FitnessesTag]]:
        result = self._operators
        self._operators = None

        return result


# TODO add individual as some type - ie.
# TODO configuration load dump mechanism
# TODO some basic set of operators and tests
class GeneticAlgorithm(Generic[IndividualsTag, ObjectivesTag, FitnessesTag]):
    @property
    def objective_fnc(self) -> ObjectiveFnc:
        return self._objective_fnc

    @property
    def fitness_fnc(self) -> FitnessFnc:
        return self._fitness_fnc

    def operator(self, operator_id) -> Operator[IndividualsTag, ObjectivesTag, FitnessesTag]:
        return self._operators[operator_id]

    def capture(self, operator_id) -> Population[IndividualsTag, ObjectivesTag, FitnessesTag]:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide captures')

        return self._captures[operator_id]

    @property
    def current_generation(self) -> int:
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide current_generation')

        return self._curr_generation

    def __init__(
            self, initialization: Initializer, operators: List[Operator[IndividualsTag, ObjectivesTag, FitnessesTag]],
            objective_fnc: ObjectiveFnc, fitness_fnc: FitnessFnc = None,
            early_stopping: Callable[['GeneticAlgorithm[IndividualsTag, ObjectivesTag, FitnessesTag]'], bool] = None,
            callbacks: Callable[['GeneticAlgorithm[IndividualsTag, ObjectivesTag, FitnessesTag]'], None] = None,
            population_size: int = 128, generation_cap: int = 1024
    ) -> None:
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

    def run(self, *args, **kwargs) -> Population[IndividualsTag, ObjectivesTag, FitnessesTag]:
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

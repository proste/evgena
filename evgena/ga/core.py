class OperatorGraphBuilder:
    @property
    def init_op(self):
        return self._init_op

    def __init__(self):
        self._operators = []
        self._init_op = Operator(None, -1)

    def add_operator(self, operation, *input_ops, **op_config):
        if self._operators is None:
            raise ValueError(
                'Operator addition forbidden on builded (finalized) instance of {!r}'.format(self.__class__.__name__)
            )

        new_op = Operator(operation, len(self._operators), *input_ops, **op_config)
        self._operators.append(new_op)

        return new_op

    def build(self):
        result = self._operators
        self._operators = None

        return result


class Population:
    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def individuals(self):
        return self._individuals

    @property
    def objective(self):
        self._evaluate_objective()
        return self._objective

    @property
    def fitness(self):
        self._evaluate_fitness()
        return self._fitness

    def __init__(self, individuals, ga):
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


class Operator:
    @property
    def op_id(self):
        return self._op_id

    def __init__(self, operation, op_id, *input_ops, **op_config):
        self._input_ids = (input_op.op_id for input_op in input_ops)

        for input_id in self._input_ids:
            if input_id >= op_id:
                raise ValueError(
                    'Input with id {0!r} will be undefined during processing of operation {1!r}'.format(input_id, op_id)
                )

        self._op_id = op_id
        self._operation = operation
        self._op_config = op_config

    def configure(self, **kwargs):
        self._op_config.update(**kwargs)

    def __call__(self, ga):
        return self._operation(*(ga.capture(input_id) for input_id in self._input_ids), **self._op_config)


# TODO add individual as some type - ie.
class GeneticAlgorithm:
    @property
    def objective_fnc(self):
        return self._objective_fnc

    @property
    def fitness_fnc(self):
        return self._fitness_fnc

    def operator(self, operator_id):
        return self._operators[operator_id]

    def capture(self, operator_id):
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide captures')

        return self._captures[operator_id]

    @property
    def current_generation(self):
        if not self._is_running:
            raise RuntimeError('GA is not running, cannot provide current_generation')

        return self._curr_generation

    def __init__(
            self, initialization, objective_fnc, fitness_fnc, operators,
            early_stopping=None, callbacks=None, population_size=128, generation_cap=1024
    ) -> None:
        self.population_size = population_size
        self.generation_cap = generation_cap

        self._initialization = initialization
        self._objective_fnc = objective_fnc
        self._fitness_fnc = fitness_fnc
        self._early_stopping = early_stopping
        self._operators = operators

        self._callbacks = callbacks

        self._is_running = False
        self._captures = None
        self._curr_generation = None

    def run(self, *args, **kwargs):
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
                self._captures[op.output_id] = op(self)

            for callback in self._callbacks:
                callback(self)

            # clear all captures but last
            self._captures[:-1] = [None] * (len(self._captures) - 1)

        # clean up
        self._is_running = False
        result = self._captures[-1]
        self._captures = None

        return result

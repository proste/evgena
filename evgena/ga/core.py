class OperatorGraphBuilder:
    def __init__(self):
        self._operators = []

    def add_operator(self, operation, *input_ids, **op_config):
        if self._operators is None:
            raise ValueError(
                'Operator addition forbidden on builded (finalized) instance of {!r}'.format(self.__class__.__name__)
            )

        op_id = len(self._operators) + 1
        self._operators.append(Operator(operation, op_id, *input_ids, **op_config))

        return op_id

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

    @property
    def objective_fnc(self):
        return self._objective_fnc

    @property
    def fitness_fnc(self):
        return self._fitness_fnc

    def __init__(self, individuals, objective_fnc, fitness_fnc):
        # define individuals and make them read only
        if individuals.flags['OWNDATA']:
            self._individuals = individuals
        else:
            self._individuals = individuals.copy()

        self._individuals.flags['WRITEABLE'] = False

        # initialize fitness and objective functions
        self._objective_fnc = objective_fnc
        self._fitness_fnc = fitness_fnc

        # initialize objective and fitness value tables
        self._objective = None
        self._fitness = None

    def _evaluate_objective(self):
        if self._objective is None:
            self._objective = self._objective_fnc(self._individuals)
            self._objective.flags['WRITEABLE'] = False

    def _evaluate_fitness(self): # TODO maybe add index for even lazier fitness evaluation and objective value evaluation
        if self._fitness is None:
            self._evaluate_objective()

            self._fitness = self._fitness_fnc(self._individuals, self._objective)
            self._fitness.flags['WRITEABLE'] = False

    def __deepcopy__(self):
        NotImplemented


class Operator:
    @property
    def output_id(self):
        return self._output_id

    def __init__(self, operation, output_id, *input_ids, **op_config):
        for in_id in input_ids:
            if in_id < output_id:
                raise ValueError(
                    'Input with id {0!r} will be undefined during processing of operation {1!r}'.format(in_id, output_id)
                )

        self._output_id = output_id
        self._input_ids = input_ids
        self._operation = operation
        self._op_config = op_config

    def configure(self, **kwargs):
        self._op_config.update(**kwargs)

    def __call__(self, captures):
        return self._operation(*(captures[input_id] for input_id in self._input_ids), **self._op_config)


class GeneticAlgorithm:
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

    def run(self, *args, **kwargs):
        # init capture list
        captures = [None] * (1 + len(self._operators))

        # run initialization with optional params
        init_individuals = self._initialization(*args, **kwargs)
        captures[0] = [Population(init_individuals, self._fitness_fnc, self._objective_fnc)]

        # loop over generations
        for generation_i in range(self.generation_cap):
            # handle early stopping
            if (self._early_stopping is not None) and self._early_stopping(captures[0]):
                break

            # loop over each operator
            for op in self._operators:
                captures[op.output_id] = op(captures)

            for callback in self._callbacks:
                callback(generation_i, captures)

            # last output as input of next iteration, clear other outputs
            captures[0] = captures[-1]
            for capture_i in range(1, len(captures)):
                captures[capture_i] = None

        return captures[0]

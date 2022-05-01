import itertools
import random
import constants


class Population:
    def __init__(
        self,
        initial_population,
        evaluation_function,
        *,
        stats=None,
        evaluate_initial=True,
        min_length=None,
        max_length=None
    ):
        self._population = initial_population
        self._eval_func = evaluation_function
        self._stats = stats
        self._fitness = None
        self._sum_fitness = None
        self._custom_mutations = []
        self._min_length = min_length if min_length else 1
        self._max_length = max_length if max_length else float('inf')
        if evaluate_initial:
            self.evaluate()

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        self._population = population
        self._fitness = None
        self._sum_fitness = None

    @property
    def fitness(self):
        return self._fitness

    @property
    def sum_fitness(self):
        return self._sum_fitness

    @property
    def total_fitness(self):
        if self.sum_fitness:
            return self.sum_fitness[-1]
        return None
    
    @property
    def convergence_count(self):
        return self._stats.get_convergence_count()

    def clear_population(self):
        self._population = None
        self._fitness = None
        self._sum_fitness = None

    def replace_and_evaluate(self, population):
        self.population = population
        for individual in self.population:
            assert type(individual) is tuple
        self.evaluate()

    def _set_raw_fitness_stats(self, raw_fitness):
        if self._stats is None:
            return
        best_fitness, index = min([(fit, i) for i, fit in enumerate(raw_fitness)])
        best_individual = self.population[index]
        self._stats.set_best_fitness(best_fitness, best_individual)
        average_fitness = sum(raw_fitness) / len(raw_fitness)
        self._stats.set_average_fitness(average_fitness)

    def _get_raw_fitness(self):
        """Raw fitness value returned by eval_func.

        Note that it is expected that eval_func returns standardized
        fitness values, where fitness scores are non-negative, and
        smaller values represent better individuals.
        """
        raw_fitness = [self._eval_func(individual) for individual in self.population]
        self._set_raw_fitness_stats(raw_fitness)

        return raw_fitness

    def _get_adjusted_fitness(self):
        """
        Converts standardized (raw) fitness to adjusted fitness values
        in the range [0..1], where 1 represents the best fitness.
        """
        raw_fitness = self._get_raw_fitness()
        adjusted_fitness = [1.0 / (1 + fitness) for fitness in raw_fitness]

        return adjusted_fitness

    def _get_normalized_fitness(self):
        """
        Converts adjusted fitness into probability values where
        sum(normalized_fitness) == 1 and adjusted fitness ratios
        are maintained.
        """
        adjusted_fitness = self._get_adjusted_fitness()
        normalized_fitness = [
            fitness / sum(adjusted_fitness) for fitness in adjusted_fitness
        ]

        return normalized_fitness

    def _get_fitness(self):
        """Gets the fitness values for the current population."""
        # dev note: this currently returns the normalized fitness,
        # but the intent is for this method to abstract away that detail.
        return self._get_normalized_fitness()

    def _mutate_locus_swap(self, individual):
        """Swap the position of two genes."""
        if len(individual) >= 2:
            i = random.randint(0, len(individual) - 2)
            j = random.randint(i + 1, len(individual) - 1)
            ilist = list(individual)
            ilist[i], ilist[j] = ilist[j], ilist[i]
            individual = tuple(ilist)
        return individual

    def _mutate_reverse_segment(self, individual):
        """Reverse a random, random-length segment in-place."""
        if len(individual) >= 2:
            i = random.randint(0, len(individual) - 2)
            j = random.randint(i + 2, len(individual))
            if i == 0:
                return individual[:i] + individual[j - 1 :: -1] + individual[j:]
            else:
                return individual[:i] + individual[j - 1 : i - 1 : -1] + individual[j:]
        return individual

    def _mutate_replicate_gene(self, individual):
        """Replicate a single random gene to a random location.

        Note: This operation modifies the length of the individual.
        """
        if len(individual) < self._max_length:
            choice = random.choice(individual)
            i = random.randint(0, len(individual))
            individual = individual[:i] + (choice,) + individual[i:]
        return individual

    def _mutate_delete_gene(self, individual):
        """Delete a single gene from a random location.

        Note: This operation modifies the length of the individual.
        """
        if len(individual) > self._min_length:
            i = random.randint(0, len(individual) - 1)
            individual = individual[:i] + individual[i + 1 :]
        return individual

    def _mutate_replicate_segment(self, individual):
        # TODO: Basically the same as replicate gene, but for a range
        raise NotImplementedError()

    def _mutate_delete_segment(self, individual):
        # TODO: Basically the same as delete gene, but for a range
        raise NotImplementedError()

    def add_custom_mutation(self, mutate_function):
        self._custom_mutations.append(mutate_function)

    def mutate(self, individual, mutation_type, mutation_rate):
        """Random mutation."""
        # If multiple mutation types are specified, they will be performed
        # in a set order, which is probably not the best thing.
        if constants.MutationType.SWAP in mutation_type:
            if random.random() < mutation_rate:
                individual = self._mutate_locus_swap(individual)
        if constants.MutationType.REVERSE in mutation_type:
            if random.random() < mutation_rate:
                individual = self._mutate_reverse_segment(individual)
        if constants.MutationType.REPLICATE in mutation_type:
            if random.random() < mutation_rate:
                individual = self._mutate_replicate_gene(individual)
        if constants.MutationType.DELETE in mutation_type:
            if random.random() < mutation_rate:
                individual = self._mutate_delete_gene(individual)
        if constants.MutationType.CUSTOM in mutation_type:
            for func in self._custom_mutations:
                if random.random() < mutation_rate:
                    individual = func(individual)

        return individual

    def evaluate(self):
        """Evaluate the current population."""
        self._fitness = self._get_fitness()
        # compute the cumulative sum of the fitness values
        self._sum_fitness = list(itertools.accumulate(self.fitness))

"""
population.py
Author: Michael P. Lang
Date:    23 April 2022

Represents the population for one generation of the genetic algorithm.
Includes the evaluation, mutation, and statistics collection methods.
"""
# standard library modules
import itertools
import random
import constants
from math import inf
from collections.abc import Collection
from typing import Callable, Union, Optional

# custom modules
from statistics import Stats

Individual = tuple[str, ...]  # typedef representing an population individual


class Population:
    """Represents the population for one generation of the genetic algorithm."""

    def __init__(
        self,
        initial_population: Collection[Individual],
        evaluation_function: Callable[[Individual], Union[int, float]],
        *,
        stats: Optional[Stats] = None,
        evaluate_initial: bool = True,
        min_length: Optional[int] = 1,
        max_length: Optional[int] = inf
    ):
        """Initialize the population.

        Parameters
        ----------
        initial_population : Collection[Individual]
            The initial (generation) 0 population.
        evaluation_function : Callable[[Individual], Union[int, float]]
             The fitness evaluation function.
        stats : Optional[Stats], optional
             Record-keeping, by default None.
        evaluate_initial : bool, optional
             Whether or not to evaluate the initial population during initialization,
             by default True.
        min_length : Optional[int], optional
            The minimum number of genes an individual may have, by default 1.
        max_length : Optional[int], optional
            The maximum number of genes an individual may have, by default inf.
        """
        self._population = initial_population
        self._eval_func = evaluation_function
        self._stats = stats
        self._fitness = None
        self._sum_fitness = None
        self._custom_mutations = []
        self._min_length = min_length if min_length else 1
        self._max_length = max_length if max_length else inf
        if evaluate_initial:
            self.evaluate()

    @property
    def population(self) -> Collection[Individual]:
        """The current population."""
        return self._population

    @population.setter
    def population(self, population: Collection[Individual]) -> None:
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
        return self._stats.convergence_count()

    def clear_population(self) -> None:
        """Clears the current population and fitness values. Statistics are not modified."""
        self._population = None
        self._fitness = None
        self._sum_fitness = None

    def replace_and_evaluate(self, population: Collection[Individual]) -> None:
        """Replace and evaluate the population.

        Parameters
        ----------
        population : Collection[Individual]
            The new population.
        """
        self.population = population
        for individual in self.population:
            assert type(individual) is tuple
        self.evaluate()

    def _set_raw_fitness_stats(
        self, raw_fitness: Collection[Union[int, float]]
    ) -> None:
        """Collects statistics on the given raw fitness values.

        Parameters
        ----------
        raw_fitness : Collection[Union[int, float]]
            The raw fitness values for the entire population.
        """
        if self._stats is None:
            return
        self._stats.update(raw_fitness, self.population)

    def _get_raw_fitness(self) -> list[Union[int, float]]:
        """Raw fitness values returned the the evaluation function.

        Note
        ----
        It is expected that the evaluation functions returns standardized
        fitness values, where fitness scores are non-negative and smaller
        values represent better individuals.

        Returns
        -------
        list[Union[int, float]]
            The raw fitness values.
        """
        raw_fitness = [self._eval_func(individual) for individual in self.population]
        self._set_raw_fitness_stats(raw_fitness)

        return raw_fitness

    def _get_adjusted_fitness(self) -> list[float]:
        """Converts standardized (raw) fitness to adjusted fitness values in
        the range [0..1], where 1 represents the best fitness.

        Returns
        -------
        list[float]
            The normalized fitness values.
        """
        raw_fitness = self._get_raw_fitness()
        adjusted_fitness = [1.0 / (1 + fitness) for fitness in raw_fitness]

        return adjusted_fitness

    def _get_normalized_fitness(self) -> list[float]:
        """Converts adjusted fitness into probability values where
        sum(normalized_fitness) == 1 and adjusted fitness ratios are maintained.

        Returns
        -------
        list[float]
            The normalized fitness values.
        """
        adjusted_fitness = self._get_adjusted_fitness()
        normalized_fitness = [
            fitness / sum(adjusted_fitness) for fitness in adjusted_fitness
        ]

        return normalized_fitness

    def _get_fitness(self) -> list[float]:
        """Gets the fitness values for the current population.

        Returns
        -------
        list[float]
            The fitness values for the current population, in the same order as
            the population.
        """
        # dev note: this currently returns the normalized fitness,
        # but the intent is for this method to abstract away that detail.
        return self._get_normalized_fitness()

    def _mutate_locus_swap(self, individual: Individual) -> Individual:
        """Swap the position of two genes.

        Parameters
        ----------
        individual : Individual
            The individual to mutate.

        Returns
        -------
        Individual
            The mutated individual.
        """
        if len(individual) >= 2:
            i = random.randint(0, len(individual) - 2)
            j = random.randint(i + 1, len(individual) - 1)
            ilist = list(individual)
            ilist[i], ilist[j] = ilist[j], ilist[i]
            individual = tuple(ilist)
        return individual

    def _mutate_reverse_segment(self, individual: Individual) -> Individual:
        """Reverse a random, random-length segment in-place.

        Parameters
        ----------
        individual : Individual
            The individual to mutate.

        Returns
        -------
        Individual
            The mutated individual.
        """
        if len(individual) >= 2:
            i = random.randint(0, len(individual) - 2)
            j = random.randint(i + 2, len(individual))
            if i == 0:
                return individual[:i] + individual[j - 1 :: -1] + individual[j:]
            else:
                return individual[:i] + individual[j - 1 : i - 1 : -1] + individual[j:]
        return individual

    def _mutate_replicate_gene(self, individual: Individual) -> Individual:
        """Replicate a single random gene to a random location.

        Note
        ----
        This operation modifies the length of the individual.

        Parameters
        ----------
        individual : Individual
            The individual to mutate.

        Returns
        -------
        Individual
            The mutated individual.
        """
        if len(individual) < self._max_length:
            choice = random.choice(individual)
            i = random.randint(0, len(individual))
            individual = individual[:i] + (choice,) + individual[i:]
        return individual

    def _mutate_delete_gene(self, individual: Individual) -> Individual:
        """Delete a single gene from a random location.

        Note
        ----
        This operation modifies the length of the individual.

        Parameters
        ----------
        individual : Individual
            The individual to mutate.

        Returns
        -------
        Individual
            The mutated individual.
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

    def add_custom_mutation(
        self, mutate_function: Callable[[Individual], Individual]
    ) -> None:
        """Add a custom mutation strategy.

        Parameters
        ----------
        mutate_function : Callable[[Individual], Individual]
            The custom mutation function implementation.
        """
        self._custom_mutations.append(mutate_function)

    def mutate(
        self,
        individual: Individual,
        mutation_type: constants.MutationType,
        mutation_rate: float,
    ) -> Individual:
        """Random mutation.

        Parameters
        ----------
        individual : Individual
            The individual to (possibly) mutate.
        mutation_type : constants.MutationType
            The mutation strategy/strategies.
        mutation_rate : float
            The mutation rate.

        Returns
        -------
        Individual
            The mutated individual.
        """
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

    def evaluate(self) -> None:
        """Evaluate the current population."""
        self._fitness = self._get_fitness()
        # compute the cumulative sum of the fitness values
        self._sum_fitness = list(itertools.accumulate(self.fitness))

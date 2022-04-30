"""
genalg.py
Authors: Michael P. Lang
Date:    23 April 2022

Genetic Algorithm algorithms and utilities.
Depends on Python3, ProgressBar, and pathos.
"""


import random
from progressbar import ProgressBar, Percentage, Bar
import constants
import itertools
import bisect
from pathos.multiprocessing import ProcessPool
from population import Population


class GA:
    def __init__(self, population, eval_func, params, stats):
        self.params = params
        self.population = Population(
            population,
            eval_func,
            stats=stats,
            min_length=self.params["min_length"],
            max_length=self.params["max_length"],
        )

    def _get_mating_pool_size(self):
        """Return the required size of the mating pool based on the
        selection type and population size.
        """
        selection_type = self.params["selection_type"]
        population_size = self.params["population_size"]
        if selection_type == constants.SelectionType.TS:
            # for selection types which return one child,
            # we need to double the size of the mating pool
            mating_pool_size = population_size * 2
        else:
            # selection types which return two children need
            # exactly as many parents as children
            mating_pool_size = population_size

        return mating_pool_size

    def _get_mating_pool(self, pool_size, selection_type):
        """Get the mating pool based on the selection type."""
        if selection_type == constants.SelectionType.SUS:
            # stochastic universal sampling
            mating_pool = self._sus(pool_size)
        elif selection_type == constants.SelectionType.TS:
            # tournament selection
            mating_pool = self._ts(pool_size)
        elif selection_type == constants.SelectionType.FPS:
            # fitness proportionate selection
            mating_pool = self._fps(pool_size)
        else:
            raise Exception(f"Invalid selection type: '{selection_type}'")

        return mating_pool
    
    def _single_point_crossover(self, parent1, parent2):
        """Single-point order crossover."""
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def _crossover_order(self, parent1, parent2):
        """
        Two-point order crossover.
        """
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1, len(parent1))
        """
        Get a list of items in parent2, starting from crossover_point2, which
        are not in the middle segment of parent1.
        """
        unused = tuple(
            [
                x
                for x in parent2[crossover_point2:] + parent2[:crossover_point2]
                if x not in parent1[crossover_point1:crossover_point2]
            ]
        )
        """
        Copy the middle segment from parent1 to child1, and fill in the empty
        slots from the unused list, beginning with crossover_point2 and
        wrapping around to the beginning.
        """
        child1 = (
            unused[len(parent1) - crossover_point2 :]
            + parent1[crossover_point1:crossover_point2]
            + unused[: len(parent1) - crossover_point2]
        )

        """
        Get a list of items in parent1, starting from crossover_point2, which
        are not in the middle segment of parent2.
        """
        unused = tuple(
            [
                x
                for x in parent1[crossover_point2:] + parent1[:crossover_point2]
                if x not in parent2[crossover_point1:crossover_point2]
            ]
        )
        """
        Copy the middle segment from parent1 to child1, and fill in the empty
        slots from the unused list, beginning with crossover_point2 and
        wrapping around to the beginning.
        """
        child2 = (
            unused[len(parent1) - crossover_point2 :]
            + parent2[crossover_point1:crossover_point2]
            + unused[: len(parent1) - crossover_point2]
        )

        return child1, child2

    def _crossover_partially_mapped(self, parent1, parent2):
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1))

        child1 = [None] * len(parent1)
        child1[crossover_point1:crossover_point2] = parent1[
            crossover_point1:crossover_point2
        ]
        for pos in range(crossover_point1, crossover_point2):
            i = parent2[pos]
            if i not in parent1[crossover_point1:crossover_point2]:
                j = parent1[pos]
                pos = parent2.index(j)
                while child1[pos]:
                    j = parent1[pos]
                    pos = parent2.index(j)
                child1[pos] = i

        unused = [i for i in parent2 if i not in child1]
        pos = 0
        for i in range(len(child1)):
            if child1[i] is None:
                child1[i] = unused[pos]
                pos += 1

        child2 = [None] * len(parent1)
        child2[crossover_point1:crossover_point2] = parent2[
            crossover_point1:crossover_point2
        ]
        for pos in range(crossover_point1, crossover_point2):
            i = parent1[pos]
            if i not in parent2[crossover_point1:crossover_point2]:
                j = parent2[pos]
                pos = parent1.index(j)
                while child2[pos]:
                    j = parent2[pos]
                    pos = parent1.index(j)
                child2[pos] = i

        unused = [i for i in parent1 if i not in child2]
        pos = 0
        for i in range(len(child2)):
            if child2[i] is None:
                child2[i] = unused[pos]
                pos += 1

        return child1, child2

    def _crossover_edge_recombination(self, parent1, parent2):
        adj1 = {
            parent1[i]: {parent1[i - 1], parent1[(i + 1) % len(parent1)]}
            for i in range(len(parent1))
        }
        adj2 = {
            parent2[i]: {parent2[i - 1], parent2[(i + 1) % len(parent2)]}
            for i in range(len(parent2))
        }
        adjset = {
            parent1[i]: adj1[parent1[i]].union(adj2[parent1[i]])
            for i in range(len(parent1))
        }

        child = []
        n = random.choice([parent1[0], parent2[0]])
        while True:
            child.append(n)

            if len(child) == len(parent1):
                break

            for node in adjset:
                if n in adjset[node]:
                    adjset[node].remove(n)

            if len(adjset[n]) != 0:
                if len(adjset[n]) == 1:
                    nxt = adjset[n].pop()
                else:
                    lengths = {
                        neighbor: len(adjset[neighbor]) for neighbor in adjset[n]
                    }
                    # next = sorted(lengths, key=lengths.get)[0]
                    nxt = random.choice(
                        [c for c in lengths if lengths[c] == min(lengths.values())]
                    )
            else:
                nxt = n
                while nxt in child:
                    nxt = random.choice(parent1)
            n = nxt

        return [child]

    def _crossover(self, parent1, parent2):
        if random.random() > self.params["crossover_rate"]:
            return parent1, parent2
        else:
            crossover_type = self.params["crossover_type"]
            if crossover_type == constants.CrossoverType.OX:
                # ordered crossover
                return self._crossover_order(parent1, parent2)
            elif crossover_type == constants.CrossoverType.PMX:
                # partially-mapped crossover
                return self._crossover_partially_mapped(parent1, parent2)
            elif crossover_type == constants.CrossoverType.ERX:
                # edge recombination crossover
                return self._crossover_edge_recombination(parent1, parent2)
            else:
                return parent1, parent2

    def _fps(self, mating_pool_size):
        """Generate the mating pool using fitness proportionate selection."""
        # Conceptially, this operation is akin to spinning a game wheel
        # where each member of the population is represented by a slice
        # of the wheel proportionate to its fitness. This is repeated once
        # for each spot in the mating pool.
        mating_pool = []
        total_fitness = self.population.total_fitness
        for i in range(mating_pool_size):
            # Generate a random value between 0 and the cumulative fitness.
            # When using the normalized fitness, this will be between 0 and 1.
            rand = random.random() * total_fitness
            # the bisect operation will select the first item greater-than rand
            mating_pool.append(
                self.population.population[
                    bisect.bisect(self.population.sum_fitness, rand)
                ]
            )

        return mating_pool

    def _sus(self, mating_pool_size):
        # stochastic universal sampling
        mating_pool = []
        p_dist = 1.0 / (mating_pool_size)
        rand = random.uniform(0, p_dist)
        points = [rand + i * p_dist for i in range(mating_pool_size)]
        i = 0
        for p in points:
            while self.population.sum_fitness[i] < p:
                i += 1
            mating_pool.append(self.population.population[i]["individual"])

        return mating_pool

    def _ts(self, mating_pool_size):
        # tournament selection
        mating_pool = []
        population_size = self.params["population_size"]
        for i in range(mating_pool_size):
            tournament_size = self.params["tournament_size"]
            contenders = random.sample(range(population_size), tournament_size)
            fit = self.population.fitness[contenders[0]]
            parent = self.population.population[contenders[0]]["individual"]
            for i in range(1, tournament_size):
                if self.population.fitness[contenders[i]] > fit:
                    fit = self.population.fitness[contenders[i]]
                    parent = self.population.population[contenders[i]]["individual"]
            mating_pool.append(parent)

        return mating_pool

    def _mate_mutate_evaluate(self, parent1, parent2, mutation_type, mutation_rate):
        return [
            self.population.mutate(child, mutation_type, mutation_rate)
            for child in self._crossover(parent1, parent2)
        ]

    def add_custom_mutation(self, mutate_function):
        self.population.add_custom_mutation(mutate_function)

    def evolve(self, display_progress=True):
        # grab some settings to avoid repeated lookups
        max_generations = self.params["max_generations"]
        mutation_type = self.params["mutation_type"]
        mutation_rate = self.params["mutation_rate"]
        convergence_termination = self.params["convergence_termination"]
        convergence_generations = self.params["convergence_generations"]
        mating_pool_size = self._get_mating_pool_size()
        selection_type = self.params["selection_type"]

        if display_progress:
            pbar = ProgressBar(
                widgets=[Percentage(), Bar()], maxval=max_generations
            ).start()
        else:
            pbar = None

        for generation in range(max_generations):
            if convergence_termination:
                if self._stats.get_count() > convergence_generations:
                    break
            mating_pool = self._get_mating_pool(
                mating_pool_size, selection_type
            )  # selection
            results = []
            with ProcessPool() as pool:
                for i in range(0, len(mating_pool), 2):
                    # pairwise recombination
                    # TODO: mating strategies for > 2 parents
                    parent1 = mating_pool[i]
                    parent2 = mating_pool[i + 1]
                    results.append(
                        pool.apipe(
                            self._mate_mutate_evaluate,
                            parent1,
                            parent2,
                            mutation_type,
                            mutation_rate,
                        )
                    )
            # Wait for mating to complete and collect the results in a list
            self.population.replace_and_evaluate(
                list(itertools.chain(*[res.get() for res in results]))
            )
            if pbar is not None:
                pbar.update(generation + 1)
        if pbar is not None:
            pbar.finish()

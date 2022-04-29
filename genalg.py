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
from collections.abc import Iterable
# import concurrent.futures
from pathos.multiprocessing import ProcessPool

class GA:
    def __init__(self, population, eval_func, params, stats):
        self.population = population
        self.eval_func = eval_func
        self.params = params
        self.stats = stats
        self.fitness = None
        self.sum_fitness = None
        self.best_fitness = None
        self.average_fitness = None

    def mutate_locus_swap(self, individual):
        """Swap the position of two genes.
        """
        if len(individual) >= 2:
            i = random.randint(0, len(individual) - 2)
            j = random.randint(i + 1, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def mutate_reverse_segment(self, individual):
        """Reverse a random, random-length segment in-place.
        """
        if len(individual) >= 2:
            i = random.randint(0, len(individual) - 2)
            j = random.randint(i + 2, len(individual))
            if i == 0:
                return individual[:i] + individual[j-1::-1] + individual[j:]
            else:
                return individual[:i] + individual[j-1:i-1:-1] + individual[j:]
        return individual
    
    def mutate_replicate_gene(self, individual):
        """Replicate a single random gene to a random location.
        
        Note: This operation modifies the length of the individual.
        """
        choice = random.choice(individual)
        i = random.randint(0, len(individual))
        individual.insert(i, choice)
        # even though this modifies the list in-place, we'll still return to be consistent
        return individual
    
    def mutate_delete_gene(self, individual):
        """Delete a single gene from a random location.
        
        Note: This operation modifies the length of the individual.
        """
        if len(individual) > 1:
            i = random.randint(0, len(individual) - 1)
            del individual[i]
            # even though this modifies the list in-place, we'll still return to be consistent
        return individual

    def mutate_replicate_segment(self, individual):
        # TODO: Basically the same as replicate gene, but for a range
        raise NotImplementedError()

    def mutate_delete_segment(self, individual):
        # TODO: Basically the same as delete gene, but for a range
        raise NotImplementedError()
    
    def mutate_replace_gene(self, individual):
        """Replace a random gene with a randomly-chosen item from the available bases.
        """
        bases = self.params['bases']
        if isinstance(bases, Iterable) and len(bases):
            choice = random.choice(bases)
            i = random.randint(0, len(individual) - 1)
            individual[i] = choice
        # even though this modifies the list in-place, we'll still return to be consistent
        return individual

    def mutate(self, individual):
        """Random mutation.
        """
        mutation_rate = self.params['mutation_rate']
        mutation_type = self.params['mutation_type']
        if constants.MutationType.SWAP in mutation_type:
            if random.random() < mutation_rate:
                individual = self.mutate_locus_swap(individual)
        if constants.MutationType.REVERSE in mutation_type:
            if random.random() < mutation_rate:
                individual = self.mutate_reverse_segment(individual)
        if constants.MutationType.REPLICATE in mutation_type:
            if random.random() < mutation_rate:
                individual = self.mutate_replicate_gene(individual)
        if constants.MutationType.DELETE in mutation_type:
            if random.random() < mutation_rate:
                individual = self.mutate_delete_gene(individual)
        if constants.MutationType.REPLACE in mutation_type:
            if random.random() < mutation_rate:
                individual = self.mutate_replace_gene(individual)
        return individual

    def get_raw_fitness(self):
        """Raw fitness value returned by eval_func.

        Note that it is expected that eval_func returns standardized
        fitness values, where fitness scores are non-negative, and
        smaller values represent better individuals.
        """
        raw_fitness = [self.eval_func(individual)
                       for individual in self.population]

        self.best_fitness = min(raw_fitness)
        index = raw_fitness.index(self.best_fitness)
        best_individual = self.population[index]
        self.stats.set_best_fitness(self.best_fitness, best_individual)
        self.average_fitness = sum(raw_fitness) / len(raw_fitness)
        self.stats.set_average_fitness(self.average_fitness)

        return raw_fitness

    def get_adjusted_fitness(self):
        """
        Converts standardized (raw) fitness to adjusted fitness values
        in the range [0..1], where 1 represents the best fitness.
        """
        raw_fitness = self.get_raw_fitness()
        adjusted_fitness = [1.0 / (1 + fitness)
                            for fitness in raw_fitness]

        return adjusted_fitness

    def get_normalized_fitness(self):
        """
        Converts adjusted fitness into probability values where
        sum(normalized_fitness) == 1 and adjusted fitness ratios
        are maintained.
        """
        adjusted_fitness = self.get_adjusted_fitness()
        normalized_fitness = [fitness / sum(adjusted_fitness)
                              for fitness in adjusted_fitness]

        return normalized_fitness

    def get_fitness(self):
        """Gets the fitness values for the current population.
        """
        # dev note: this currently returns the normalized fitness,
        # but the intent is for this method to abstract away that detail.
        return self.get_normalized_fitness()

    def evaluate(self):
        """Evaluate the current population.
        """
        self.fitness = self.get_fitness()
        # compute the cumulative sum of the fitness values 
        self.sum_fitness = list(itertools.accumulate(self.fitness))

    def get_mating_pool_size(self):
        """Return the required size of the mating pool based on the
        selection type and population size.
        """
        selection_type = self.params['selection_type']
        population_size = self.params['population_size']
        if selection_type == constants.SelectionType.TS:
            # for selection types which return one child,
            # we need to double the size of the mating pool
            mating_pool_size = population_size * 2
        else:
            # selection types which return two children need
            # exactly as many parents as children
            mating_pool_size = population_size

        return mating_pool_size

    def get_mating_pool(self):
        """Get the mating pool based on the selection type.
        """
        selection_type = self.params['selection_type']
        if selection_type == constants.SelectionType.SUS:
            # stochastic universal sampling
            mating_pool = self.sus()
        elif selection_type == constants.SelectionType.TS:
            # tournament selection
            mating_pool = self.ts()
        else:  # selection_type == constants.SelectionType.FPS:
            # fitness proportionate selection
            mating_pool = self.fps()

        return mating_pool

    def crossover_order(self, parent1, parent2):
        """
        Two-point order crossover.
        """
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1, len(parent1))
        """
        Get a list of items in parent2, starting from crossover_point2, which
        are not in the middle segment of parent1.
        """
        unused = [x for x in parent2[crossover_point2:] +
                  parent2[:crossover_point2]
                  if x not in parent1[crossover_point1:crossover_point2]]
        """
        Copy the middle segment from parent1 to child1, and fill in the empty
        slots from the unused list, beginning with crossover_point2 and
        wrapping around to the beginning.
        """
        child1 = (unused[len(parent1) - crossover_point2:] +
                  parent1[crossover_point1:crossover_point2] +
                  unused[:len(parent1) - crossover_point2])

        """
        Get a list of items in parent1, starting from crossover_point2, which
        are not in the middle segment of parent2.
        """
        unused = [x for x in parent1[crossover_point2:] +
                  parent1[:crossover_point2]
                  if x not in parent2[crossover_point1:crossover_point2]]
        """
        Copy the middle segment from parent1 to child1, and fill in the empty
        slots from the unused list, beginning with crossover_point2 and
        wrapping around to the beginning.
        """
        child2 = (unused[len(parent1) - crossover_point2:] +
                  parent2[crossover_point1:crossover_point2] +
                  unused[:len(parent1) - crossover_point2])

        return child1, child2

    def crossover_partially_mapped(self, parent1, parent2):
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(crossover_point1 + 1, len(parent1))

        child1 = [None] * len(parent1)
        child1[crossover_point1:crossover_point2] = \
            parent1[crossover_point1:crossover_point2]
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
        child2[crossover_point1:crossover_point2] = \
            parent2[crossover_point1:crossover_point2]
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

    def crossover_edge_recombination(self, parent1, parent2):
        adj1 = {parent1[i]: {parent1[i-1], parent1[(i+1) % len(parent1)]}
                for i in range(len(parent1))}
        adj2 = {parent2[i]: {parent2[i-1], parent2[(i+1) % len(parent2)]}
                for i in range(len(parent2))}
        adjset = {parent1[i]: adj1[parent1[i]].union(adj2[parent1[i]])
                  for i in range(len(parent1))}

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
                    next = adjset[n].pop()
                else:
                    lengths = {neighbor: len(adjset[neighbor])
                               for neighbor in adjset[n]}
                    #next = sorted(lengths, key=lengths.get)[0]
                    next = random.choice([c for c in lengths
                                          if lengths[c] ==
                                          min(lengths.values())])
            else:
                next = n
                while next in child:
                    next = random.choice(parent1)
            n = next

        return [child]

    def crossover(self, parent1, parent2):
        if random.random() > self.params['crossover_rate']:
            return parent1[:], parent2[:]
        else:
            crossover_type = self.params['crossover_type']
            if crossover_type == constants.CrossoverType.OX:
                # ordered crossover
                return self.crossover_order(parent1[:], parent2[:])
            elif crossover_type == constants.CrossoverType.PMX:
                # partially-mapped crossover
                return self.crossover_partially_mapped(parent1[:],
                                                       parent2[:])
            elif crossover_type == constants.CrossoverType.ERX:
                # edge recombination crossover
                return self.crossover_edge_recombination(parent1[:],
                                                         parent2[:])
            else:
                return parent1[:], parent2[:]

    def fps(self):
        """Generate the mating pool using fitness proportionate selection.
        """
        # Conceptially, this operation is akin to spinning a game wheel
        # where each member of the population is represented by a slice
        # of the wheel proportionate to its fitness. This is repeated once
        # for each spot in the mating pool.
        mating_pool = []
        for i in range(self.get_mating_pool_size()):
            # Generate a random value between 0 and the cumulative fitness.
            # When using the normalized fitness, this will be between 0 and 1.
            rand = random.random() * self.sum_fitness[-1]
            # the bisect operation will select the first item greater-than rand
            mating_pool.append(self.population[bisect.bisect(self.sum_fitness,
                                                             rand)][:])

        return mating_pool

    def sus(self):
        # stochastic universal sampling
        mating_pool = []
        mating_pool_size = self.get_mating_pool_size()
        p_dist = 1.0 / (mating_pool_size)
        rand = random.uniform(0, p_dist)
        points = [rand + i * p_dist for i in range(mating_pool_size)]
        i = 0
        for p in points:
            while self.sum_fitness[i] < p:
                i += 1
            mating_pool.append(self.population[i])

        return mating_pool

    def ts(self):
        # tournament selection
        mating_pool = []
        mating_pool_size = self.get_mating_pool_size()
        population_size = self.params['population_size']
        for i in range(mating_pool_size):
            tournament_size = self.params['tournament_size']
            contenders = random.sample(range(population_size), tournament_size)
            fit = self.fitness[contenders[0]]
            parent = self.population[contenders[0]]
            for i in range(1, tournament_size):
                if self.fitness[contenders[i]] > fit:
                    fit = self.fitness[contenders[i]]
                    parent = self.population[contenders[i]]
            mating_pool.append(parent[:])

        return mating_pool
    

    def _mate_mutate_evaluate(self, parent1, parent2):
        children = self.crossover(parent1, parent2)
        return [self.mutate(child) for child in children]
        

    def evolve(self, display_progress=True):
        max_generations = self.params['max_generations']
        if display_progress:
            pbar = ProgressBar(widgets=[Percentage(), Bar()],
                               maxval=max_generations).start()
        else:
            pbar = None
        for generation in range(max_generations):
            if self.params['convergence_termination']:
                if self.stats.get_count() > \
                   self.params['convergence_generations']:
                    break
            self.evaluate() # calculate fitness
            mating_pool = self.get_mating_pool() # selection
            results = []
            with ProcessPool() as pool:
                for i in range(0, len(mating_pool), 2):
                    # pairwise recombination
                    # TODO: mating strategies for > 2 parents
                    parent1 = mating_pool[i][:]
                    parent2 = mating_pool[i+1][:]
                    results.append(pool.apipe(self._mate_mutate_evaluate, parent1, parent2))
            # Wait for mating to complete and collect the results in a list
            self.population = list(itertools.chain(*[res.get() for res in results]))
            if pbar is not None:
                pbar.update(generation + 1)
        if pbar is not None:
            pbar.finish()

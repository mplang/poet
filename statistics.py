"""
statistics.py
Authors: Michael P. Lang
Date:    23 April 2022

Collects statistics for the genetic algorithm.
"""


class Stats(object):
    def __init__(self):
        # best overall fitness value for entire run of the algorithm
        self.best_fitness = None
        self.best_individual = None
        self.best_generation = None
        # list of best fitness values for each generation
        self.best_fitnesses = []
        self.best_individuals = []
        # list of average fitness values for each generations
        self.average_fitnesses = []
        self.convergence_count = 0

    def set_best_fitness(self, fitness, individual):
        """Saves the best standard fitness values."""
        if len(self.best_fitnesses) > 0 and fitness == self.best_fitnesses[-1]:
            self.convergence_count += 1
        else:
            self.convergence_count = 0
        self.best_fitnesses.append(fitness)
        self.best_individuals.append(individual)
        if self.best_fitness is None or fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = individual
            self.best_generation = len(self.best_fitnesses)

    def set_average_fitness(self, average_fitness):
        self.average_fitnesses.append(average_fitness)

    def get_best_fitness(self):
        return self.best_fitness

    def get_best_individual(self):
        return self.best_individual

    def get_best_generation(self):
        return self.best_generation

    def get_best_fitnesses(self):
        return self.best_fitnesses

    def get_best_individuals(self):
        return self.best_individuals

    def get_average_fitnesses(self):
        return self.average_fitnesses

    def get_convergence_count(self):
        return self.convergence_count

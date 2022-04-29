# generate random population
#   1. random letters
#   2. random words

# evaluation function
#   1. allow different metrics, normalized to same range(?)
import functools
from Levenshtein import distance
from PIL import Image, ImageFont, ImageDraw
from progressbar import ProgressBar, Percentage, Bar
import matplotlib.pyplot as plt
import time
import datetime
import random
import string
import statistics
import genalg

def plot_stats(params, stats):
    best_fitnesses = stats.get_best_fitnesses()
    best_fitness = stats.get_best_fitness()
    average_fitnesses = stats.get_average_fitnesses()
    selection_type = params['selection_type'].name.lower()
    crossover_type = params['crossover_type'].name.lower()
    mutation_type = params['mutation_type'].name.lower()
    generations = len(best_fitnesses)

    plt.plot(range(generations), best_fitnesses, label="best")
    plt.plot(range(generations), average_fitnesses, label="average")
    plt.title("selection: {}, recombination: {}, mutation: {}\n".
              format(selection_type, crossover_type, mutation_type) +
              "pop_size: {}, x_rate: {}, m_rate: {}\n".
              format(params['population_size'],
                     params['crossover_rate'],
                     params['mutation_rate']))
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.annotate(best_fitness, (stats.get_best_generation(),
                                best_fitness),
                 xytext=(generations / 2,
                         (best_fitnesses[0] +
                          best_fitness) // 2),
                 arrowprops={"width": 1,
                             "shrink": 0.02})
    plt.annotate(best_fitnesses[0], (0, best_fitnesses[0]),
                 xytext=(generations / 3,
                         (best_fitnesses[0] +
                          best_fitness) // 1.5),
                 arrowprops={"width": 1,
                             "shrink": 0.02})
    d = datetime.date.today().isoformat()
    plt.savefig("plot" + d + "-" + str(time.time())[5::2] + ".png")


def get_tour_length(dist_matrix, tour):
    """ Computes the length of the closed tour """
    num_cities = len(tour)
    total = 0.0

    for i in range(num_cities):
        j = (i + 1) % num_cities    # tour is a closed loop
        a, b = tour[i], tour[j]
        if a > b:
            a, b = b, a
        total += dist_matrix[a, b]

    return total

def generate_population(pop_size, max_length, bases):
    return [random.choices(list(bases), k=random.randint(1, max_length)) for i in range(pop_size)]

def run(params=None):
    # generate an initial population
    population = generate_population(params['population_size'], params['max_length'], params['bases'])
    eval_func = lambda x: distance(params['target'], ''.join(x))
    stats = statistics.Stats()
    ga = genalg.GA(population, eval_func, params, stats)
    print("Evolving...")
    ga.evolve(display_progress=True)

    best_fitnesses = stats.get_best_fitnesses()
    # best_individuals = stats.get_best_individuals()
    print("Competed generations: {}".format(len(best_fitnesses)))
    print("Best fitness: {}".format(stats.get_best_fitness()))
    print("Best individual: {}".format(stats.get_best_individual()))
    # if(params['save_images']):
    #     print("Saving images...")
    #     pbar = ProgressBar(widgets=[Percentage(), Bar()],
    #                        maxval=len(best_fitnesses)).start()
    #     for i, tour in enumerate(best_individuals):
    #         save_tour_image(city_coords, tour,
    #                         best_fitnesses[i],
    #                         "images/test{}".format(i),
    #                         (params.get_dimensions()))
    #         pbar.update(i + 1)
    #     pbar.finish()
    plot_stats(params, stats)
    print("Done!")


if __name__ == '__main__':
    run({'population_size': 10, 'max_length': 10, 'target': 'michael'})
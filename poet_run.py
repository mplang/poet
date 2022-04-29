"""
poet_run.py
Authors: Michael P. Lang
Date:    22 April 2022

CLI/main program for the genetic algorithm poet.
"""


import argparse
import constants
import poet
import string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--max-width', default=800)
    parser.add_argument('-H', '--max-height', default=600)
    parser.add_argument('-n', '--population-size', default=100)
    parser.add_argument('-g', '--max-generations', default=1000)
    parser.add_argument('-x', '--crossover-type', default="ox")
    parser.add_argument('-m', '--mutation-type', default="all")
    parser.add_argument('-s', '--selection-type', default="fps")
    parser.add_argument('-o', '--optimization-type', default="min")
    parser.add_argument('-X', '--crossover-rate', default=0.6)
    parser.add_argument('-M', '--mutation-rate', default=0.001)
    parser.add_argument('-G', '--convergence-generations', default=20)
    parser.add_argument('-t', '--tournament-size', default=5)
    parser.add_argument('-b', '--bases', default=[])
    parser.add_argument('-I', '--save-images', action="store_true")
    parser.add_argument('-B', '--save-best', action="store_true")
    parser.add_argument('-T', '--convergence-termination',
                        action="store_true")
    args = parser.parse_args()

    # Mutation types can be supplied as comma-delimited arg values.
    # Here we split them, convert them to flag values and combine them.
    mutation_args = [constants.MutationType[val.strip().upper()] for val in args.mutation_type.split(',')]
    mutation_types = mutation_args[0]
    for m in mutation_args[1:]:
        mutation_types |= m

    # TODO: error handling of the enums
    params = dict(
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        population_size=int(args.population_size),
        max_generations=int(args.max_generations),
        crossover_type=constants.CrossoverType[args.crossover_type.upper()],
        mutation_type=mutation_types,
        selection_type=constants.SelectionType[args.selection_type.upper()],
        optimization_type=constants.OptimizationType[args.optimization_type.upper()],
        crossover_rate=float(args.crossover_rate),
        mutation_rate=float(args.mutation_rate),
        save_best=args.save_best,
        save_images=args.save_images,
        convergence_generations=int(args.convergence_generations),
        convergence_termination=args.convergence_termination,
        tournament_size=int(args.tournament_size),
        bases=args.bases
    )
    params['population_size'] = 100
    params['max_length'] = 100
    params['target'] = 'michael'
    params['bases'] = string.ascii_lowercase
    params['mutation_rate']=0.001
    params['max_generations'] = 100

    poet.run(params)


if __name__ == "__main__":
    main()

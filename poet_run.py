"""
poet_run.py
Authors: Michael P. Lang
Date:    22 April 2022

CLI/main program for the genetic algorithm poet.
"""


import argparse
import constants
import poet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--population-size",
        type=int,
        default=100,
        help="the number of individuals to generate in the population (default: %(default)s)",
    )
    parser.add_argument(
        "-g",
        "--max-generations",
        type=int,
        default=1000,
        help="The maximum number of generations to run before terminating (default: %(default)s)",
    )
    parser.add_argument(
        "-x",
        "--crossover-type",
        choices=[e.name.lower() for e in constants.CrossoverType],
        default="ox",
        help="The crossover/mating strategy (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--mutation-type",
        choices=[e.name.lower() for e in constants.MutationType],
        default="all",
        help="A comma-separated list of mutation strategies to enable (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--selection-type",
        choices=[e.name.lower() for e in constants.SelectionType],
        default="fps",
        help="The mating pool selection strategy (default: %(default)s)",
    )
    # parser.add_argument(
    #     "-o",
    #     "--optimization-type",
    #     choices=[e.name.lower() for e in constants.OptimizationType],
    #     default="min",
    # )
    parser.add_argument(
        "-X",
        "--crossover-rate",
        type=float,
        default=0.6,
        help="The crossover rate (default: %(default)s)",
    )
    parser.add_argument(
        "-M",
        "--mutation-rate",
        type=float,
        default=0.001,
        help="The mutation rate (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--tournament-size",
        type=int,
        default=5,
        help="The size of the tournament for tournament selection (default: %(default)s)",
    )
    parser.add_argument(
        "-T",
        "--convergence-termination",
        action="store_true",
        help="Terminate once the population converges to a solution",
    )
    parser.add_argument(
        "-G",
        "--convergence-generations",
        type=int,
        default=25,
        help="The number of sequential generations necessary for convergence (default: %(default)s)",
    )
    parser.add_argument(
        "--min-generations",
        type=int,
        default=1,
        help="The minimum number of generations required before a convergence termination will be considered (default: %(default)s",
    )
    parser.add_argument(
        "--target-fitness-termination",
        action="store_true",
        help="Terminate once a solution at or below the target fitness is found",
    )
    parser.add_argument(
        "--target-fitness",
        type=float,
        default=0,
        help="The target fitness value for fitness termination",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("-I", "--save-images", action="store_true")
    parser.add_argument("-B", "--save-best", action="store_true")
    args = parser.parse_args()

    # Mutation types can be supplied as comma-delimited arg values.
    # Here we split them, convert them to flag values and combine them.
    mutation_args = [
        constants.MutationType[val.strip().upper()]
        for val in args.mutation_type.split(",")
    ]
    mutation_types = mutation_args[0]
    for m in mutation_args[1:]:
        mutation_types |= m

    # TODO: error handling of the enums
    params = dict(
        population_size=args.population_size,
        max_generations=args.max_generations,
        selection_type=constants.SelectionType[args.selection_type.upper()],
        crossover_type=constants.CrossoverType[args.crossover_type.upper()],
        mutation_type=mutation_types,
        # optimization_type=constants.OptimizationType[args.optimization_type.upper()],
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        convergence_termination=args.convergence_termination,
        convergence_generations=args.convergence_generations,
        min_generations=(
            args.max_generations // 10
            if args.min_generations is None
            else args.min_generations
        ),
        target_fitness_termination=args.target_fitness_termination,
        target_fitness=args.target_fitness,
        tournament_size=args.tournament_size,
        verbose=args.verbose,
        save_best=args.save_best,
        save_images=args.save_images,
    )
    params["min_length"] = 5
    params["max_length"] = 25
    params["target"] = "michael"

    poet.run(params)


if __name__ == "__main__":
    main()

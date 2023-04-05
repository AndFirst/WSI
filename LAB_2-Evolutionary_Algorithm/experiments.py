import json
import numpy as np
from LAB_2.cec2017.simple import f1, f9
from LAB_2.algorithm import generate_start_population, solver, grad_solver


def convergence_evo(f, n_repeat, population_size, dimension, min_range,
                    max_range, t_max, mutation_value, elite_size, output_name):
    all_data = {}

    for i in range(n_repeat):
        pop = generate_start_population(population_size=population_size,
                                        dimension=dimension,
                                        min_range=min_range,
                                        max_range=max_range)
        w, o, log = solver(f=f, population=pop.copy(), t_max=t_max,
                           mutation_value=mutation_value, elite_size=elite_size)
        all_data.update({i: [y for x, y in log]})
    with open(f"results/{output_name}.json", "w") as file:
        json.dump(all_data, file)


def convergence_grad(f, n_repeat, dimension, t_max, step_size,
                     min_range, max_range, output_name):
    all_data = {}
    for i in range(n_repeat):
        x0 = np.random.uniform(low=min_range, high=max_range, size=dimension)
        w, o, log = grad_solver(f=f, x0=x0, iterations=t_max, step_size=step_size)
        all_data.update({i: log})
    with open(f"results/{output_name}.json", "w") as file:
        json.dump(all_data, file)


def elite_sizes(f, n_repeat, population_size, dimension, min_range,
                max_range, t_max, mutation_value, output_name):
    all_data = {}
    elites = list(range(0, population_size + 1))
    for elite_size in elites:
        results = []
        for i in range(n_repeat):
            pop = generate_start_population(population_size=population_size,
                                            dimension=dimension,
                                            min_range=min_range,
                                            max_range=max_range)
            w, o, log = solver(f=f, population=pop.copy(), t_max=t_max,
                               mutation_value=mutation_value, elite_size=elite_size)
            results.append((w, o))
        best_res = min(results, key=lambda x: x[1])
        avg = np.mean([y for x, y in results])
        all_data.update({elite_size: {
            "best_result": best_res[1],
            "average": avg
        }})
    with open(f"results/{output_name}.json", "w") as file:
        json.dump(all_data, file)


def population_sizes(f, n_repeat, max_population_size, population_step, dimension, min_range,
                     max_range, t_max, mutation_value, output_name):
    all_data = {}
    populations = list(range(5, max_population_size + 1, population_step))
    for population_size in populations:
        results = []
        for i in range(n_repeat):
            pop = generate_start_population(population_size=population_size,
                                            dimension=dimension,
                                            min_range=min_range,
                                            max_range=max_range)
            w, o, log = solver(f=f,
                               population=pop.copy(),
                               t_max=t_max,
                               mutation_value=mutation_value,
                               elite_size=population_size // 2)
            results.append((w, o))
        best_res = min(results, key=lambda x: x[1])
        avg = np.mean([y for x, y in results])
        all_data.update({population_size: {
            "best_result": best_res[1],
            "average": avg
        }})
    with open(f"results/{output_name}.json", "w") as file:
        json.dump(all_data, file)


def iterations_sizes(f, n_repeat, population_size, dimension, min_range,
                     max_range, iterations_min, iteration_max, iteration_step, mutation_value, elite_size, output_name):
    all_data = {}
    iterations = list(range(iterations_min, iteration_max + 1, iteration_step))
    for iteration_size in iterations:
        results = []
        for i in range(n_repeat):
            pop = generate_start_population(population_size=population_size,
                                            dimension=dimension,
                                            min_range=min_range,
                                            max_range=max_range)
            w, o, log = solver(f=f,
                               population=pop.copy(),
                               t_max=iteration_size,
                               mutation_value=mutation_value,
                               elite_size=elite_size
                               )
            results.append((w, o))
        best_res = min(results, key=lambda x: x[1])
        avg = np.mean([y for x, y in results])
        all_data.update({iteration_size: {
            "best_result": best_res[1],
            "average": avg
        }})
    with open(f"results/{output_name}.json", "w") as file:
        json.dump(all_data, file)


def mutation_values(f, n_repeat, population_size, dimension,
                    min_range, max_range, t_max,
                    min_mutation, mutation_count,
                    elite_size, output_name):
    all_data = {}
    mutations = [i / 2 + min_mutation for i in range(mutation_count)]
    for mutation_value in mutations:
        results = []
        for i in range(n_repeat):
            pop = generate_start_population(population_size=population_size,
                                            dimension=dimension,
                                            min_range=min_range,
                                            max_range=max_range)
            w, o, log = solver(f=f,
                               population=pop.copy(),
                               t_max=t_max,
                               mutation_value=mutation_value,
                               elite_size=elite_size
                               )
            results.append((w, o))
        best_res = min(results, key=lambda x: x[1])
        avg = np.mean([y for x, y in results])
        all_data.update({mutation_value: {
            "best_result": best_res[1],
            "average": avg
        }})
    with open(f"results/{output_name}.json", "w") as file:
        json.dump(all_data, file)


def main():
    convergence_evo(f=f1, n_repeat=25, population_size=20, dimension=10,
                    min_range=-100.0, max_range=100.0, t_max=500,
                    mutation_value=1.0, elite_size=10,
                    output_name="f1_convergence")
    convergence_evo(f=f9, n_repeat=25, population_size=20, dimension=10,
                    min_range=-100.0, max_range=100.0, t_max=500,
                    mutation_value=1.0, elite_size=10,
                    output_name="f9_convergence")

    convergence_grad(f=f1, n_repeat=25, dimension=10, t_max=500,
                     step_size=0.00000001, min_range=-100.0, max_range=100.0,
                     output_name="f1_gradient_convergence")

    convergence_grad(f=f9, n_repeat=25, dimension=10, t_max=500,
                     step_size=0.0005, min_range=-100.0, max_range=100.0,
                     output_name="f9_gradient_convergence")

    elite_sizes(f=f1, n_repeat=50,
                population_size=20, dimension=10,
                min_range=-100.0, max_range=100.0,
                t_max=500, mutation_value=1.0,
                output_name="f1_elite_sizes")
    elite_sizes(f=f9, n_repeat=50,
                population_size=20, dimension=10,
                min_range=-100.0, max_range=100.0,
                t_max=500, mutation_value=1.0,
                output_name="f9_elite_sizes")

    population_sizes(f=f1, n_repeat=50,
                     max_population_size=95,
                     population_step=10,
                     dimension=10,
                     min_range=-100.0, max_range=100.0,
                     t_max=500, mutation_value=1.0,
                     output_name="f1_populations")
    population_sizes(f=f9, n_repeat=50,
                     max_population_size=95,
                     population_step=10,
                     dimension=10,
                     min_range=-100.0, max_range=100.0,
                     t_max=500, mutation_value=1.0,
                     output_name="f9_populations")

    iterations_sizes(f=f1, n_repeat=50,
                     population_size=20,
                     dimension=10,
                     min_range=-100.0,
                     max_range=100.0,
                     iterations_min=500,
                     iteration_max=5000,
                     iteration_step=500,
                     mutation_value=1.0,
                     elite_size=10,
                     output_name="f1_iterations")
    iterations_sizes(f=f9, n_repeat=50,
                     population_size=20,
                     dimension=10,
                     min_range=-100.0,
                     max_range=100.0,
                     iterations_min=500,
                     iteration_max=5000,
                     iteration_step=500,
                     mutation_value=1.0,
                     elite_size=10,
                     output_name="f9_iterations")

    mutation_values(f=f1, n_repeat=50,
                    population_size=20,
                    dimension=10,
                    min_range=-100.0,
                    max_range=100.0,
                    t_max=500,
                    min_mutation=0.5,
                    mutation_count=20,
                    elite_size=10,
                    output_name="f1_mutation_values")

    mutation_values(f=f9, n_repeat=50,
                    population_size=20,
                    dimension=10,
                    min_range=-100.0,
                    max_range=100.0,
                    t_max=500,
                    min_mutation=0.5,
                    mutation_count=20,
                    elite_size=10,
                    output_name="f9_mutation_values")


if __name__ == "__main__":
    main()

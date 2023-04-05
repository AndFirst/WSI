import random
from numpy import ndarray
import numpy as np
import autograd as ad


def grad_solver(f, x0, iterations, step_size):
    """
    :param f: target function
    :param x0: start point (vector) of algorithm
    :param iterations: number of iterations algorithm go on
    :param step_size: size of step to next x point
    :return: final point, route of function values
    """
    grad_func = ad.grad(f)
    values = [f(x0), ]
    for i in range(iterations):
        step = step_size * grad_func(x0)
        x0 -= step
        values.append(f(x0))
    return x0, f(x0), values


def generate_start_population(population_size: int, dimension: int, min_range: float, max_range: float) -> ndarray:
    """
    :param population_size: number of individuals to create
    :param dimension: dimension of one individual
    :param min_range: lower range of draw
    :param max_range: upper range of draw
    :return: population
    """
    return np.array([np.random.uniform(min_range, max_range, dimension) for _ in range(population_size)])


def reproduction(population: ndarray, population_quality: ndarray) -> ndarray:
    """
    :param population: population to be reproduction
    :param population_quality: quality of each individual in population
    :return: population of individuals chosen in tournament

    Tournament reproduction, randomly choose two individuals and add better of them to new population.
    New population size is equal to start population size.
    """
    population_size = len(population)
    new_population = []
    for i in range(population_size):
        first_index = random.randint(0, population_size - 1)
        second_index = random.randint(0, population_size - 1)
        if population_quality[first_index] <= population_quality[second_index]:
            new_population.append(population[first_index])
        else:
            new_population.append(population[second_index])
    return np.array(new_population)


def genetic_operations(population: ndarray, mutation_value: float) -> ndarray:
    """
    :param population: population to be crossed and mutated
    :param mutation_value: standard deviation of mutation
    :return: crossed and mutated population
    """
    crossed = crossing(population)
    mutants = mutation(crossed, mutation_value)
    return mutants


def crossing(population: ndarray) -> ndarray:
    """
    :param population: population to be crossed
    :return: crossed population

    Averaging crossing. New individual is creating from two randomly chosen parents.
    Each new gene is sum of parents genes multiplied by random wages (random for each gene).
    """
    crossed_population = []
    while len(crossed_population) < len(population):
        parent_1 = random.choice(population)
        parent_2 = random.choice(population)
        wages = np.random.uniform(0, 1, len(parent_2))
        new_individual = parent_1 * wages + parent_2 * (np.ones(len(parent_2)) - wages)
        crossed_population.append(new_individual)
    return np.array(crossed_population)


def mutation(population: ndarray, mutation_value: float) -> ndarray:
    """
    :param population: population to be mutated
    :param mutation_value: standard deviation of mutation
    :return: mutated population

    Gaussian mutation. New gene is sum of gene before mutation and randomly
    chosen value from normal distribution.
    """
    mutants = []
    for individual in population:
        new_individual = individual + np.array([random.gauss(0, mutation_value) for _ in range(len(individual))])
        mutants.append(new_individual)
    return np.array(mutants)


def sort_population(population: ndarray, quality: ndarray) -> tuple[ndarray, ndarray]:
    """
    :param population: population to sort
    :param quality: qualities to sort
    :return: sorted population and sorted qualities

    Population and quality is ascending sorted by each individual quality.
    """
    zipped_lists = list(sorted(zip(quality, population), key=lambda x: x[0]))
    sorted_quality, sorted_population = [a for a, b in zipped_lists], [b for a, b in zipped_lists]
    return np.array(sorted_population), np.array(sorted_quality)


def succession(population: ndarray, mutants: ndarray, population_quality: ndarray, mutants_quality: ndarray,
               elite_size: int) -> tuple[ndarray, ndarray]:
    """
    :param population: population before genetic operations
    :param mutants: population after genetic operations
    :param population_quality: population before genetic operations quality
    :param mutants_quality: population after genetic operations quality
    :param elite_size: number of individuals from start population surviving to next iteration
    :return: new population and new population quality

    Elite succession. Returns population build from given number of the best individuals from
    start population and (population size - elite size) the best mutants.
    """
    if elite_size == 0:
        return mutants, mutants_quality

    population, population_quality = sort_population(population, population_quality)
    new_population = population[:elite_size]
    new_population_quality = population_quality[:elite_size]

    mutants, mutants_quality = sort_population(mutants, mutants_quality)
    new_population = np.concatenate((new_population, mutants[0:-elite_size]))
    new_population_quality = np.concatenate((new_population_quality, mutants_quality[0:-elite_size]))
    return new_population, new_population_quality


def find_best(population: ndarray, population_quality: ndarray) -> tuple[ndarray, float]:
    """
    :param population: population of individuals to chose best
    :param population_quality: quality of population
    :return: the best individual and the best quality

    Returns the best individual and its quality from given population.
    """
    best_quality = np.min(population_quality)
    best_index = np.argmin(population_quality)
    best_individual = population[best_index]
    return best_individual, best_quality


def calculate_quality(f: callable, population: ndarray) -> ndarray:
    """
    :param f: quality function
    :param population: population to calculate quality
    :return: quality of each individual in population
    """
    return np.array([f(individual) for individual in population])


def solver(f: callable, population: ndarray, t_max: int, mutation_value: float,
           elite_size: int) -> tuple[ndarray, float, list]:
    """
    :param f: quality function
    :param population: start population
    :param t_max: max number of iterations
    :param mutation_value: standard deviation of mutation
    :param elite_size: number of individuals from start population surviving to next iteration
    :return: the best individual, the best quality, list of the best qualities in every iteration

    Evolutionary algorithm with changing mutation value. Algorithm is executed for given number of iterations.
    Every each iteration if the best individual is changed, number of successful mutations increments.
    If proportion of successful mutation to all mutations is lower than 1/5, mutation value decreases *0.82.
    Otherwise, it increases by 1/0.82.
    """
    good_mutation_counter = 0
    t = 0
    population_quality = calculate_quality(f, population)
    best_individual, best_quality = find_best(population, population_quality)
    best_values = [(best_individual, best_quality), ]
    while t < t_max:
        current_reproduction = reproduction(population, population_quality)
        current_mutants = genetic_operations(current_reproduction, mutation_value)
        mutants_quality = calculate_quality(f, current_mutants)
        best_mutant, best_mutant_quality = find_best(current_mutants, mutants_quality)
        if best_mutant_quality <= best_quality:
            good_mutation_counter += 1
            best_individual = best_mutant
            best_quality = best_mutant_quality
        population, population_quality = succession(population, current_mutants, population_quality, mutants_quality,
                                                    elite_size)
        t += 1
        if good_mutation_counter / t < 0.2:
            mutation_value = mutation_value * 0.82
        else:
            mutation_value = mutation_value / 0.82
        best_values.append((best_individual.copy(), best_quality))
    return best_individual, best_quality, best_values

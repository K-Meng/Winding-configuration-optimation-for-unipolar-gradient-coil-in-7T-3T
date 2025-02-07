import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt import gp_minimize
from Bz_tot import Bz_tot
from inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality
from parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt

max_inductance = 300
bounds = (0, 2)
num_vars = 41

def objective_function(winding_configuration):
    diff = linearity_diff(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)


def constraint_function(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality(winding_configuration, radius, wire_radius)
    return inductance <= max_inductance


def simulated_annealing(objective_function, constraint_function, bounds, num_vars, n_iterations, step_size,
                        initial_temp, cooling_rate):
    current_solution = np.random.randint(bounds[0], bounds[1] + 1, size=num_vars)
    while not constraint_function(current_solution):
        current_solution = np.random.randint(bounds[0], bounds[1] + 1, size=num_vars)

    current_energy = objective_function(current_solution)
    best_solution = np.copy(current_solution)
    best_energy = current_energy
    temp = initial_temp

    for i in range(n_iterations):
        candidate = np.copy(current_solution)
        pos = np.random.randint(0, num_vars)
        candidate[pos] = candidate[pos] + np.random.choice([-step_size, step_size])
        candidate = np.clip(candidate, bounds[0], bounds[1])
        if not constraint_function(candidate):
            continue

        candidate_energy = objective_function(candidate)
        if candidate_energy < current_energy:
            current_solution, current_energy = candidate, candidate_energy
        else:
            diff = candidate_energy - current_energy
            metropolis = np.exp(-diff / temp)
            if np.random.rand() < metropolis:
                current_solution, current_energy = candidate, candidate_energy

        if current_energy < best_energy:
            best_solution, best_energy = current_solution, current_energy

        temp *= cooling_rate

    return best_solution, best_energy


def bayesian_annealing_search(params):
    n_iterations, step_size, initial_temp, cooling_rate = params

    best_solution, best_energy = simulated_annealing(objective_function, constraint_function, bounds, num_vars,
                                                     int(n_iterations), step_size, initial_temp, cooling_rate)
    return best_energy


search_space = [
    Integer(5000, 20000, name='n_iterations'),
    Real(0.5, 2.0, name='step_size'),
    Real(500, 2000, name='initial_temp'),
    Real(0.99, 0.999, name='cooling_rate')
]


result = gp_minimize(bayesian_annealing_search, search_space, n_calls=20, random_state=42)


best_params = result.x
print("Best parameters found:")
print("n_iterations =", best_params[0])
print("step_size =", best_params[1])
print("initial_temp =", best_params[2])
print("cooling_rate =", best_params[3])


# best_solution, best_energy = simulated_annealing(objective_function, constraint_function, bounds, num_vars,
#                                                  int(best_params[0]), best_params[1], best_params[2], best_params[3])
#
# print("Optimal winding configuration:", best_solution)
# print("Objective function value:", best_energy)
#
# inductance_final = inductance_n_loops_kaiqitst_reality(best_solution, radius, wire_radius)
# print("Inductance:", inductance_final)

import numpy as np
from T_inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality
from T_parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt
from Bz_tot import Bz_tot
from concurrent.futures import ThreadPoolExecutor

weight_f1 = 0.5
weight_f2 = 1 - weight_f1



def f1(winding_configuration):
    diff = linearity_diff(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)


def f2(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality(winding_configuration, radius, wire_radius)
    return inductance


def objective_function(winding_configuration, weight_f1, weight_f2):
    return weight_f1 * f1(winding_configuration) + weight_f2 * f2(winding_configuration)


def simulated_annealing(objective_function, candidate, bounds, num_vars, n_iterations, step_size, initial_temp, cooling_rate):
    current_solution = np.copy(candidate)
    current_energy = objective_function(current_solution, weight_f1, weight_f2)
    best_solution = np.copy(current_solution)
    best_energy = current_energy
    temp = initial_temp

    for i in range(n_iterations):
        new_candidate = np.copy(current_solution)
        pos = np.random.randint(0, num_vars)
        new_candidate[pos] = new_candidate[pos] + np.random.choice([-step_size, step_size])
        new_candidate = np.clip(new_candidate, bounds[0], bounds[1])
        new_energy = objective_function(new_candidate, weight_f1, weight_f2)

        if new_energy < current_energy:
            current_solution, current_energy = new_candidate, new_energy
        else:
            diff = new_energy - current_energy
            t = temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if np.random.rand() < metropolis:
                current_solution, current_energy = new_candidate, new_energy

        if current_energy < best_energy:
            best_solution, best_energy = current_solution, current_energy

        temp *= cooling_rate
    return best_solution, best_energy


def genetic_algorithm_with_annealing(objective_function, bounds, num_vars, pop_size, n_generations, crossover_rate, mutation_rate, n_iterations_annealing, step_size, initial_temp, cooling_rate):

    population = np.random.randint(bounds[0], bounds[1] + 1, size=(pop_size, num_vars))
    best_individual = None
    best_fitness = float('inf')

    for generation in range(n_generations):
        fitness = np.array([objective_function(individual, weight_f1, weight_f2) for individual in population])

        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_individual = population[min_fitness_idx]

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        selected_indices = np.random.choice(np.arange(pop_size), size=pop_size, replace=True, p=fitness / fitness.sum())
        selected_population = population[selected_indices]

        next_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, num_vars)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1, child2 = parent1, parent2
            next_population.extend([child1, child2])

        for individual in next_population:
            if np.random.rand() < mutation_rate:
                pos = np.random.randint(0, num_vars)
                individual[pos] = np.random.randint(bounds[0], bounds[1] + 1)

        population = []
        for individual in next_population:
            optimized_individual, optimized_fitness = simulated_annealing(objective_function, individual, bounds, num_vars, n_iterations_annealing, step_size, initial_temp, cooling_rate)
            population.append(optimized_individual)

        population = np.array(population)

    return best_individual, best_fitness


bounds = (0, 5)
num_vars = W_positions
pop_size = 20
n_generations = 4
crossover_rate = 0.7
mutation_rate = 0.1
n_iterations_annealing = 500
step_size = 2
initial_temp = 15000
cooling_rate = 0.99

best_solution, best_fitness = genetic_algorithm_with_annealing(objective_function, bounds, num_vars, pop_size, n_generations, crossover_rate, mutation_rate, n_iterations_annealing, step_size, initial_temp, cooling_rate)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)



inductance_final = inductance_n_loops_kaiqitst_reality(best_solution, radius, wire_radius)
print("Inductance:", inductance_final)


Bz_ideal_1A = -1 * strength / Imax * (z - 10)
Bz_tot_est = Bz_tot(best_solution, zspace_winding_bins, radius, z, mu_0, Imax_1a)

offset_field_ideal = Bz_ideal_1A[Iz_ROI] - np.mean(Bz_ideal_1A[Iz_ROI] - Bz_tot_est[Iz_ROI])

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)

plt.plot(z, Bz_tot_est, 'm', label='Optimized Bz', linewidth=2)
plt.plot(z[Iz_ROI], offset_field_ideal, '--', color=[1, 0.85, 0], linewidth=2, label=f'Target Bz = {strength} mT/m')
plt.plot(z[Iz_ROI], 0.5 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Bz range for linearity')
plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')

plt.ylabel('Bz (mT)')
plt.xlabel('z (m)')
plt.ylim([-0.1, 0.1])
plt.xlim([0, 0.40])
plt.title(f'Bz(mT) along central axis, I = 600A')
plt.legend()


plt.subplot(1, 2, 2)

offset_field = Bz_tot_est[Iz_ROI] - np.mean(Bz_tot_est[Iz_ROI] - Bz_ideal[Iz_ROI])
plt.plot(z[Iz_ROI], (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100, 'm', label='Bz deviation', linewidth=2)
plt.plot(z[Iz_ROI], 0.1 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Target ROI')
plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')
plt.plot(line_tube_X, 10 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range upper')
plt.plot(line_tube_X, -10 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range lower')

Bz_deviation_percentage = (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100
percentage_below_10 = np.sum(Bz_deviation_percentage < 10) / len(Bz_deviation_percentage) * 100
print(percentage_below_10)

plt.ylabel('Bz (%)')
plt.xlabel('z (m)')
plt.ylim([-20, 20])
plt.title(f'%Bz(mT) deviation along central axis, I = 600A')
plt.legend()

plt.tight_layout()
plt.show()


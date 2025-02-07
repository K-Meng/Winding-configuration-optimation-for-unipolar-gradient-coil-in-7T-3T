import numpy as np
from T_inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality
from T_parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt
from Bz_tot import Bz_tot

weight_f1 = 0.35
weight_f2 = 1 - weight_f1



def f1(winding_configuration):
    diff = linearity_diff(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)


def f2(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality(winding_configuration, radius, wire_radius)
    return inductance


def objective_function(winding_configuration, weight_f1, weight_f2):
    return weight_f1 * f1(winding_configuration) + weight_f2 * f2(winding_configuration)




def genetic_simulated_annealing(objective_function, bounds, num_vars, n_generations, population_size, crossover_rate, mutation_rate, initial_temp, cooling_rate):
    population = np.random.randint(bounds[0], bounds[1] + 1, size=(population_size, num_vars))
    best_solution = None
    best_energy = float('inf')
    temp = initial_temp

    for generation in range(n_generations):
        energies = np.array([objective_function(individual, weight_f1, weight_f2) for individual in population])

        min_energy_index = np.argmin(energies)
        if energies[min_energy_index] < best_energy:
            best_energy = energies[min_energy_index]
            best_solution = population[min_energy_index].copy()

        selected = []
        for _ in range(population_size):
            i, j = np.random.randint(0, population_size, size=2)
            if energies[i] < energies[j]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        selected = np.array(selected)

        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i+1)%population_size]
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, num_vars)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            offspring.extend([child1, child2])
        offspring = np.array(offspring)

        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                candidate = offspring[i].copy()
                pos = np.random.randint(0, num_vars)
                candidate[pos] += np.random.choice([-1, 1])
                candidate = np.clip(candidate, bounds[0], bounds[1])

                candidate_energy = objective_function(candidate, weight_f1, weight_f2)

                energy_diff = candidate_energy - energies[i]
                if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temp):
                    offspring[i] = candidate
                    energies[i] = candidate_energy

        population = offspring

        temp *= cooling_rate

        if generation % 10 == 0:
            print(f"Generation {generation}: Best Energy = {best_energy}, Temperature = {temp}")

    return best_solution, best_energy



bounds = (0, 5)
num_vars = W_positions
n_generations = 1000
population_size = 100
crossover_rate = 0.8
mutation_rate = 0.1
initial_temp = 10000
cooling_rate = 0.99

best_solution, best_energy = genetic_simulated_annealing(
    objective_function, bounds, num_vars, n_generations, population_size,
    crossover_rate, mutation_rate, initial_temp, cooling_rate
)



print("Optimal winding configuration:", best_solution)
print("Objective function value:", best_energy)

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


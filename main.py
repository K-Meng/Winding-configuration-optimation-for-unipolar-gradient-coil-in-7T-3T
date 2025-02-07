import numpy as np

from Bz_tot import Bz_tot
from inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality
from parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt

max_inductance = 300




def objective_function(winding_configuration):
    diff = linearity_diff(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)



def constraint_function(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality(winding_configuration, radius, wire_radius)
    return inductance <= max_inductance


def simulated_annealing(objective_function, constraint_function, bounds, num_vars, n_iterations, step_size, initial_temp, cooling_rate):
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
        candidate[pos] = candidate[pos] + np.random.choice([-step_size, step_size])  # 随机增加或减少一个变量的值

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


bounds = (0, 2)
num_vars = 41
n_iterations = 10000
step_size = 1
initial_temp = 1000
cooling_rate = 0.999

best_solution, best_energy = simulated_annealing(objective_function, constraint_function, bounds, num_vars,
                                                 n_iterations, step_size, initial_temp, cooling_rate)

print("Optimal winding configuration:", best_solution)
print("Objective function value:", best_energy)

inductance_final = inductance_n_loops_kaiqitst_reality(best_solution, radius, wire_radius)
print("Inductance:", inductance_final)


Bz_ideal_1A = -1 * strength / Imax * (z - 10)
Bz_tot_est = Bz_tot(best_solution, zspace_winding_bins, radius, z, mu_0, Imax_1a)
# 计算偏移磁场理想值
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
plt.xlim([-0.20, 0.20])
plt.title(f'Bz(mT) along central axis, N turns = {N}, I = 600A')
plt.legend()


plt.subplot(1, 2, 2)

offset_field = Bz_tot_est[Iz_ROI] - np.mean(Bz_tot_est[Iz_ROI] - Bz_ideal[Iz_ROI])
plt.plot(z[Iz_ROI], (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100, 'm', label='Bz deviation', linewidth=2)
plt.plot(z[Iz_ROI], 0.1 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Target ROI')
plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')
plt.plot(line_tube_X, 5 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range upper')
plt.plot(line_tube_X, -5 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range lower')

plt.ylabel('Bz (%)')
plt.xlabel('z (m)')
plt.ylim([-20, 20])
plt.title(f'%Bz(mT) deviation along central axis, N turns = {N}, I = 600A')
plt.legend()

plt.tight_layout()
plt.show()
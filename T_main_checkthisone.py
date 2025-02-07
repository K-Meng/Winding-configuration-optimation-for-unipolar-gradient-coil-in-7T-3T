import numpy as np
from T_inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality
from T_parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt
from Bz_tot import Bz_tot
from T_Uniformity import calculate_uniformity_x

weight_f1 = 0.5
weight_f2 = 1 - weight_f1
weight_f3 = 0.5



def f1(winding_configuration):
    diff = linearity_diff(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)

def f2(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality(winding_configuration, radius, wire_radius)
    return inductance

def f3(winding_configuration):
    uniformity_x = calculate_uniformity_x(winding_configuration,radius)
    return uniformity_x

def objective_function(winding_configuration, weight_f1, weight_f2):
    return weight_f1 * f1(winding_configuration) + weight_f2 * f2(winding_configuration) + weight_f3 * f3(winding_configuration)


def simulated_annealing(objective_function, bounds, num_vars, n_iterations, step_size, initial_temp, cooling_rate):
    current_solution = np.random.randint(bounds[0], bounds[1] + 1, size=num_vars)

    current_energy = objective_function(current_solution, weight_f1, weight_f2)
    best_solution = np.copy(current_solution)
    best_energy = current_energy

    temp = initial_temp

    for i in range(n_iterations):
        candidate = np.copy(current_solution)
        pos = np.random.randint(0, num_vars)
        candidate[pos] = candidate[pos] + np.random.choice([-step_size, step_size])  # 随机增加或减少一个变量的值

        candidate = np.clip(candidate, bounds[0], bounds[1])

        candidate_energy = objective_function(candidate, weight_f1, weight_f2)

        if candidate_energy < current_energy:
            current_solution, current_energy = candidate, candidate_energy
        else:
            diff = candidate_energy - current_energy
            t = temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if np.random.rand() < metropolis:
                current_solution, current_energy = candidate, candidate_energy

        if current_energy < best_energy:
            best_solution, best_energy = current_solution, current_energy

        temp *= cooling_rate

        if i % 100 == 0:
            print(f"Iteration {i}: Best Energy = {best_energy}, Current Energy = {current_energy}, Temperature = {temp}")

    return best_solution, best_energy

bounds = (0, 5)
num_vars = W_positions
n_iterations = 2000
step_size = 1
initial_temp = 1000
cooling_rate = 0.99



best_solution, best_energy = simulated_annealing(objective_function, bounds, num_vars,
                                                 n_iterations, step_size, initial_temp, cooling_rate)


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


import numpy as np
from inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality
from parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt
from Bz_tot import Bz_tot
from field_map_2D import unit_field_3D_beyond_nogrid_unipolar

weight_f1 = 0.78 # weight of linearity & efficiency
weight_f2 = 1 - weight_f1 # weight of inductance



def f1(winding_configuration):
    diff = linearity_diff(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)


def f2(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality(winding_configuration, radius, wire_radius)
    return inductance


def objective_function(winding_configuration, weight_f1, weight_f2):
    return weight_f1 * f1(winding_configuration) + weight_f2 * f2(winding_configuration)


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


# 参数设置
bounds = (0, 2)  # number of winding layers, up to 5
num_vars = int( (tube_length + wire_spacing) / (wire_outer_diameter + wire_spacing) ) # possible number of winding places along the coil in z direction
n_iterations = 4000
step_size = 1
initial_temp = 1000
cooling_rate = 0.999



best_solution, best_energy = simulated_annealing(objective_function, bounds, num_vars,
                                                 n_iterations, step_size, initial_temp, cooling_rate)


print("Optimal winding configuration:", best_solution)
# print("Objective function value:", best_energy)

inductance_final = inductance_n_loops_kaiqitst_reality(best_solution, radius, wire_radius)
print("Inductance:", inductance_final)
N = np.sum(best_solution)

Bz_ideal_1A = -1 * strength / Imax * (z - 10)
Bz_tot_est = Bz_tot(best_solution, zspace_winding_bins, radius, z, mu_0, Imax_1a)

# strength2 = 150
# Bz_ideal_1A2 = -1 * strength / Imax * (z - 10)
# Bz_tot_est2 = Bz_tot(best_solution, zspace_winding_bins, radius, z, mu_0, Imax_1a)


offset_field_ideal = Bz_ideal_1A[Iz_ROI] - np.mean(Bz_ideal_1A[Iz_ROI] - Bz_tot_est[Iz_ROI])
# offset_field_ideal2 = Bz_ideal_1A2[Iz_ROI] - np.mean(Bz_ideal_1A2[Iz_ROI] - Bz_tot_est[Iz_ROI])
# np.save('offset_field_ideal',offset_field_ideal)
# np.save('offset_field_ideal2',offset_field_ideal2)

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})


plt.figure(figsize=(15, 7))



plt.subplot(1, 2, 1)
# plt.subplot2grid((1, 5), (0, 0), colspan=3)
plt.plot(z, Bz_tot_est, 'm', label='Bz from unipolar coil', linewidth=2)
# plt.plot(z[Iz_ROI], offset_field_ideal, '--', color=[1, 0.85, 0], linewidth=2, label=f'Linear Bz = {strength} mT/m')
plt.plot(z[Iz_ROI], offset_field_ideal, '--', color=[0, 0, 1], linewidth=2, label=f'Linear Bz = {strength} mT/m')
# plt.plot(z[Iz_ROI], 0.5 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Bz range for linearity')
# plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')
plt.axvspan(xmin=0.065, xmax=0.22, color='lightblue', alpha=0.3, label='ROI')

plt.ylabel('Bz (mT)')
plt.xlabel('z (m)')
plt.ylim([0, 0.1])
plt.xlim([-0.1, 0.3])
plt.title(f'Bz(mT) 1D Profile, N turns = {N}')
plt.legend()

plt.subplot(1, 2, 2)
# plt.subplot2grid((1, 5), (0, 3), colspan=2)
# plt.subplot(1, 3, 2)

# offset_field = Bz_tot_est[Iz_ROI] - np.mean(Bz_tot_est[Iz_ROI] - Bz_ideal[Iz_ROI])
# plt.plot(z[Iz_ROI], (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100, 'm', label='Bz deviation', linewidth=2)
# plt.plot(z[Iz_ROI], 0.1 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Target ROI')
# plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')
# plt.plot(line_tube_X, 10 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range upper')
# plt.plot(line_tube_X, -10 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range lower')
#
Bz_deviation_percentage = (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100
percentage_below_10 = np.sum(Bz_deviation_percentage < 10) / len(Bz_deviation_percentage) * 100
print(percentage_below_10)
#
Bz_deviation_percentage2 = (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100
percentage_below_102 = np.sum(Bz_deviation_percentage2 < 10) / len(Bz_deviation_percentage2) * 100
print(percentage_below_102)


#
# plt.ylabel('Bz (%)')
# plt.xlabel('z (m)')
# plt.ylim([-20, 20])
# plt.title(f'%Bz(mT) deviation along central axis with {strength}, I = 600A')
# plt.legend()
#
# plt.subplot(1, 3, 3)
# Bz_ideal2 = -1 * strength2 * (z - 10)
# offset_field2 = Bz_tot_est[Iz_ROI] - np.mean(Bz_tot_est[Iz_ROI] - Bz_ideal2[Iz_ROI])
plt.plot(z[Iz_ROI], (Bz_tot_est[Iz_ROI] / offset_field_ideal - 1) * 100, 'm', label='Bz deviation', linewidth=2)
plt.plot(z[Iz_ROI], 0.1 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Target ROI')
plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')
# plt.plot(line_tube_X, 10 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range upper')
# plt.plot(line_tube_X, -10 * np.ones(len(line_tube_X)), '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range lower')
plt.plot([-0.1875,0.22],[-10,-10], '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range lower')
plt.plot([-0.1875,0.22],[10,10], '--', color=[0.5, 0.5, 0.5], linewidth=1, label='Bz range upper')

plt.ylabel('Bz (%)')
plt.xlabel('z (m)')
plt.ylim([-20, 20])
plt.title(f'%Bz(mT) 1D Deviation with {efficiency}mT/m/A')
plt.legend()

plt.tight_layout()
# plt.show(block=False)
plt.show()

#
# plt.subplot(1, 3, 3)
# x = np.arange(-0.20, 0.20, dz)
# z2 = np.arange(-0.18, 0.22, dz)
# X, Z = np.meshgrid(x, z2)
# X3D, Y3D, Z3D = np.meshgrid(x, [0], z2, indexing='ij')
# RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(best_solution, X3D.ravel(), Y3D.ravel(), Z3D.ravel())
# RealisticField = -RealisticField.reshape(X3D.shape).squeeze().T
# sinOmegaZ = RealisticField * Imax
#
# add_left = np.linspace(offset_field_ideal2[0] - (z2.size - len(offset_field_ideal2) - (z2[-1] - z[Iz_ROI[-1]])/dz - 1) * -strength2 / Imax * 1e-3, offset_field_ideal2[0], int(z2.size - len(offset_field_ideal2) - (z2[-1] - z[Iz_ROI[-1]])/dz))
# # add_left = np.linspace(0.05077206664533085 - (126 - 1) * -0.00025, 0.05077206664533085, 125)
# # add_right = np.linspace(0.01227206664533087, 0.01227206664533087 + 0.00025 * (120 - 1), 120)
# add_right = np.linspace(offset_field_ideal2[-1], offset_field_ideal2[-1] + strength2 / Imax * 1e-3 * ((z2[-1] - z[Iz_ROI[-1]])/dz - 1), int((z2[-1] - z[Iz_ROI[-1]])/dz))
# offset_field_ideal2_long = np.concatenate([add_left, offset_field_ideal2, add_right])
#
# offset_field_ideal2_2D = np.tile(offset_field_ideal2_long, (400, 1)).T
# deviation_map = (sinOmegaZ / offset_field_ideal2_2D - 1) * 100
# deviation_map = np.clip(deviation_map, -60, 60)
#
# xmin, xmax = x[0], x[-1]
# zmin, zmax = z2[0], z2[-1]
# plt.imshow(deviation_map, cmap='coolwarm', interpolation='none', extent=[xmin, xmax, zmin, zmax])
# plt.colorbar()
# # plt.ylim([-0.06, 0.34])
# # plt.xlim([-0.16, 0.16])
# plt.title("Deviation Map")
# plt.tight_layout()
# plt.show(block=False)
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(offset_field_ideal2_2D, cmap='coolwarm', interpolation='none', extent=[xmin, xmax, zmin, zmax])
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(sinOmegaZ , cmap='coolwarm', interpolation='none', extent=[xmin, xmax, zmin, zmax])
# plt.colorbar()
# plt.tight_layout()
# plt.show()



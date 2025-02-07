import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from Bz_tot_bys import Bz_tot_bys
from inductance_n_loops_kaiqitst_reality_bys import inductance_n_loops_kaiqitst_reality_bys
from parameters import *
from linearity_diff_bys import linearity_diff_bys
import matplotlib.pyplot as plt


def f1(winding_configuration):
    diff = linearity_diff_bys(winding_configuration, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal)
    return abs(diff)



def calculate_inductance(winding_configuration):
    inductance = inductance_n_loops_kaiqitst_reality_bys(winding_configuration, radius, wire_radius)
    return inductance



dimensions = [Integer(0, 2, name=f'winding_{i}') for i in range(41)]



@use_named_args(dimensions=dimensions)
def objective_function(**winding_configuration):
    # 将 winding_configuration 转换为一个列表以传递给其他函数
    winding_configuration_list = [winding_configuration[f'winding_{i}'] for i in range(41)]

    inductance = calculate_inductance(winding_configuration_list)


    if inductance > 300:
        return 1e6


    return f1(winding_configuration_list)



result = gp_minimize(objective_function, dimensions, n_calls=100, random_state=0)


result.x = np.array(result.x).tolist()
print("Optimal winding configuration:", result.x)
print("Objective function value (diff):", result.fun)


inductance_final = calculate_inductance(result.x)
print("Inductance:", inductance_final)


Bz_ideal_1A = -1 * strength / Imax * (z - 10)
Bz_tot_est = Bz_tot_bys(result.x, zspace_winding_bins, radius, z, mu_0, Imax_1a)

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
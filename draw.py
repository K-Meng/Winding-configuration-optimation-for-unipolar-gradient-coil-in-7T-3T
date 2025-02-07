import numpy as np

from Bz_tot import Bz_tot
from inductance_n_loops_kaiqitst_reality import inductance_n_loops_kaiqitst_reality

from parameters import *
from linearity_diff import linearity_diff
import matplotlib.pyplot as plt



Bz_ideal_1A = -1 * strength / Imax * (z - 10)
best_solution = np.array([0, 0, 1, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,1, 0, 0, 1, 0, 2, 0, 1, 0, 1, 2, 1, 0 ,0 ,2 ,2 ,1 ,2 ,2 ,2 ,2])
solution0 = np.array([1, 0, 0 ,0 ,0 ,1 ,0 ,0 ,1, 0, 0 ,2 ,0 ,1, 2, 0, 1 ,1 ,1 ,0, 0, 1, 0, 2 ,0 ,2 ,0 ,1 ,2, 1, 2, 0 ,1, 2, 1, 0 ,2, 0 ,1 ,2])
Bz_tot_0 = Bz_tot(solution0, zspace_winding_bins, radius, z, mu_0, Imax_1a)
Bz_tot_est = Bz_tot(best_solution, zspace_winding_bins, radius, z, mu_0, Imax_1a)

offset_field_ideal = Bz_ideal_1A[Iz_ROI] - np.mean(Bz_ideal_1A[Iz_ROI] - Bz_tot_est[Iz_ROI])


plt.figure(figsize=(12, 6))



plt.subplot(1, 2, 1)

plt.plot(z, Bz_tot_est, 'm', label='Optimized Bz', linewidth=2)
plt.plot(z[Iz_ROI], offset_field_ideal, '--', color=[1, 0.85, 0], linewidth=2, label=f'Target Bz = {strength} mT/m')
plt.plot(z[Iz_ROI], 0.5 * np.ones(len(Iz_ROI)), '--', color=[0, 1, 0], linewidth=2, label='Bz range for linearity')
plt.plot(line_tube_X, np.zeros(len(line_tube_X)), 'k', linewidth=2, label='Tube length')
plt.plot(z, Bz_tot_0, 'r', label='Optimized Bz', linewidth=2)

plt.ylabel('Bz (mT)')
plt.xlabel('z (m)')
plt.ylim([-0.1, 0.1])
plt.xlim([-0.20, 0.20])
plt.title(f'Bz(mT) along central axis, N turns = , I = 600A')
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
plt.title(f'%Bz(mT) deviation along central axis, N turns =, I = 600A')
plt.legend()

plt.tight_layout()
plt.show()
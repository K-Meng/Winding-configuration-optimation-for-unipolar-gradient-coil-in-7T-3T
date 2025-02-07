
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from field_map_2D import unit_field_3D_beyond_nogrid_unipolar

z_est = np.arange(-5e-2, 5e-2 + 1e-3, 1e-3)
X3D, Y3D, Z3D = np.meshgrid([0], [0], z_est, indexing='ij')


wbins = [0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0 ,0 ,1 ,1 ,0, 0, 0, 0, 0 ,0 ,0 ,1, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 2 ,1 ,0 ,2 ,1 ,2,2 ,2, 2, 2]
wbins = np.array(wbins)


RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(wbins, X3D.ravel(), Y3D.ravel(), Z3D.ravel())


GradientFit = np.polyfit(z_est.ravel(), -RealisticField.ravel(), 1)
ScalingFactor = GradientFit[0]

x = np.linspace(-0.15, 0.15, 100)  # 假设 x 范围为 [-0.15, 0.15]
z = np.linspace(-0.3, 0.3, 100)    # 假设 z 范围为 [-0.3, 0.3]
X3D, Y3D, Z3D = np.meshgrid(x, [0], z, indexing='ij')

shift_coils = 0.01
RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(wbins, X3D.ravel(), Y3D.ravel(), Z3D.ravel() + shift_coils)

RealisticField = -RealisticField.reshape(X3D.shape).squeeze().T

sinOmegaZ = RealisticField / ScalingFactor

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

from matplotlib import colormaps
cmap = colormaps.get_cmap('coolwarm')

surf = axs[0].imshow(sinOmegaZ, extent=[x.min(), x.max(), z.min(), z.max()], origin='lower', cmap=cmap, aspect='auto')
axs[0].set_title('Realistic gradient field')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('z [m]')
fig.colorbar(surf, ax=axs[0], label='G_z [normalized]')

center_idx = len(x) // 2
axs[1].plot(z, sinOmegaZ[:, center_idx], linewidth=1.5)
axs[1].set_title('line through center')
axs[1].set_xlabel('z [m]')
axs[1].set_ylabel('G_z [normalized]')
axs[1].axis('square')
axs[1].grid(True)

plt.tight_layout()
plt.show()


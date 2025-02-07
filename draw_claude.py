import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorcet as cc
from field_map_2D import unit_field_3D_beyond_nogrid_unipolar

x = np.arange(-0.2, 0.2, 0.001)
z = np.arange(-0.3, 0.1, 0.001)
X, Z = np.meshgrid(x, z)
X3D, Y3D, Z3D = np.meshgrid(x, [0], z, indexing='ij')

wbins = [0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0 ,0 ,1 ,1 ,0, 0, 0, 0, 0 ,0 ,0 ,1, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 2 ,1 ,0 ,2 ,1 ,2,2 ,2, 2, 2]
wbins = np.array(wbins)

RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(wbins, X3D.ravel(), Y3D.ravel(), Z3D.ravel())


RealisticField = -RealisticField.reshape(X3D.shape).squeeze().T  # 数据翻转

sinOmegaZ = RealisticField * 1000

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, sinOmegaZ, edgecolor='none')
ax.set_box_aspect((np.ptp(X), np.ptp(Z), np.ptp(sinOmegaZ)))
ax.view_init(90, -90)
plt.colorbar(surf, label='G_z [normalized]')
plt.ylabel('z [m]')
plt.xlabel('x [m]')
plt.title('Realistic gradient field')
plt.gca().set_aspect('equal')
plt.gca().tick_params(labelsize=10)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)

strength_ideal = -0.00025
field_ideal = 0 + np.arange(400) * strength_ideal
move_ideal_field = field_ideal + 0.152
move_ideal_field2 = move_ideal_field.reshape(-1, 1)
move_ideal_field3 = np.tile(move_ideal_field2, (1, 400))

# Plot Ideal gradient field
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, move_ideal_field3, edgecolor='none')
ax.set_box_aspect((np.ptp(X), np.ptp(Z), np.ptp(move_ideal_field3)))
ax.view_init(90, -90)
plt.colorbar(surf)
plt.ylabel('z [m]')
plt.xlabel('x [m]')
plt.title('Ideal gradient field')
plt.gca().set_aspect('equal')
plt.gca().tick_params(labelsize=10)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)

# Calculate and plot deviation map
deviation_map = (sinOmegaZ / move_ideal_field3 - 1) * 100
deviation_map = np.clip(deviation_map, -60, 60)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, deviation_map, edgecolor='none')
ax.set_box_aspect((np.ptp(X), np.ptp(Z), np.ptp(deviation_map)))
ax.view_init(90, -90)
plt.colorbar(surf)
plt.ylabel('z [m]')
plt.xlabel('x [m]')
plt.title('deviation map')
plt.gca().set_aspect('equal')
plt.gca().tick_params(labelsize=10)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)

# Calculate and plot deviation map region
move_ideal_field3[move_ideal_field3 < 0.01] = 0.01
deviation_map = (sinOmegaZ / move_ideal_field3 - 1) * 100

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, deviation_map, edgecolor='none')
ax.set_box_aspect((np.ptp(X), np.ptp(Z), np.ptp(deviation_map)))
ax.view_init(90, -90)
plt.colorbar(surf)
plt.ylabel('z [m]')
plt.xlabel('x [m]')
plt.title('deviation map region')
plt.gca().set_aspect('equal')
plt.gca().tick_params(labelsize=10)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorcet as cc
from field_map_2D import unit_field_3D_beyond_nogrid_unipolar
from create_zspace_radius_vector import create_zspace_radius_vector
from parameters import *



x = np.arange(-0.2, 0.2, 0.001)
z = np.arange(-0.3, 0.2, 0.001)
X, Z = np.meshgrid(x, z)
X3D, Y3D, Z3D = np.meshgrid(x, [0], z, indexing='ij')

wbins = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ,0 ,0, 0, 0, 1, 0 ,0 ,0 ,0 ,2 ,0 ,0 ,1 ,0 ,1 ,0 ,2 ,0 ,0 ,1, 2, 2 ,1, 2 ,2, 2 ,2, 2]
wbins = np.array(wbins)

Positions, radius_vector = create_zspace_radius_vector(zspace_winding_bins, wbins, radius)

RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(wbins, X3D.ravel(), Y3D.ravel(), Z3D.ravel())

RealisticField = -RealisticField.reshape(X3D.shape).squeeze().T

sinOmegaZ = RealisticField * 1000

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, sinOmegaZ, cmap=cc.cm['CET_D1'], edgecolor='none')
ax.view_init(elev=90, azim=-90)  # 设置视角
fig.colorbar(surf, ax=ax, label='G_z [normalized]')
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_zlim(-0.3, 0.3)
ax.set_title('Realistic gradient field')

for i, position in enumerate(Positions):
    theta = np.linspace(0, 2 * np.pi, 100)
    radius1 = radius_vector[i]
    X_circle = radius1 * np.cos(theta)
    Z_circle = radius1 * np.sin(theta)
    Y_circle = np.full_like(X_circle, -position)
    ax.plot(X_circle, Y_circle, Z_circle, color='k', linewidth=1)

plt.show(block = False)

strength_ideal = -0.00025
field_ideal = 0 + np.arange(500) * strength_ideal
move_ideal_field = field_ideal + 0.152
move_ideal_field2 = move_ideal_field.reshape(-1, 1)
move_ideal_field3 = np.tile(move_ideal_field2, (1, 400))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, move_ideal_field3, cmap=cc.cm['CET_D1'], edgecolor='none')
ax.view_init(elev=90, azim=-90)
fig.colorbar(surf, ax=ax)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_title('Ideal gradient field')
plt.show(block = False)

deviation_map = (sinOmegaZ / move_ideal_field3 - 1) * 100
deviation_map = np.clip(deviation_map, -60, 60)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, deviation_map, cmap=cc.cm['CET_D1'], edgecolor='none')
ax.view_init(elev=90, azim=-90)
fig.colorbar(surf, ax=ax)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_title('Deviation map')
plt.show(block = False)

# with np.errstate(divide='ignore', invalid='ignore'):
#     deviation_map = (sinOmegaZ / move_ideal_field3 - 1) * 100
#     deviation_map = np.nan_to_num(deviation_map, nan=0, posinf=60, neginf=-60)

# move_ideal_field3[move_ideal_field3 < 0.01] = 0.01
deviation_map = (sinOmegaZ / move_ideal_field3 - 1) * 100

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, deviation_map, cmap=cc.cm['CET_D1'], edgecolor='none')
ax.view_init(elev=90, azim=-90)
fig.colorbar(surf, ax=ax)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_title('Deviation map region')
plt.show()

# fig, ax = plt.subplots()
#
# img = ax.imshow(deviation_map, cmap='viridis', origin='lower',
#                 extent=[X.min(), X.max(), Z.min(), Z.max()])
#
# fig.colorbar(img, ax=ax)
#
# ax.set_xlabel('x [m]')
# ax.set_ylabel('z [m]')
# ax.set_title('Deviation map region')

# plt.show()
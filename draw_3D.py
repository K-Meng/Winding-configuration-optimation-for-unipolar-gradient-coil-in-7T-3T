import numpy as np
import matplotlib.pyplot as plt
from fontTools.unicodedata import block
from mpl_toolkits.mplot3d import Axes3D
import colorcet as cc
from field_map_2D import unit_field_3D_beyond_nogrid_unipolar
from create_zspace_radius_vector import create_zspace_radius_vector
from parameters import radius, zspace_winding_bins
from create_sphere import create_sphere
# from main_without_constrain import best_solution, offset_field_ideal, offset_field_ideal2

offset_field_ideal = np.load('offset_field_ideal.npy')
offset_field_ideal2 = np.load('offset_field_ideal2.npy')

# x = np.arange(-0.2, 0.2, 0.001)
# z = np.arange(-0.3, 0.2, 0.001)
x = np.linspace(-0.2, 0.2, 400)
z = np.linspace(-0.3, 0.2, 500)
X, Z = np.meshgrid(x, z)
X3D, Y3D, Z3D = np.meshgrid(x, [0], z, indexing='ij')

wbins = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0 ,0 ,0 ,0 ,0 ,1 ,0 ,0, 0, 0 ,0 ,1 ,0 ,1 ,0 ,0, 1, 1, 0 ,0 ,1 ,1 ,1 ,1 ,2 ,2, 1 ,2 ,1, 2, 2, 2]
# wbins = [1 ,0, 0, 0, 0, 0 ,0 ,1 ,0, 0 ,0 ,0 ,0, 1 ,0 ,0 ,0, 0 ,0 ,0 ,1 ,1 ,1, 0, 1, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 2 ,2 ,2 ,2, 2, 2]
# wbins = best_solution
wbins = np.array(wbins)

Positions, radius_vector = create_zspace_radius_vector(zspace_winding_bins, wbins, radius)


RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(wbins, X3D.ravel(), Y3D.ravel(), Z3D.ravel())


RealisticField = -RealisticField.reshape(X3D.shape).squeeze().T

sinOmegaZ = RealisticField * 1000

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, sinOmegaZ, cmap=cc.cm['CET_D1'], edgecolor='none', vmin =-0.2 , vmax =0.2 )
surf = ax.plot_surface(X, Z, sinOmegaZ, cmap=cc.cm['CET_D1'], edgecolor='none')
ax.view_init(elev=90, azim=-90)
fig.colorbar(surf, ax=ax, label='G_z [normalized]')
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_zlabel('strength')
ax.set_zlim(-0.3, 0.3)
ax.set_box_aspect([1, 1, 1.5])
ax.set_title('Realistic gradient field')

for i, position in enumerate(Positions):
    theta = np.linspace(0, 2 * np.pi, 100)
    radius1 = radius_vector[i]
    X_circle = radius1 * np.cos(theta)
    Z_circle = radius1 * np.sin(theta)
    Y_circle = np.full_like(X_circle, -position)
    ax.plot(X_circle, Y_circle, Z_circle, color='k', linewidth=1)

# radius = 0.08
# center = [0, -0.14, 0]
# x_sphere, y_sphere, z_sphere = create_sphere(radius, center)
# ax.plot_surface(x_sphere, y_sphere, z_sphere, color='w', edgecolor='none')

plt.show(block = False)

x_index = np.argmin(np.abs(X[0, :]))

Z_slice = Z[:, x_index]
sinOmegaZ_slice = sinOmegaZ[:, x_index]

plt.figure()
plt.plot(Z_slice, sinOmegaZ_slice)
plt.xlabel('z [m]')
plt.ylabel('G_z [normalized]')
plt.title('YZ plane at X=0')
plt.grid(True)


strength_ideal = 0.00025
field_ideal = 0 + np.arange(500) * strength_ideal
Iz_ROI2 = np.where((z >= -0.22) & (z <= -0.065))[0]
offset_field_ideal = field_ideal - np.mean(field_ideal[Iz_ROI2] - sinOmegaZ_slice[Iz_ROI2])
move_ideal_field2 = offset_field_ideal.reshape(-1, 1)
move_ideal_field3 = np.tile(move_ideal_field2, (1, 400))
# move_ideal_field2 = offset_field_ideal.reshape(-1, 1)
# move_ideal_field3 = np.tile(move_ideal_field2, (1, 400))

plt.plot(z, move_ideal_field3)
plt.xlabel('z [m]')
plt.ylabel('Field Value at x=0')
plt.title('YZ Plane at X=0')
plt.grid(True)
plt.show(block = False)

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
deviation_map = np.clip(deviation_map, -20, 20)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, deviation_map, cmap=cc.cm['CET_D1'], edgecolor='none')
# ax.view_init(elev=90, azim=-90)
# fig.colorbar(surf, ax=ax)
# ax.set_xlabel('x [m]')
# ax.set_ylabel('z [m]')
# ax.set_title('Deviation map')
#
# contour_levels = [-10, 10]
# ax.contour(X, Z, deviation_map, levels=contour_levels, colors='k', linestyles='--', linewidths=2.5)

circle_center_x = 0
circle_center_y = -0.14
radius = 0.08
fig, ax = plt.subplots()
c = ax.imshow(deviation_map, cmap=cc.cm['CET_D1'], extent=[X.min(), X.max(), Z.min(), Z.max()], origin='lower', aspect='auto')
fig.colorbar(c, ax=ax, label='Deviation')
circle = plt.Circle((circle_center_x, circle_center_y), radius, color='k', fill=False, linewidth=2.5, linestyle='--')
# contour_levels = [-10, 10]
# ax.contour(X, Z, deviation_map, levels=contour_levels, colors='k', linestyles='--', linewidths=2.5)
ax.add_patch(circle)
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_title('Deviation within the head')
plt.show(block = False)

distance_from_center = np.sqrt((X - circle_center_x)**2 + (Z - circle_center_y)**2)
deviation_map_masked = np.where(distance_from_center <= radius, deviation_map, np.nan)
condition = (deviation_map_masked >= -10) & (deviation_map_masked <= 10)
dx = np.abs(X[0, 1] - X[0, 0])
dz = np.abs(Z[1, 0] - Z[0, 0])
grid_area = dx * dz
area_within_range = np.sum(condition) * grid_area
total_circle_area = np.pi * radius**2
percentage_within_range = (area_within_range / total_circle_area) * 100
print(f"the area of percentage of -10 to 10 in the ROI: {percentage_within_range:.2f}%")
# def compute_rotational_volume(X, Z, condition, dx, dz):
#     # volume： 2 * pi * sum( Z * x * dx * dz)
#     volume = 0
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             if condition[i, j]:
#                 volume += 2 * np.pi * np.abs(X[i, j]) * np.abs(Z[i, j]) * dx * dz
#     return volume
#
# total_volume = compute_rotational_volume(X, Z, np.ones_like(deviation_map, dtype=bool), dx, dz)
#
# selected_volume = compute_rotational_volume(X, Z, condition, dx, dz)
#
# percentage_volume = (selected_volume / total_volume) * 100
# print(f"the volume of percentage of -10 to 10 in the sphere: {percentage_volume:.2f}%")

# set values smaller than 0.01 as 0.01
move_ideal_field3[move_ideal_field3 < 0.01] = 0.01
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

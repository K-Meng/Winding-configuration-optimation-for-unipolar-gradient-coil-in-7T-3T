import numpy as np

from field_map_2D import unit_field_3D_beyond_nogrid_unipolar
from create_zspace_radius_vector import create_zspace_radius_vector
from T_parameters import radius, zspace_winding_bins

def calculate_uniformity_x(wbins,radius):
    x = np.linspace(-0.2, 0.2, 400)
    z = np.linspace(-0.3, 0.2, 500)
    X, Z = np.meshgrid(x, z)
    X3D, Y3D, Z3D = np.meshgrid(x, [0], z, indexing='ij')

    wbins = np.array(wbins)

    Positions, radius_vector = create_zspace_radius_vector(zspace_winding_bins, wbins, radius)


    RealisticField, _, _, _, _ = unit_field_3D_beyond_nogrid_unipolar(wbins, X3D.ravel(), Y3D.ravel(), Z3D.ravel())


    RealisticField = -RealisticField.reshape(X3D.shape).squeeze().T  # 数据翻转

    sinOmegaZ = RealisticField * 1000

    circle_center_x = 0
    circle_center_y = -0.22
    radius = 0.08


    distance_from_center = np.sqrt((X - circle_center_x) ** 2 + (Z - circle_center_y) ** 2)
    mask = distance_from_center <= radius
    sinOmegaZ_in_circle = sinOmegaZ[mask]

    X_in_circle = X[mask]
    Z_in_circle = Z[mask]

    unique_x_positions = np.unique(X_in_circle)
    total_uniformity_sum = 0

    for x in unique_x_positions:
       sinOmegaZ_row = sinOmegaZ_in_circle[X_in_circle == x]
       row_uniformity = np.std(sinOmegaZ_row)
       total_uniformity_sum += row_uniformity

    return total_uniformity_sum
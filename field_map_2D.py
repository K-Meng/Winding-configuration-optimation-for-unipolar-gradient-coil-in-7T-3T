import numpy as np
from parameters import *
from create_zspace_radius_vector import create_zspace_radius_vector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def surf(X, Y, Z, C=None, cmap='viridis'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if C is None:
        C = Z

    # 绘制表面图
    surf_plot = ax.plot_surface(X, Y, Z, facecolors=plt.cm.get_cmap(cmap)(C), rstride=1, cstride=1, antialiased=True)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(C)
    fig.colorbar(mappable)

    plt.show()


def ellipke(m, tol=None):

    if np.any(m < 0) or np.any(m > 1):
        raise ValueError("Input m must be in the range [0, 1].")

    if tol is None:
        tol = np.finfo(m.dtype).eps if isinstance(m, np.ndarray) else np.finfo(float).eps

    a0 = 1.0
    b0 = np.sqrt(1 - m)
    c0 = np.nan
    s0 = m
    i1 = 0
    mm = np.inf

    while mm > tol:
        a1 = (a0 + b0) / 2
        b1 = np.sqrt(a0 * b0)
        c1 = (a0 - b0) / 2
        i1 += 1
        w1 = 2 ** i1 * c1 ** 2
        mm = np.max(w1)

        if np.array_equal(c0, c1):  # Check for stagnation
            raise RuntimeError("Failed to converge")

        s0 += w1
        a0 = a1
        b0 = b1
        c0 = c1

    k = np.pi / (2 * a1)
    e = k * (1 - s0 / 2)

    im = (m == 1)
    e[im] = 1.0
    k[im] = np.inf

    return k, e


def analytical_field_loop_coil_nogrid(Current, position, radius, X3d, Y3d, Z3d):
    # Analytical calculation of field for a loop coil
    Z3d = Z3d + position

    r_sq = X3d**2 + Y3d**2 + Z3d**2
    alpha_sq = radius**2 + r_sq - 2 * radius * np.sqrt(X3d**2 + Y3d**2)
    beta_sq = radius**2 + r_sq + 2 * radius * np.sqrt(X3d**2 + Y3d**2)
    k_sq = 1 - alpha_sq / beta_sq
    K, E = ellipke(k_sq)

    d = radius * np.sqrt(3/2)
    m_o = 4 * np.pi * 10**(-7)
    C = m_o * Current / np.pi

    B_z_output = (C / (2 * alpha_sq * np.sqrt(beta_sq))) * ((radius**2 - r_sq) * E + alpha_sq * K)
    B_x_output = (C * X3d * Z3d) / (2 * alpha_sq * np.sqrt(beta_sq) * (X3d**2 + Y3d**2)) * ((radius**2 + r_sq) * E - alpha_sq * K)
    B_y_output = (C * Y3d * Z3d) / (2 * alpha_sq * np.sqrt(beta_sq) * (X3d**2 + Y3d**2)) * ((radius**2 + r_sq) * E - alpha_sq * K)

    return B_z_output, B_y_output, B_x_output, X3d, Y3d, Z3d

def unit_field_3D_beyond_nogrid_unipolar(wbins, X3d, Y3d, Z3d):
    # Unit field distribution in 3D for unit current
    # Returns B_grad_z, x, y, z, Total_field (B-field in x, y, z direction in T)
    Positions, radius_vector = create_zspace_radius_vector(zspace_winding_bins, wbins, radius)

    # Calculate the number of turns in each layer
    N1 = np.count_nonzero(wbins)
    wbins = wbins - 1
    wbins[wbins < 0] = 0
    N2 = np.count_nonzero(wbins)
    wbins = wbins - 1
    wbins[wbins < 0] = 0
    N3 = np.count_nonzero(wbins)

    N = N1 + N2 + N3

    # Current_strength = np.ones_like(Positions)
    Current_strength = np.sign(Positions)

    # Pre-allocation
    Coils_z = np.zeros((X3d.size, len(Positions)))
    Coils_x = np.zeros((X3d.size, len(Positions)))
    Coils_y = np.zeros((X3d.size, len(Positions)))


    for i, position in enumerate(Positions):
        I = Current_strength[i]
        Coils_z[:, i], Coils_x[:, i], Coils_y[:, i], x, y, z = analytical_field_loop_coil_nogrid(I, position,
                                                                                                 radius_vector[i], X3d, Y3d,
                                                                                                 Z3d)

    Total_field = np.zeros((X3d.size, 3))
    Total_field[:, 0] = np.sum(Coils_x, axis=1)  # x-component B-field
    Total_field[:, 1] = np.sum(Coils_y, axis=1)  # y-component B-field
    Total_field[:, 2] = np.sum(Coils_z, axis=1)  # z-component B-field

    B_grad_z = Total_field[:, 2]

    return B_grad_z, x, y, z, Total_field

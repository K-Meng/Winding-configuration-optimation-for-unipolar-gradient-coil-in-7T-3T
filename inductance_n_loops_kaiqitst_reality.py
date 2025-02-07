import numpy as np
from scipy.special import comb
from parameters import *
from create_zspace_radius_vector import create_zspace_radius_vector
from itertools import combinations


def mutual_inductance_circular_coils(r1, r2, dist):
    """
    Compute the mutual inductance between two circular coils.
    Radius and distance in meters, output in microhenries (uH).
    """

    # King et al., computation of mutual inductance using series expansion, recursive coefficients
    a = np.sqrt((r1 + r2) ** 2 + dist ** 2)
    b = np.sqrt((r1 - r2) ** 2 + dist ** 2)
    c = (a - b)
    ci = 1
    cs = c ** 2
    co = c + 0.00001  # Add a small value to start the while loop, no consequence

    for t in range(7):
        ao = (a + b) / 2
        b = np.sqrt(a * b)
        a = ao
        co = c
        c = (a - b)
        ci *= 2
        cs += ci * c ** 2

    Mij = 0.05 * np.pi ** 2 * cs / a  # in uH

    return Mij


def inductance_n_loops_kaiqitst_reality(wbins, radius, wire_radius):
    zspace, radius_vector = create_zspace_radius_vector(zspace_winding_bins, wbins, radius)

    # Calculate the number of turns in each layer
    N1 = np.count_nonzero(wbins)
    wbins = wbins - 1
    wbins[wbins < 0] = 0
    N2 = np.count_nonzero(wbins)
    wbins = wbins - 1
    wbins[wbins < 0] = 0
    N3 = np.count_nonzero(wbins)
    wbins = wbins - 1
    wbins[wbins < 0] = 0
    N4 = np.count_nonzero(wbins)
    wbins = wbins - 1
    wbins[wbins < 0] = 0
    N5 = np.count_nonzero(wbins)

    N = N1 + N2 + N3 + N4 + N5

    # Use itertools.combinations to generate combinations of indices
    coil_combinations = np.array(list(combinations(range(1, N + 1), 2)))
    signMatrix = np.ones((N, N))

    sign_list = signMatrix[coil_combinations[:, 0] - 1, coil_combinations[:, 1] - 1]

    zspace_all = np.array(zspace)
    radius_all = np.array(radius_vector)

    dist_matrix = np.abs(zspace_all[coil_combinations[:, 0] - 1] - zspace_all[coil_combinations[:, 1] - 1])

    radius_list = np.vstack((radius_all[coil_combinations[:, 0] - 1], radius_all[coil_combinations[:, 1] - 1])).T

    # Compute self inductance of all coils
    L0 = (2 * N1 * mutual_inductance_circular_coils(radius[0], radius[0], 0.7788 * wire_radius) +
          2 * N2 * mutual_inductance_circular_coils(radius[1], radius[1], 0.7788 * wire_radius) +
          2 * N3 * mutual_inductance_circular_coils(radius[2], radius[2], 0.7788 * wire_radius) +
          2 * N4 * mutual_inductance_circular_coils(radius[2], radius[2], 0.7788 * wire_radius) +
          2 * N5 * mutual_inductance_circular_coils(radius[2], radius[2], 0.7788 * wire_radius))
    L0 = L0 / 2

    # Compute mutual inductance of all coil combinations
    Mij = np.zeros(len(dist_matrix))
    for i in range(len(dist_matrix)):
        Mij[i] = mutual_inductance_circular_coils(radius_list[i, 0], radius_list[i, 1], dist_matrix[i])

    # Compute total inductance
    L = L0 + 2 * np.sum(Mij * sign_list)


    return L

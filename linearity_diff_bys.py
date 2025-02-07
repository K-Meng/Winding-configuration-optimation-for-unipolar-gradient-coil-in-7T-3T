import numpy as np

from create_zspace_radius_vector_bys import create_zspace_radius_vector_bys


def linearity_diff_bys(optim_vars, zspace_winding_bins, radius, z, mu_0, Imax, Iz_ROI, Bz_ideal):
    wbins = optim_vars
    zspace, radius_vector = create_zspace_radius_vector_bys(zspace_winding_bins, wbins, radius)

    Maxlinearity = 10
    Bz_left = np.zeros((len(zspace), len(z)))

    for i in range(len(zspace)):
        Bz_left[i, :] = (1 / (((z - zspace[i]) ** 2 + radius_vector[i] ** 2) ** (3/2)) *
                         mu_0 / 2 * radius_vector[i] ** 2 * Imax * 1e3)  # mT/m

    Bz_tot = np.sum(Bz_left, axis=0)

    offset_field_ideal = Bz_ideal[Iz_ROI] - np.mean(Bz_ideal[Iz_ROI] - Bz_tot[Iz_ROI])

    linearity = np.max(np.abs((Bz_tot[Iz_ROI] / offset_field_ideal - 1) * 100)) - Maxlinearity

    return linearity


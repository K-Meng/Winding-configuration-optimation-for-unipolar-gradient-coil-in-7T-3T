import numpy as np


def create_zspace_radius_vector(zspace_winding_bins, wbins, radius):
    zspace_vector = []
    radius_vector = []
    k = 0

    while np.sum(wbins) > 0:
        # Extract zspace values where wbins is non-zero
        zspace_vector.extend(zspace_winding_bins[np.nonzero(wbins)])

        # Append corresponding radius values
        radius_vector.extend([radius[k]] * np.sum(wbins > 0))

        # Update wbins by subtracting 1 where it's non-zero
        wbins = wbins - (wbins > 0)
        k += 1

    return np.array(zspace_vector), np.array(radius_vector)

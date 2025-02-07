import numpy as np

def create_zspace_radius_vector_bys(zspace_winding_bins, wbins, radius):
    zspace_vector = []
    radius_vector = []
    k = 0

    wbins = np.array(wbins)

    while np.sum(wbins) > 0:
        zspace_vector.extend(zspace_winding_bins[np.nonzero(wbins)])

        radius_vector.extend([radius[k]] * np.sum(wbins > 0))

        wbins = wbins - 1
        k += 1

    return np.array(zspace_vector), np.array(radius_vector)



import numpy as np

strength = 150
Imax = 600  # Amperes, current
efficiency = strength / Imax
Imax_1a = 1
# U = 800

mu_0 = 4 * np.pi * 1e-7
density_copper = 8.9  # kg/dm^3
rho_copper = 1.68e-8  # resistivity, ohm m

tube_length = 0.1875
tube_radius = 0.17  # m

wire_outer_diameter = 2.6 / 1000
wire_inner_diameter = 0 / 1000
wire_spacing = 2 / 1000  # convert mm to m

winding_spacing = (wire_spacing + wire_outer_diameter)  # m

wire_surface = wire_outer_diameter**2 - wire_inner_diameter**2  # m^2, for square shaped wire

wire_radius_square = np.sqrt(wire_surface / np.pi)  # m, for equivalent circular shaped wire
wire_radius = wire_outer_diameter/2
# wire_length = 32  # m
# wire_weight = wire_surface * 1e-4 * wire_length * 1e1 * density_copper  # kg

radius = [
    tube_radius,
    tube_radius + 2 * wire_radius + wire_spacing,
    tube_radius + 4 * wire_radius + 2 * wire_spacing,
    tube_radius + 6 * wire_radius + 3 * wire_spacing,
    tube_radius + 8 * wire_radius + 4 * wire_spacing
]

# Field along central axis
I = np.array([Imax, -Imax])
dz = 0.001
z = np.arange(-tube_length - 0.1, 0.7, dz)
# z = np.arange(-0.1,0.3, dz)
line_tube_X = np.arange(-tube_length, 0, 0.01)
Bz_ideal = -1 * strength * (z - 10)

# Iz_ROI = np.where((z > 0.04) & (z < 0.24))[0]
Iz_ROI = np.where((z > 0.0625) & (z < 0.2225))[0]

W_positions = int(tube_length / winding_spacing)

zspace_winding_bins = np.arange(-1 * tube_length, 0, winding_spacing)

winding_bins_0 = np.zeros(W_positions, dtype=int)


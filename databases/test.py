import numpy as np
from ase import Atoms
from ase.visualize import view

# Plane parameters
z_plane = 5.0
nx, ny = 20, 20
spacing = 0.5

# Generate grid points
positions = []

for i in range(nx):
    for j in range(ny):
        x = i * spacing
        y = j * spacing
        positions.append([x, y, z_plane])

# Use dummy atoms (H here)
plane = Atoms('H' * len(positions), positions=positions)

# Optional: make a real structure
# structure = ...
# structure += plane

view(plane)
from ase.io import read, write
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

# Input / output
traj_file = "test_relax.traj"
gif_file = "test_relax.gif"

# Read all frames
atoms_list = read(traj_file, index=":")

images = []

# Temporary folder for frames
tmp_dir = "tmp_frames"
os.makedirs(tmp_dir, exist_ok=True)

for i, atoms in enumerate(atoms_list):
    fig, ax = plt.subplots()

    # Plot atoms
    plot_atoms(atoms, ax, rotation=("0x,0y,0z"))

    ax.set_axis_off()

    filename = f"{tmp_dir}/frame_{i:04d}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    images.append(imageio.imread(filename))

# Save GIF
imageio.mimsave(gif_file, images, fps=5)

# Cleanup (optional)
for f in os.listdir(tmp_dir):
    os.remove(os.path.join(tmp_dir, f))
os.rmdir(tmp_dir)

print(f"Saved GIF to {gif_file}")
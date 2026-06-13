import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from gen_catalyst_design.utils import get_atom_color_dict, get_full_element_pool_no_saas
from ase.io import read
import os


element_pool = get_full_element_pool_no_saas()
ATOM_COLORS = get_atom_color_dict(element_pool=element_pool)
ATOM_COLORS["O"] = "red"


def main():
    framerate = 20
    frame_skips = 2
    noise_types = ["Uniform"] ##rad_scale = 2500
    rad_scale = 2500

    for noise_type in noise_types:
        traj_dir = os.path.join("..", noise_type)
        traj_files = [file for file in os.listdir(traj_dir) if "traj" in file]

        for traj_file in traj_files:
            traj_filename = traj_file.split(".")[0]
            render(
                input_traj=os.path.join(traj_dir, traj_file),
                output_mp4=os.path.join(noise_type+traj_filename+".mp4"),
                fps=framerate,
                dpi=200,
                stride=frame_skips,
                elev=25,
                azim=-15,
                rad_scale=rad_scale
            )
            #print(traj_files)
            #exit()



def set_equal_axes(ax, xyz: np.ndarray, pad: float = 1.5):
    """Make 3D axes have equal scale and fixed limits for the whole trajectory."""
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    center = (mins + maxs) / 2
    radius = max(maxs - mins) / 2 + pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    


def render(input_traj: str, output_mp4: str, fps: int, dpi: int, stride: int,
           elev: float, azim: float, rad_scale:float=3000):

    frames = read(str(input_traj), index=f"::{stride}")
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise ValueError(f"No frames found in {input_traj}")

    xyz = np.array([atoms.get_positions() for atoms in frames], dtype=float)

    colors_per_frame = []
    sizes_per_frame = []
    for atoms in frames:
        symbols = atoms.get_chemical_symbols()
        nums = atoms.get_atomic_numbers()
        colors_per_frame.append([
            ATOM_COLORS.get(symbol, "lightgray") for symbol in symbols
        ])

        sizes_per_frame.append([
            rad_scale * float(0.8) for num in nums
        ])

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    #fig = plt.figure(figsize=(7, 7))
    #ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    #set_equal_axes(ax, xyz)
    ax.set_axis_off()
    ax.margins(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #ax.set_xlabel("x / Å")
    #ax.set_ylabel("y / Å")
    #ax.set_zlabel("z / Å")
    #ax.set_box_aspect((1, 1, 1))

    scat = ax.scatter(
        xyz[0, :, 0],
        xyz[0, :, 1],
        xyz[0, :, 2],
        s=sizes_per_frame[0],
        c=colors_per_frame[0],
        edgecolors="k",
        linewidths=2.0,
        depthshade=False,
        
    )

    xmin, xmax = xyz[0, :, 0].min(), xyz[0, :, 0].max()
    ymin, ymax = xyz[0, :, 1].min(), xyz[0, :, 1].max()
    zmin, zmax = xyz[0, :, 2].min(), xyz[0, :, 2].max()

    span = max(xmax - xmin, ymax - ymin, zmax - zmin)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    zoom = 0.60 #0.50 

    ax.set_xlim(cx - zoom * span, cx + zoom * span)
    ax.set_ylim(cy - zoom * span, cy + zoom * span)
    ax.set_zlim(cz - zoom * span, cz + zoom * span)

    #title = ax.set_title(f"Denoising trajectory: frame 1/{len(frames)}")

    bond_collection = None

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, str(output_mp4), dpi=dpi):
        for k in range(len(frames)):
            scat._offsets3d = (
                xyz[k, :, 0],
                xyz[k, :, 1],
                xyz[k, :, 2],
            )

            scat.set_facecolor(colors_per_frame[k])
            scat.set_edgecolor("black")
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {output_mp4} using {len(frames)} rendered frames")



if __name__ == "__main__":
    main()
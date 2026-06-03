from ase.db import connect
from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.utilities import get_edges_list_from_connectivity
from ase.io import read, write
from ase.atoms import Atoms
import numpy as np
import os
from ase_ml_models.databases import get_atoms_list_from_db


def main():
    facets = ["100", "111"]
    rot_x = +45
    rot_y = +0
    species_for_plot = {
        "100":{
            "H*":3,
            "CO*":16,
            "O*":13,
            "CO2**":46
    },
        "111":{
            "H*":8,
            "CO*":29,
            "O*":17,
            "CO2**":59
        }
    }
    zorder_dict = {"100":{
        0:1,
        1:1,
        2:2,
        3:2,
        4:1,
        5:0,
        6:1,
        7:0,
        8:0,
        9:0,
        10:1,
        11:0,
        12:0,
        13:0,
        14:2,
        15:3,
        16:2,
        17:1,
        18:2,
        19:1,
        20:3},
        "111":{
            0:1,
            1:1,
            2:2,
            3:2,
            4:1,
            5:1,
            6:2,
            7:0,
            8:0,
            9:0,
            10:0,
            11:1,
            12:1,
            13:0,
            14:0,
            15:3,
            16:3,
            17:3,
            18:1,
            19:3,
            20:2,
            21:3
        }
    }

    #indices_dict = {
    #    "100":15,
    #    "111":19
    #}

    for facet in facets:
        db = connect(f"{facet}_templates.db")
        atoms_list = get_atoms_list_from_db(db)
        atoms_for_image = [atoms_list[species_for_plot[facet][species]] for species in species_for_plot[facet]]
        #print(atoms_list[1].info.keys())
        #for atoms in atoms_list:
        #    print(atoms.info["species"])
        for atoms in atoms_for_image:
            for i, view in enumerate([dict(), dict(rotation='-75x')]):
                species_name = atoms.info['species'].split("*")[0]
                write(f"{facet}_{species_name}_view_{i}.png", images=[atoms], **view)
                fig, ax = plot_connectivity(
                    atoms=atoms, 
                    plot_fig_name=f"{facet}_{species_name}_graph.png", 
                    alpha=1.0, 
                    show_plot=False,
                    zorders_dict=zorder_dict[facet]
                    )


def plot_connectivity(
    atoms: Atoms,
    plot_fig_name:str,
    connectivity: np.ndarray = None,
    edges_pbc: bool = False,
    show_plot: bool = True,
    show_axis: bool = False,
    colors: str = "jmol",
    alpha: float = None,
    scale_radii: float = 100,
    zorders_dict:dict = None
):
    """
    Plot the atoms and bonds of an ase.Atoms object.
    """
    import matplotlib.pyplot as plt
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors
    # Get edges list.
    if connectivity is None:
        connectivity = atoms.info["connectivity"]
    edges_list = get_edges_list_from_connectivity(connectivity=connectivity)
    # Delete edges from pbc.
    if edges_pbc is False:
        remove = []
        for ii, (a0, a1) in enumerate(edges_list):
            distance = atoms.get_distance(a0=a0, a1=a1, mic=False)
            distance_mic = atoms.get_distance(a0=a0, a1=a1, mic=True)
            if distance > distance_mic+1e-6:
                remove.append(ii)
        edges_list = [edge for ii, edge in enumerate(edges_list) if ii not in remove]
    # Get the radii, and colors.
    radii = covalent_radii[atoms.numbers]
    if colors == "jmol":
        colors = jmol_colors[atoms.numbers]
    # Get the 3D edges.
    edges_xyz = np.array([
        (atoms.positions[a0], atoms.positions[a1]) for a0, a1 in edges_list
    ])
    # Prepare a figure.
    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(projection="3d")
    # Plot the nodes.
    # Plot the edges.
    indices_ads = atoms.info["indices_ads"]
    for edge_idxs, edge in zip(edges_list, edges_xyz):
        if edge_idxs[-1] not in indices_ads:
            ax.plot(*edge.T, color="k", lw=0.5, zorder=-10)

    # 2. Front/highlight edges
    for edge_idxs, edge in zip(edges_list, edges_xyz):
        if edge_idxs[-1] in indices_ads:
            ax.plot(*edge.T, color="k", lw=1, zorder=1)
    
    # 3. Atoms last
    #print(atoms.positions)
    for i, color, position, rad in zip(range(len(colors)), colors, atoms.positions, radii):
        if zorders_dict is not None:
            if i in zorders_dict:
                zorder = zorders_dict[i]
            else:
                zorder = -2
        else:
            zorder = 5
        
        ax.plot(
            xs=[position[0]],
            ys=[position[1]],
            zs=[position[2]],
            marker="o",
            markersize=rad+4,
            c=color,
            markeredgecolor="k",
            alpha=alpha,
            zorder=-zorder,
            #depthshade=False,
        )
    # Adjust the figure.
    #scatter = ax.scatter(*atoms.positions.T, s=scale_radii*radii, c=colors, ec="k", alpha=alpha, zorder=10, depthshade=False)
    #print(scatter.get_zorder())
    #exit()
    ax.grid(False)
    if show_axis is True:
        for ax_i in (ax.xaxis, ax.yaxis, ax.zaxis):
            ax_i.set_ticks([])
            ax_i.set_alpha(0.)
            ax_i.pane.fill = False
            ax_i.pane.set_alpha(0.)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    rot_x = +15
    rot_y = +0.0
    ax.elev = rot_x
    ax.azim = -90-rot_y
    
    plt.savefig(plot_fig_name, transparent=True) #,  bbox_inches="tight",  pad_inches=0
    # Show the figure.

    if show_plot is True:
        plt.show()
    # Return the axis.
    return fig, ax


if __name__ == "__main__":
    main()
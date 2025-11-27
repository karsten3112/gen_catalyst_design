# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI
import os
import sys
from ase_ml_models.yaml import read_atoms_from_yaml
from ase_ml_models.databases import write_atoms_list_to_db
from catalyst_opt_tools.adsorption import (
    get_cluster_from_surface,
    get_surface_edges,
    get_adsorption_sites,
    get_bidentate_sites,
    get_sites_directions,
)

from gen_catalyst_toolkit.adsorbate_placers import setup_adsorbate_placer


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = False
    write_to_db = True
    miller_indices = ["100", "111", "211"] # 100 | 111 | 211

    species_list = ["CO*", "H*", "O*", "OH*", "H2O*", "CO2**"]

    adsorbate_placer = setup_adsorbate_placer(filename="molecules.yaml",
                                              pth_header="yaml_files/molecules",
                                              filter_species=species_list
                                              )
    #exit()
    for miller_index in miller_indices:
        # Get periodic surface.
        atoms_periodic, indices_site = get_periodic_surface(miller_index=miller_index)
    
        # Get cluster from periodic surface.
        atoms_surf = get_cluster_from_surface( #Specific for these types of surfaces
            atoms=atoms_periodic,
            method="ase",
            bond_cutoff=1,
            indices_ads=[],
            indices_site=indices_site,
            remove_pbc=True,
            skin=0.20
        )
        atoms_surf.info["species"] = "clean"
        atoms_surf.info["indices_ads"] = []
        atoms_surf.info["scores"] = {}
        atoms_surf.info["miller_index"] = miller_index

        indices_surf = atoms_surf.info["indices_site"]
        # Get edges from connectivity.
        # Place adsorbates
        atoms_surfads_list = adsorbate_placer(atoms_surface=atoms_surf, indices_site=indices_surf)
        #for atoms in atoms_surfads_list:
        #    print(atoms.info["species"])
        # Show atoms in GUI.
        #exit()
        if show_atoms is True:
            gui = GUI(atoms_surfads_list)
            gui.run()

        # Write atoms to database.
        if write_to_db is True:
            if os.path.exists(f"databases/templates/{miller_index}/"):
                pass
            else:
                os.mkdir(path=f"databases/templates/{miller_index}/")
            db_ase_name = f"databases/templates/{miller_index}/{miller_index}_templates.db"
            db_ase = connect(db_ase_name, append=False)
            write_atoms_list_to_db(atoms_list=atoms_surfads_list, db_ase=db_ase)


# -------------------------------------------------------------------------------------
# CONSTRUCT PERIODIC SURFACE
# -------------------------------------------------------------------------------------

def get_periodic_surface(miller_index) -> tuple:
    if miller_index == "100":
        from ase.build import fcc100
        atoms_periodic = fcc100(symbol="Au", size=(3, 3, 4), vacuum=10.0)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "111":
        from ase.build import fcc111
        atoms_periodic = fcc111(symbol="Au", size=(3, 3, 4), vacuum=10.0)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "211":
        from ase.build import fcc211
        atoms_periodic = fcc211(symbol="Au", size=(6, 3, 4), vacuum=10.0)
        indices_site = [0, 1, 7, 10, 15, 16]
    # Highlight site atoms.
    for ii in indices_site:
        atoms_periodic[ii].symbol = "Cu"
    return atoms_periodic, indices_site


# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
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
    adsorption_monodentate,
    adsorption_bidentate,
    get_adsorption_sites,
    get_bidentate_sites,
    get_surface_edges,
    get_cluster_from_surface
)
from ase.data import atomic_numbers, reference_states
from ase_ml_models.utilities import get_connectivity
from ase.io import write
from gen_catalyst_design.stability import get_connectivity_inverted_slab, inversion_symmetry_repeat, center_slab
from ase.atoms import Atoms

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    show_atoms = False
    write_to_db = True
    surface_type = "surface" # cluster | surface
    miller_indices = ["100", "111", "211"] # 100 | 111 | 211

    species_list = ["CO*", "H*", "O*", "OH*", "H2O*", "CO2**"]

    adsorbate_placer = setup_adsorbate_placer(
        filename="molecules.yaml",
        pth_header="../yaml_files/molecules",
        filter_species=species_list
    )
    #exit()
    for miller_index in miller_indices:
        # Get periodic surface.
        atoms_periodic, indices_site = get_periodic_surface(miller_index=miller_index)
    
        # Get cluster from periodic surface.
        if surface_type == "cluster":
            atoms_surf = get_cluster_from_surface( #Specific for these types of surfaces
                atoms=atoms_periodic,
                method="ase",
                bond_cutoff=1,
                indices_ads=[],
                indices_site=indices_site,
                remove_pbc=True,
                skin=0.20
            )
        if surface_type == "surface":
            atoms_surf = atoms_periodic
            n_atoms = len(atoms_surf)
            #print(miller_index, n_atoms)
            expanded_surface = apply_inversion_symmetry(
                atoms=atoms_surf.copy(),
                miller_index=miller_index,
                vacuum=100.0
            )
            connectivity = get_connectivity_inverted_slab(
                atoms=expanded_surface,
                method="ase",
                bond_cutoff=1,
                skin=0.20,
                remove_pbc=True
            )[:n_atoms,:n_atoms]
            atoms_surf.info["connectivity"] = connectivity
            atoms_surf.info["indices_site"] = indices_site
        atoms_surf.info["species"] = "clean"
        atoms_surf.info["indices_ads"] = []
        atoms_surf.info["scores"] = {}
        atoms_surf.info["miller_index"] = miller_index
        indices_surf = atoms_surf.info["indices_site"]

        atoms_surfads_list = adsorbate_placer(atoms_surface=atoms_surf, indices_site=indices_surf)
        
        if show_atoms is True:
            gui = GUI(atoms_surfads_list)
            gui.run()

        out_dir = f"{surface_type}_templates"
        # Write atoms to database.
        if write_to_db is True:
            if os.path.exists(out_dir):
                pass
            else:
                os.makedirs(out_dir)
            db_ase_name = f"{out_dir}/{miller_index}_templates.db"
            db_ase = connect(db_ase_name, append=False)
            write_atoms_list_to_db(atoms_list=atoms_surfads_list, db_ase=db_ase)


# -------------------------------------------------------------------------------------
# CONSTRUCT PERIODIC SURFACE
# -------------------------------------------------------------------------------------


def apply_inversion_symmetry(atoms:Atoms, miller_index:str, vacuum:float=10.0):
    if miller_index == "100":
        pass
    atoms = center_slab(atoms=atoms)
    z = atomic_numbers["Au"]
    a_lat = reference_states[z]["a"]
    interlayer_dist = a_lat / 2
    atoms.center(vacuum=interlayer_dist / 2, axis=2)
    atoms = inversion_symmetry_repeat(atoms=atoms)
    atoms.center(vacuum=vacuum, axis=2)
    return atoms


def get_periodic_surface(miller_index, vacuum:float=10.0) -> tuple:
    if miller_index == "100":
        from ase.build import fcc100
        atoms_periodic = fcc100(symbol="Au", size=(3, 3, 4), vacuum=vacuum)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "111":
        from ase.build import fcc111
        atoms_periodic = fcc111(symbol="Au", size=(3, 3, 4), vacuum=vacuum)
        indices_site = [27, 28, 30, 31]
    elif miller_index == "211":
        from ase.build import fcc211
        atoms_periodic = fcc211(symbol="Au", size=(6, 3, 4), vacuum=vacuum)
        indices_site = [0, 1, 7, 10, 15, 16]
    # Highlight site atoms.
    for ii in indices_site:
        atoms_periodic[ii].symbol = "Cu"
    return atoms_periodic, indices_site



class AdsorbatePlacer:
    def __init__(self, atoms_mol_dict):
        self.atoms_mol_dict = atoms_mol_dict

    def __call__(self, atoms_surface:Atoms, indices_site:list, add_clean:bool=True) -> list:
        if add_clean:
            atoms_clean = atoms_surface.copy()
            atoms_clean.info["bond_info"] = {}
            result_atoms_list = [atoms_clean]
        else:
            result_atoms_list = []
        
        sites_dict = self.get_sites_dict(atoms_surface=atoms_surface, indices_site=indices_site)
        #print(sites_dict)
       # exit()
        for species in self.atoms_mol_dict:
            atoms_mol = self.atoms_mol_dict[species]
            surf_bound = atoms_mol.info["surf_bound"]
            sites_names = atoms_mol.info["sites_names"]
            for site_name in sites_names:
                for site_indices in sites_dict[site_name]:
                    if len(surf_bound) == 1:
                    # Mono-dentate adsorption.
                        atoms_surfads = adsorption_monodentate(
                            atoms_mol=atoms_mol,
                            atoms_surf=atoms_surface,
                            surf_bound=surf_bound,
                            site_indices=site_indices,
                        )
                    elif len(surf_bound) == 2:
                        # Bi-dentate adsorption.
                        atoms_surfads = adsorption_bidentate(
                            atoms_mol=atoms_mol,
                            atoms_surf=atoms_surface,
                            surf_bound=surf_bound,
                            site_indices=site_indices,
                        )
                    result_atoms_list.append(atoms_surfads)
                    atoms_surfads.info.update({"species": atoms_mol.info["species"]})
                    
                    atoms_surfads.info["bond_info"] = {
                        "site_name": site_name,
                        "sites_conf": site_indices
                    }
        return result_atoms_list

    def get_sites_dict(self, atoms_surface:Atoms, indices_site:list) -> dict:
        edges_surf = get_surface_edges(
            connectivity=atoms_surface.info["connectivity"],
            indices_surf=indices_site,
        )
        # Get mono-dentate adsorption sites.
        sites_dict = get_adsorption_sites(
            indices_surf=indices_site,
            edges_surf=edges_surf,
        )
        # Get bi-dentate adsorption sites.
        sites_bi_dict = get_bidentate_sites(
            sites_dict=sites_dict,
        )
        sites_dict.update(sites_bi_dict)
        return sites_dict


    def construct_atoms_from_bond_info(self, atoms_surface:Atoms, add_clean:bool=True):
        if add_clean:
            atoms_list = [atoms_surface.copy()]
        else:
            atoms_list = []
        bond_info_dict = atoms_surface.info["bond_info"]
        for species in bond_info_dict:
            if species == "clean":
                pass
            else:
                atoms_surfads = atoms_surface.copy()
                bond_dict = bond_info_dict[species]
    
                atoms_mol = self.atoms_mol_dict[bond_dict["ads_mol"]]
                surf_bound = atoms_mol.info["surf_bound"]
                site_indices = bond_dict["sites_conf"]
                if len(surf_bound) == 1:
                    # Mono-dentate adsorption.
                        atoms_surfads = adsorption_monodentate(
                            atoms_mol=atoms_mol,
                            atoms_surf=atoms_surface,
                            surf_bound=surf_bound,
                            site_indices=site_indices,
                        )
                elif len(surf_bound) == 2:
                        # Bi-dentate adsorption.
                        atoms_surfads = adsorption_bidentate(
                            atoms_mol=atoms_mol,
                            atoms_surf=atoms_surface,
                            surf_bound=surf_bound,
                            site_indices=site_indices,
                        )
                atoms_list.append(atoms_surfads)
        return atoms_list

def setup_adsorbate_placer(filename:str, pth_header:str=None, filter_species:list=None) -> AdsorbatePlacer:
    atoms_mol_dict = {}
    if pth_header is not None:
        filename = os.path.join(pth_header, filename)
    else:
        pass
    atoms_mol_list = read_atoms_from_yaml(filename=filename)
    if filter_species is not None:
        for atoms in atoms_mol_list:
            species = atoms.info["species"]
            if species in filter_species:
                atoms_mol_dict[species] = atoms
    else:
        for atoms in atoms_mol_list:
            species = atoms.info["species"]
            atoms_mol_dict[species] = atoms
    #print(atoms_mol_dict)
    return AdsorbatePlacer(atoms_mol_dict=atoms_mol_dict)



# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
from ase.atoms import Atoms
import yaml
from ase_ml_models.yaml import read_atoms_from_yaml
import os
from catalyst_opt_tools.adsorption import (
    adsorption_monodentate,
    adsorption_bidentate,
    get_adsorption_sites,
    get_bidentate_sites,
    get_surface_edges
)



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
                    
                    atoms_surfads.info["bond_info"] = {"site_name": site_name,
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
    return AdsorbatePlacer(atoms_mol_dict=atoms_mol_dict)
    

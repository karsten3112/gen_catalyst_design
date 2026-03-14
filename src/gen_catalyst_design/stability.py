from ase.atoms import Atoms
import numpy as np
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase_ml_models.utilities import (
    get_connectivity,
    get_edges_list_from_connectivity,
    get_connectivity_from_edges_list
)
from chgnet.model.dynamics import CHGNetCalculator
from ase.data import atomic_numbers, reference_states
import yaml
import os
# -------------------------------------------------------------------------------------
# RELAX ATOMS AND CELL
# -------------------------------------------------------------------------------------

def relax_atoms_and_cell(
    atoms: Atoms,
    fmax: float = 0.05,
    trajectory: str = None,
    relax_z: bool = True,
):
    """
    Relax slab and (x, y) cell axes.
    """
    # Set mask for cell filter.
    mask = np.identity(n=3, dtype=bool)
    if relax_z is False:
        mask[2, 2] = False
    # Relax atoms and cell.
    opt = BFGS(
        atoms=FrechetCellFilter(atoms=atoms, mask=mask),
        trajectory=trajectory,
    )
    opt.run(fmax=fmax)
    # Return relaxed atoms.
    return atoms

# -------------------------------------------------------------------------------------
# CALCULATE FORMATION ENERGY
# -------------------------------------------------------------------------------------

def calculate_formation_energy(
    atoms: Atoms,
    energies_ref: dict,
) -> float:
    """
    Calculate formation energy.
    """
    from collections import Counter
    # Calculate formation energy.
    energy = atoms.get_potential_energy()
    composition = dict(Counter(atoms.get_chemical_symbols()))
    e_form = energy - sum(composition[ee] * energies_ref[ee] for ee in composition)
    # Return formation energy.
    return e_form

# -------------------------------------------------------------------------------------
# GET INVERSION CENTER
# -------------------------------------------------------------------------------------

def center_slab(
    atoms: Atoms,
    centers: list = None,
):
    if centers is None:
        centers = list(range(len(atoms)))
    center = atoms[centers].positions.mean(axis=0)
    transl = np.dot([0.5] * 3, atoms.cell) - center
    atoms.translate(transl)
    atoms.wrap()
    return atoms

# -------------------------------------------------------------------------------------
# INVERSION SYMMETRY REPEAT
# -------------------------------------------------------------------------------------

def inversion_symmetry_repeat(
    atoms: Atoms,
    center: str = "cell",
) -> Atoms:
    """
    Mirror slab along the specified axis.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    # Work in fractional coordinates.
    scaled = atoms.get_scaled_positions(wrap=False)
    s0 = np.array([0.5, 0.5, 0.0], dtype=float)
    scaled_inv = 2.0 * s0 - scaled
    pos_inv = scaled_inv @ cell
    symbols = atoms.get_chemical_symbols()
    atoms_inv = Atoms(
        symbols=symbols,
        positions=pos_inv,
        cell=cell,
        pbc=pbc,
    )
    # Combine.
    atoms += atoms_inv
    return atoms

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY OF INVERTED SLAB
# -------------------------------------------------------------------------------------

def get_connectivity_inverted_slab(
    atoms: Atoms,
    **kwargs,
) -> np.ndarray:
    """
    Get connectivity of the slab with inversion symmetry.
    """
    n_atoms = len(atoms) // 2
    connectivity = get_connectivity(atoms=atoms, **kwargs)
    edges_list = get_edges_list_from_connectivity(connectivity=connectivity)
    edges_list_new = []
    for edge in reversed(edges_list):
        aa, bb = sorted(edge)
        if bb < n_atoms and aa < n_atoms:
            edges_list_new.append([aa, bb])
        elif bb >= n_atoms and aa < n_atoms:
            edges_list_new.append([aa, bb - n_atoms])
    # Return connectivity.
    return get_connectivity_from_edges_list(atoms=atoms, edges_list=edges_list_new)


def apply_inversion_symmetry(atoms:Atoms, miller_index:str, vacuum:float=10.0):
    atoms = center_slab(atoms=atoms)
    z = atomic_numbers["Au"]
    a_lat = reference_states[z]["a"]
    interlayer_dist = a_lat / 2
    atoms.center(vacuum=interlayer_dist / 2, axis=2)
    atoms = inversion_symmetry_repeat(atoms=atoms)
    atoms.center(vacuum=vacuum, axis=2)
    return atoms


class Stabilizer:
    def __init__(
            self,
            template_atoms:Atoms,
            calculator:CHGNetCalculator,
            ref_energy_file:str,
            vacuum:float=10.0,
            fmax:float=0.05,
            ref_energy_pth_header:str=None
        ):
        self.template_atoms = template_atoms
        self.calculator = calculator
        self.ref_energy_dict = self.load_ref_energy_dict(
            filename=ref_energy_file, 
            pth_header=ref_energy_pth_header
        )
        self.vacuum = vacuum
        self.fmax = fmax


    def load_ref_energy_dict(self, filename:str, pth_header:str=None):
        if pth_header is not None:
            filename = os.path.join(pth_header, filename)
        with open(filename, mode="r") as fileobj:
            data = yaml.safe_load(fileobj)
        return data
        

    def get_formation_energy_from_symbols(self, symbols:list, trajectory=None):
        atoms = self.template_atoms.copy()
        atoms.set_chemical_symbols(symbols=symbols)
        atoms = apply_inversion_symmetry(
            atoms=atoms,
            miller_index=self.template_atoms.info["miller_index"],
            vacuum=self.vacuum
        )
        atoms.calc = self.calculator
        atoms = relax_atoms_and_cell(
            atoms=atoms,
            fmax=self.fmax,
            trajectory=trajectory,
            relax_z=False
        )
        #Divide by 2 here such that we dont double count?
        e_form = calculate_formation_energy(
            atoms=atoms,
            energies_ref=self.ref_energy_dict
        )/2.0
        return {"e_form":e_form, "atoms":atoms}
    

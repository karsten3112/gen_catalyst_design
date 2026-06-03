from ase.atoms import Atoms
import numpy as np
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase_ml_models.utilities import (
    get_connectivity,
    get_edges_list_from_connectivity,
    get_connectivity_from_edges_list
)
from .utils import get_periodic_surface
from chgnet.model.dynamics import CHGNetCalculator
from ase.data import atomic_numbers, reference_states
from ase.io import write
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
    maxsteps:int=1000,
    interval:int=10,
    recon_check_func:callable=None
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
    converged = False
    has_reconstructed = False
    while not converged and not has_reconstructed and opt.maxstep < maxsteps:
        converged = opt.run(
            fmax=fmax,
            steps=interval
        )
        if recon_check_func is not None:
            has_reconstructed = recon_check_func(
                atoms=atoms
            )
    # Return relaxed atoms.
    return atoms, has_reconstructed

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

def calculate_surface_formation_energy(
        atoms: Atoms,
        energies_ref:dict
    ) -> float:

    e_form = calculate_formation_energy(
        atoms=atoms,
        energies_ref=energies_ref
    )
    return e_form/2.0


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
    miller_index:str="100",
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
    if miller_index == "111":
        scaled_inv += np.array([-1/9, -1/9, 0.0])
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


def apply_inversion_symmetry(
        atoms:Atoms, 
        miller_index:str, 
        vacuum:float=10.0, 
        a_lat:float=None
    ):
    if miller_index == "100":
        atoms = center_slab(atoms=atoms)
    if a_lat is None:
        a_lat = reference_states[atomic_numbers["Au"]]["a"]
    if miller_index == "100":
        interlayer_dist = a_lat /(2.0*np.sqrt(1)) #divide by two to get half the inter-layer distance and sqrt(1) from miller index
    elif miller_index == "111":
        interlayer_dist = a_lat/(np.sqrt(3)) #divide by two to get half the inter-layer distance and sqrt(3) from miller index
    else:
        raise Exception(f"miller index of type: {miller_index} is not implemented")
    
    atoms.center(vacuum=interlayer_dist / 2, axis=2)
    atoms = inversion_symmetry_repeat(
        atoms=atoms,
        miller_index=miller_index
    )
    atoms.center(vacuum=vacuum, axis=2)
    return atoms


def recon_check_from_connectivity(
        atoms:Atoms,
    ):
    has_reconstructed = False
    connectivity_kwargs = atoms.info["connectivity_kwargs"]
    init_connectivity = atoms.info["init_connectivity"]
    current_connectivity = get_connectivity(atoms=atoms, **connectivity_kwargs)
    con_diff = init_connectivity - current_connectivity
    mask = np.bool(np.abs(con_diff))
    #print(np.argwhere(mask == True))
    if True in mask:
        has_reconstructed = True
    return has_reconstructed


def get_mean_lattice_const(
        symbols:list
    ):
    return np.mean([reference_states[atomic_numbers[symbol]]["a"] for symbol in symbols])

def recon_check_from_lattice(
        template_atoms:Atoms,
        atoms:Atoms,
        a_lat:float,
    ):
    has_reconstructed = False
    recon_radius = a_lat/2.0
    lattice_positions = template_atoms.positions
    current_positions = atoms.positions
    dists = np.linalg.norm(current_positions-lattice_positions, axis=1)
    recon_checks = dists > recon_radius
    if True in recon_checks:
        has_reconstructed = True
    return has_reconstructed


class Stabilizer:
    def __init__(
            self,
            template_atoms:Atoms,
            calculator:CHGNetCalculator,
            ref_energy_file:str,
            vacuum:float=10.0,
            fmax:float=0.05,
            maxsteps:int=1000,
            interval:int=10,
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
        self.maxsteps = maxsteps
        self.interval = interval


    def load_ref_energy_dict(self, filename:str, pth_header:str=None):
        if pth_header is not None:
            filename = os.path.join(pth_header, filename)
        with open(filename, mode="r") as fileobj:
            data = yaml.safe_load(fileobj)
        return data
        

    def get_formation_energy_from_symbols(
            self, 
            symbols:list, 
            trajectory=None, 
            apply_recon_check:bool=False,
            start_from_mean_lattice:bool=True, 
            recon_check_kwargs:dict={}
        ):
        if start_from_mean_lattice:
            a_lat = get_mean_lattice_const(symbols=symbols)
        else:
            a_lat = None
        
        atoms, _ = get_periodic_surface(
            miller_index=self.template_atoms.info["miller_index"],
            vacuum=self.vacuum,
            a_lat=a_lat
        )

        atoms.set_chemical_symbols(symbols=symbols)
        
        atoms = apply_inversion_symmetry(
            atoms=atoms,
            miller_index=self.template_atoms.info["miller_index"],
            vacuum=self.vacuum,
            a_lat=a_lat
        )

        atoms_init = atoms.copy()

        if apply_recon_check:
            connectivity_kwargs = recon_check_kwargs.pop("connectivity_kwargs", {})
            init_connectivity = get_connectivity(
                atoms=atoms,
                **connectivity_kwargs
            )
            atoms.info["init_connectivity"] = init_connectivity
            atoms.info["connectivity_kwargs"] = connectivity_kwargs
        
        atoms.calc = self.calculator
        atoms, has_reconstructed = relax_atoms_and_cell(
            atoms=atoms,
            fmax=self.fmax,
            trajectory=trajectory,
            relax_z=False,
            maxsteps=self.maxsteps,
            interval=self.interval,
            recon_check_func=recon_check_from_connectivity if apply_recon_check else None
        )
    
        e_form = calculate_surface_formation_energy(
            atoms=atoms,
            energies_ref=self.ref_energy_dict
        )
        
        return {"e_form":e_form, "atoms_final":atoms, "atoms_init":atoms_init, "recon":has_reconstructed}
        
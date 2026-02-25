from ase.atoms import Atoms
import numpy as np
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from collections import Counter
from catalyst_opt_tools.adsorption
from chgnet.model.dynamics import CHGNetCalculator
import os


class Stabilizer:
    def __init__(
            self,
            miller_index:str,
            calc,
            reference_energies_yaml:str,
            reference_energies_pth_header:str=None,
            fmax:float=0.05,

        ):
        self.template_surface = template_surface
        self.fmax = fmax
        pass

    def relax_atoms_and_cell(
            self,
            atoms: Atoms,
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
        opt.run(fmax=self.fmax)
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
        # Calculate formation energy.
        energy = atoms.get_potential_energy()
        composition = dict(Counter(atoms.get_chemical_symbols()))
        e_form = energy - sum(composition[ee] * energies_ref[ee] for ee in composition)
        # Return formation energy.
        return e_form

    def get_stability_from_atoms(self, atoms:Atoms):
        pass

    
    def get_stability_from_symbols(self, symbols:list):
        pass


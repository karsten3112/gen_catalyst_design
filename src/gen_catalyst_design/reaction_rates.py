from catalyst_opt_tools.utilities import update_atoms_list
from .calculators import Calculator
from .utils import get_atoms_from_template_db, get_features_bulk_and_gas
from ase.atoms import Atoms
from mikimoto.thermodynamics import ThermoNASA7
from mikimoto.microkinetics import Species
from mikimoto import units
import numpy as np
import yaml
import os

class ReactionMechanism:
    def __init__(
            self,
            mechanism_file:str="mechanism.yaml",
            template_atoms_list:list=None,
            features_bulk:dict=None,
            features_gas:dict=None, 
            mechanism_pth_header:str=None,
            temperature_celsius:float=500.0,
            pressure:float=1.0,
            calculator:Calculator=None
        ):
        self.template_atoms_list = template_atoms_list
        if template_atoms_list is not None:
            for atoms in template_atoms_list:
                if atoms.info["species"] == "clean":
                    self.n_atoms_surf = len(atoms)
                    self.clean_surface = atoms.copy()
                    self.clean_surface.info = atoms.info.copy()
        else:
            self.n_atoms_surf = None
            self.clean_surface = None
        self.features_bulk = features_bulk
        self.features_gas = features_gas
        self.calculator = calculator
        self.pressure = pressure
        self.temperature_celsius = temperature_celsius
        if mechanism_pth_header is not None:
            mechanism_file = os.path.join(mechanism_pth_header, mechanism_file)
        
        with open(mechanism_file, "r") as fileobj:
            mechanism_dict = yaml.safe_load(fileobj)
        self.nasa_coeffs_dict = self.get_NASA7coeffefients(mechanism_dict=mechanism_dict)

        self.gas_molfracs_inlet = {
            "CO2": 0.28,
            "H2": 0.28,
            "H2O": 0.02,
            "CO": 0.02,
            "N2": 0.40,
        }

    def set_template_atoms_list(
            self,
            db_filename:str,
            pth_header:str=None
        ):
        self.template_atoms_list, self.n_atoms_surf = get_atoms_from_template_db(
            db_filename=db_filename,
            pth_header=pth_header
        )
        
        for atoms in self.template_atoms_list:
            if atoms.info["species"] == "clean":
                self.clean_surface = atoms.copy()
                self.clean_surface.info = atoms.info.copy()
        

    def set_features(
            self,
            bulk_filename:str="features_bulk.yaml",
            gas_filename:str="features_gas.yaml",
            pth_header:str=None
        ):
        self.features_bulk, self.features_gas = get_features_bulk_and_gas(
            bulk_filename=bulk_filename,
            gas_filename=gas_filename,
            pth_header=pth_header
        )
        

    def set_calculator(self, calculator:Calculator):
        self.calculator = calculator
    
    def get_NASA7coeffefients(self, mechanism_dict:dict):
        spec_dict = {}
        temperature = units.Celsius_to_Kelvin(self.temperature_celsius)
        for species_type in ["gas", "adsorbates", "reactions"]:
            for species_data in mechanism_dict[f"species-{species_type}"]:
                spec = Species(
                    name=species_data["name"],
                    thermo=ThermoNASA7(
                        temperature=temperature,
                        coeffs_NASA=species_data["thermo"]["data"][0],
                    )
                )
                spec_dict[spec.name] = spec
        return spec_dict

    def get_rate_RDS_from_e_form_dict(self, e_form_dict:dict, bep_relation:callable):
        temperature = units.Celsius_to_Kelvin(self.temperature_celsius) # [K]
        pressure = self.pressure * units.atm # [Pa]

        delta_h = e_form_dict["CO(X)"] + e_form_dict["O(X)"] - e_form_dict["CO2(X)"]
        e_act_RDS = bep_relation(delta_h=delta_h)
        e_form_dict["CO2(X) + (X) <=> CO(X) + O(X)"] = e_act_RDS + e_form_dict["CO2(X)"]

        e_corr_dict = e_form_dict.copy()
        e_corr_dict.update({spec: 0.0 for spec in self.gas_molfracs_inlet})

        g0_form_dict = {}

        for species in e_corr_dict:
            if species in self.nasa_coeffs_dict:
                spec = self.nasa_coeffs_dict[species]
                g0_form_dict[spec.name] = spec.thermo.Gibbs_std + e_corr_dict[spec.name] * (units.eV/units.molecule) # [J/kmol]

        p_red = {
            species: self.gas_molfracs_inlet[species] * pressure / units.atm
                for species in self.gas_molfracs_inlet
        } # [-]
        # Calculate the Gibbs free energies of adsorption.
        g_ads_dict = {
            "CO(X)": g0_form_dict["CO(X)"] - g0_form_dict["CO"],
            "H(X)": g0_form_dict["H(X)"] - g0_form_dict["H2"] * 0.5,
            "O(X)": g0_form_dict["O(X)"] - (g0_form_dict["H2O"] - g0_form_dict["H2"]),
        } # [J/kmol]
        # Calculate equilibrium constants.
        k_eq_dict = {}
        for species in g_ads_dict:
            try:
                g_ads = np.exp(-g_ads_dict[species] / (units.Rgas * temperature))
                k_eq_dict[species] = g_ads
            except RuntimeWarning as w:
                print("WE HAVE RUNTIME WARNING IN EXPONENTIAL function, need to look at this, something is wrong")
                return w
    
        # Calculate coverage of free sites.
        coverage_free = 1 / (
            1 + 
            k_eq_dict["CO(X)"] * p_red["CO"] + 
            k_eq_dict["H(X)"] * p_red["H2"] ** 0.5 +
            k_eq_dict["O(X)"] * p_red["H2O"] / p_red["H2"]
        )
        # Calculate activation energy and kinetic constant of RDS.
        g0_act_RDS = (
            g0_form_dict["CO2(X) + (X) <=> CO(X) + O(X)"] - g0_form_dict["CO2"]
        ) # [eV]
        a_for = units.kB * temperature / units.hP # [1/s]
        k_for_RDS = a_for * np.exp(-g0_act_RDS / (units.Rgas * temperature)) # [1/s]
        # Calculate reaction rate.
        rate = k_for_RDS * p_red["CO2"] * coverage_free ** 2 # [1/s]
        # Return the reaction rate.
        return float(rate)
    

    def get_e_form_scoredict_from_symbols(
            self,
            symbols:list,
        ):
        if self.calculator is None:
            raise Exception("No calculator has been assigned, so energies cannot be calculated")
        if self.template_atoms_list is None:
            raise Exception("No template atoms have been provided yet, so no calculation can be performed")
        if len(symbols) != self.n_atoms_surf:
            raise Exception("Amount of elements provided do not match amount of elements in template surfaces")
        if self.features_bulk is None or self.features_gas is None:
            raise Exception("No features have been provided yet, so no calculation can be performed")
        
        update_atoms_list(
            atoms_list=self.template_atoms_list,
            features_bulk=self.features_bulk,
            features_gas=self.features_gas,
            symbols=symbols,
            n_atoms_surf=self.n_atoms_surf,
        )
        score_dict = self.calculator(atoms_list=self.template_atoms_list)
        return score_dict

    def get_e_form_scoredict_from_atoms(
            self,
            atoms:Atoms
        ):
        symbols = atoms.get_chemical_symbols()
        score_dict = self.get_e_form_scoredict_from_symbols(
            symbols=symbols
        )
        return score_dict

    def get_rate_of_RDS_from_symbols(
            self,
            symbols:list
        ):
        score_dict = self.get_e_form_scoredict_from_symbols(
            symbols=symbols
        )
        rate = self.get_rate_RDS_from_e_form_dict(
            e_form_dict=score_dict["e_form_info"],
            bep_relation=self.calculator.bep_relation
        )
        score_dict.update({"rate":rate})
        return score_dict

    def get_rate_RDS_from_atoms(
            self,
            atoms:Atoms 
        ):
        symbols = atoms.get_chemical_symbols()
        score_dict = self.get_rate_of_RDS_from_symbols(
            symbols=symbols
        )
        return score_dict

    def __call__(self, atoms_list):
        if self.calculator is None:
            raise Exception("No calculator has been assigned, so energies cannot be calculated")
        score_dict = self.calculator(atoms_list=atoms_list)
        e_form_dict = score_dict["e_form_info"]
        rate = self.get_rate_from_RDS(e_form_dict=e_form_dict, bep_relation=self.calculator.bep_relation)
        score_dict.update({"rate":rate})
        return score_dict

from gen_catalyst_design.utils import get_atoms_from_template_db, get_full_element_pool
from gen_catalyst_design.stability import Stabilizer
from chgnet.model.dynamics import CHGNetCalculator
from mace.calculators import mace_mp
from distutils.util import strtobool
from ase.io import read, write
import argparse
import random
import os

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--element_pool",
    "-elem_pool",
    type=str,
    required=False,
    default="full",
)

parser.add_argument(
    "--calculator",
    "-calc",
    type=str,
    required=False,
    default="chgnet",
)

parser.add_argument(
    "--filename",
    "-file",
    type=str,
    required=False,
    default="test.traj",
)

parser.add_argument(
    "--outdir",
    "-out",
    type=str,
    required=False,
    default="results",
)

parser.add_argument(
    "--mean_lat",
    "-m_lat",
    type=fbool,
    required=False,
    default=False,
)


parsed_args = parser.parse_args()


def main():
    random_seed = 42
    random.seed(random_seed)
    n_samples = 2
    miller_index = "100"
    calculator_type = parsed_args.calculator
    start_from_mean_lattice = parsed_args.mean_lat
    element_pool = get_full_element_pool()
    outdir = parsed_args.outdir
    filename = parsed_args.filename
    if outdir is not None:
        filename = os.path.join(outdir, filename)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db",
        pth_header="../../databases/surface_templates"
    )

    calculator = get_calculator(
        calculator_type=calculator_type
    )

    stabilizer = Stabilizer(
        template_atoms=template_atoms_list[0],
        calculator=calculator,
        ref_energy_file="chgnet_ref_energies.yaml",
        ref_energy_pth_header="../../yaml_files/reference_energies",
        interval=400
    )
    atoms_list = []
    for i in range(n_samples):
        symbols = random.choices(population=element_pool, k=n_atoms_surf)
        result_dict = stabilizer.get_formation_energy_from_symbols(
            symbols=symbols, 
            trajectory=None,
            apply_recon_check=False,
            start_from_mean_lattice=start_from_mean_lattice
        )
        atoms_init = result_dict["atoms_init"]
        atoms_init.info["sample_num"] = i
        atoms_init.info["relaxed"] = False
        atoms_final = result_dict["atoms_final"]
        atoms_final.info["sample_num"] = i
        atoms_final.info["relaxed"] = True
        atoms_list+=[atoms_init, atoms_final]
    
    write(filename=filename, images=atoms_list)


def get_element_pool_from_kw(
        keyword:str
    ):
    if keyword == "full":
        return get_full_element_pool()
    elif keyword == "all_fcc":
        return ['Ni', 'Cu', 'Rh', 'Ir', 'Pd', 'Pt', 'Au', 'Ag']
    elif keyword == "au_close_fcc":
        return ['Ag', 'Pt', 'Pd', 'Ir', 'Au']
    elif keyword == "ni_close_fcc":
        return ['Ni', 'Cu', 'Rh', 'Ir', 'Pd']
    else:
        raise Exception(f"provided keyword {keyword} does not have element pool")


def get_calculator(
        calculator_type:str
    ):
    if calculator_type == "chgnet":
        return CHGNetCalculator()
    elif calculator_type == "mace_mh1":
        return mace_mp(model="../../../../mace_models/mace-mh-1.model", head="omat_pbe")
    else:
        raise Exception(f"calculator type: {calculator_type} is not implemented")



if __name__ == "__main__":
    main()
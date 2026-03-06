from gen_catalyst_design.utils import get_atoms_from_template_db, get_calculator, get_features_bulk_and_gas
from gen_catalyst_design.stability import Stabilizer



def main():
    miller_index = "100"
    element_pool = ["Rh", "Cu", "Au"]


    template_atoms_list, n_atoms_surf = get_atoms_from_template_db(
         db_filename=f"{miller_index}_templates.db", 
         pth_header=f"../databases/bulk_templates/{miller_index}"
    )
    get_atoms_from_template_db()

    stabilizer = Stabilizer()


if __name__ == "__main__":
    main()
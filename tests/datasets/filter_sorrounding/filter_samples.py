from gen_catalyst_design.utils import get_atoms_from_template_db
from ase.io import read, write



def main():
    miller_index = "100"
    atoms_list = read("../../../results/reconstruction_check/chgnet_result_all_fcc.traj", index=":")

    cluster_templates, _ = get_atoms_from_template_db(
        db_filename=f"{miller_index}_templates.db",
        pth_header="../../../databases/cluster_templates"

    )
    clean_cluster_surface = cluster_templates[0]
    element_indices = clean_cluster_surface.info["indices_original"]
    result_cluster_atoms_list = []

    for atoms in atoms_list:
        result_atoms = clean_cluster_surface.copy()
        symbols = [atoms[index].symbol for index in element_indices]
        result_atoms.symbols = symbols
        result_atoms.info["rate"] = atoms.info["rate"]
        result_atoms.info["e_form"] = atoms.info["e_form"]
        result_atoms.info["is_recon"] = atoms.info["is_recon"]
        result_cluster_atoms_list.append(result_atoms)

    write("filtered_samples.traj", result_cluster_atoms_list)
    



if __name__ == "__main__":
    main()
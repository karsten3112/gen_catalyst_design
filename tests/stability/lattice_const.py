from gen_catalyst_design.utils import get_full_element_pool
from ase.data import reference_states, atomic_numbers
import numpy as np



def main():
    element_pool = get_full_element_pool()
    reference_info = get_reference_state_info(
        element_pool=element_pool
    )
    print("Information on all elements:")
    print(reference_info)
    print("All elements in element pool")
    print(list(reference_info.keys()))

    symmetry = "fcc"
    filtered_reference_info = filter_reference_states_symmetry(
        reference_info=reference_info,
        symmetry=symmetry
    )
    sorted_dict = sort_lattice_const_dict(
        lattice_const_dict={element:filtered_reference_info[element]["a"] for element in filtered_reference_info}
    )
    print(f"All elements with {symmetry} packing")
    print(list(sorted_dict.keys()))
    print(list(sorted_dict.values()))
    #print(sorted_dict)

    element = "Au"
    nearest_elements = get_nearest_lattice_const_to_elem(
        lattice_const_dict={element:reference_info[element]["a"] for element in reference_info},
        element=element
    )
    print(f"elements with lattice constants closes to: {element}")
    print(nearest_elements)
    nearest_elements = get_nearest_lattice_const_to_elem(
        lattice_const_dict={element:filtered_reference_info[element]["a"] for element in filtered_reference_info},
        element=element
    )
    print(f"elements with lattice constants closes to: {element} and {symmetry} packing")
    print(nearest_elements)


def get_reference_state_info(
        element_pool:list
    ):
    result_dict = {}
    for element in element_pool:
        z = atomic_numbers[element]
        reference_info = reference_states[z]
        result_dict[element] = {"symmetry":reference_info["symmetry"], "a":reference_info["a"]}
    return result_dict


def filter_reference_states_symmetry(
        reference_info:dict,
        symmetry:str="fcc"    
    ):
    result_dict = {}
    for element in reference_info:
        sym = reference_info[element]["symmetry"]
        if sym == symmetry:
            result_dict[element] = reference_info[element]
    return result_dict


def get_nearest_lattice_const_to_elem(
        lattice_const_dict:dict,
        element:str,
        k_nearest:int=4
    ):
    lattice_dict = lattice_const_dict.copy()
    if element in lattice_dict:
        a_lat_ref = lattice_dict.pop(element)
    else:
        a_lat_ref = reference_states[atomic_numbers[element]]["a"]
    

    lattice_constants = np.array([lattice_dict[element] for element in lattice_dict])
    lattice_species = list(lattice_dict.keys())
    diff = np.abs(lattice_constants-a_lat_ref)
    indices = np.argpartition(diff, k_nearest)[:k_nearest]
    nearest_elements = [lattice_species[index] for index in indices]
    return nearest_elements

def sort_lattice_const_dict(
        lattice_const_dict:dict
    ):
    return dict(sorted(lattice_const_dict.items(), key=lambda item: item[1]))

if __name__ == "__main__":
    main()
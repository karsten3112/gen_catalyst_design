from ase.atoms import Atoms
from ase.io import write
from gen_catalyst_design.db import Database, load_datadicts_from_db, load_atoms_list_from_db
from ase_ml_models.databases import write_atoms_to_db
from ase.io import write
import numpy as np

def main():
    #ase_db = connect("test_pred.db")
    #template_atoms = get_atoms_list_from_db(ase_db)[0]
    #print(template_atoms.info.keys())
    #ase_db = connect("test_pred.db")
    database = Database.establish_connection(
        filename="test_pred.db"
    )
    atoms_list = load_atoms_list_from_db(
        database=database
    )
    write("test.traj", images=atoms_list)
    #datadicts = load_data_from_db(database=database)
    #print(datadicts)
    #write_atoms_to_db(atoms=template_atoms, db_ase=ase_db)
    #connection = sqlite3.connect(database="test_pred.db")
    #cursor = connection.cursor()
    #command = f"""SELECT * FROM {'Elements'} AS {'elems'} \n"""
    #cursor.execute(command)
    #for data_row in cursor.fetchall():
    #    print(data_row)
    #Database.establish_connection(
        #filename="test_pred.db",
        #miller_index="100",
        #add_e_form=True,
        #surface_type="surface"
    #)
    #datadicts = load_data_from_db(
    #    database=db
    #)
    #print(len(datadicts[0]["elements"]))


if __name__ == "__main__":
    main()
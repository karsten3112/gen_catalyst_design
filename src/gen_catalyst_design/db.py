from sqlite3 import Connection, Cursor
from ase.atoms import Atoms
import tempfile
import sqlite3
import os


# -------------------------------------------------------------------------------------
# Database Description
# -------------------------------------------------------------------------------------

species_list = [
    "(X)",
    "CO2(X,X)",
    "CO2(X)",
    "CO(X)",
    "O(X)",
    "H(X)",
    "OH(X)",
    "H2O(X)",
    "CO2(X,X) <=> CO(X) + O(X)",
    "CO2(X) + (X) <=> CO(X) + O(X)"
]

# -------------------------------------------------------------------------------------
# Base Table Class
# -------------------------------------------------------------------------------------

class Table:
    def __init__(self, table_name, entries_dict, abbrev):
        self.table_name = table_name
        self.entries_dict = entries_dict
        self.abbrev = abbrev
        self.n_coloums = len(entries_dict)

    def convert_list_to_str(self, list:list):
        return ",".join(list)

    def get_insertion_command(self, entries_list:list):
        return f"""INSERT INTO {self.table_name} ({self.convert_list_to_str(entries_list)}) VALUES"""

    def insert_data_to_table(self, cursor:Cursor, data_dict:dict) -> Cursor:
        raise Exception("Must be implemented by sub-class")
    
    def convert_data(self, *data_elems):
        raise Exception("Must be implemented by sub-class")

    def select_data_from_table(self, cursor:Cursor, selection:str):
        pass

    @staticmethod
    def get_datatype_str(dataype:object):
        supported_types = {int: "INTEGER", str: "VARCHAR(20)", float:"REAL"}
        if dataype in supported_types:
            return supported_types[dataype]
        else:
            raise Exception(f"Specific data type of: {dataype} has not been implemented yet")

    @property
    def make_entries_strings(self):
        entries_list = []
        foreign_key_list = []
        for entry in self.entries_dict:
            datatype, key_constraint = self.entries_dict[entry]
            entries_str = f"{entry} {Table.get_datatype_str(dataype=datatype)}"
            if key_constraint is None:
                #entries_str += "\n"
                pass
            elif isinstance(key_constraint, PrimaryKey) or isinstance(key_constraint, BothPrimaryForeignKey):
                entries_str += " "+str(key_constraint) #+"\n"
                if isinstance(key_constraint, BothPrimaryForeignKey):
                    foreign_key_list.append(key_constraint.get_foreign_key_str) #+ ", \n "
            else:
                raise Exception(f"No key is defined for {key_constraint}, if no key should be used set this as None")
            entries_list.append(entries_str)
        return ",\n".join(entries_list), ",\n".join(foreign_key_list)
    
    def create_table(self, cursor:Cursor) -> Cursor:
        entries_str, foreign_key_str = self.make_entries_strings
        command = f"""CREATE TABLE IF NOT EXISTS {self.table_name}"""
        if len(foreign_key_str) == 0:
            command +=f""" ({entries_str})"""
        else:
            command +=f""" ({entries_str}, \n {foreign_key_str})"""
        cursor = cursor.execute(command)
        return cursor

    def drop_table(self, cursor:Cursor) -> Cursor:
        command = f"""DROP TABLE IF EXISTS {self.table_name}"""
        cursor = cursor.execute(command)
        return cursor


# -------------------------------------------------------------------------------------
# Element Table Class
# -------------------------------------------------------------------------------------

class ElementTable(Table):
    def __init__(self, miller_index:str):
        table_name = "Elements"
        abbreviation = "elems"
        self.num_surface_atoms_dict = {"100":21, "111":22, "211":29}
        entries_dict = {"struct_ID":(int, PrimaryKey(add_auto_increment=True)),
                    "batch":(int, None),
                    "rate":(float, None)
                    }
        if miller_index in self.num_surface_atoms_dict:
            idx_dict = {f"idx{i}":(str, None) for i in range(self.num_surface_atoms_dict[miller_index])}
        else:
            raise Exception(f"No is known with miller_index: {miller_index}")
        entries_dict.update(idx_dict)
        super().__init__(table_name, entries_dict, abbreviation)
    
    def get_insertion_command(self, elements:list, score_dict:dict, batch:int):
        entries_list = list(self.entries_dict.keys())
        entries_list.remove("struct_ID")
        init_string = super().get_insertion_command(entries_list)
        rate = score_dict["rate"]
        datalist = [str(batch), str(rate)] + ['"'+'","'.join(elements)+'"']
        return init_string+f""" ({self.convert_list_to_str(datalist)})"""

    def insert_data_to_table(self, cursor:Cursor, data_dict:dict) -> Cursor:
        insert_command = self.get_insertion_command(elements=data_dict["elements"], score_dict=data_dict["score_dict"], batch=data_dict["batch"])
        return cursor.execute(insert_command)
    
    def convert_data(self, *data_elems):
        elements_list = []
        result_dict = {}
        for entry, data_elem in zip(self.entries_dict, data_elems):
            if "idx" in entry:
                elements_list.append(data_elem)
            else:
                dtype, _ = self.entries_dict[entry]
                result_dict[entry] = dtype(data_elem)
        result_dict.update({"elements":elements_list})
        return result_dict

# -------------------------------------------------------------------------------------
# BondInfo table
# -------------------------------------------------------------------------------------

class BondInfoTable(Table):
    def __init__(self, element_table:ElementTable, species_list:list=species_list):
        table_name = "BondInfo"
        abbreviation = "bond_info"
        #BothPrimaryForeignKey(foreign_key=ForeignKey(entry="struct_ID", ref_table=element_table.table_name, ref_entry="struct_ID"))
        entries_dict = {"struct_ID":(int, BothPrimaryForeignKey(foreign_key=ForeignKey(entry="struct_ID", ref_table=element_table.table_name, ref_entry="struct_ID")))}
        species_entry_dict = {species:(str, None) for species in species_list}
        entries_dict.update(species_entry_dict)
        super().__init__(table_name, entries_dict, abbreviation)


# -------------------------------------------------------------------------------------
# Formation energies table
# -------------------------------------------------------------------------------------

class EformInfoTable(Table):
    def __init__(self, element_table:ElementTable, species_list:list=species_list):
        table_name = "EformInfo"
        abbreviation = "e_form"
        entries_dict = {"struct_ID":(int, BothPrimaryForeignKey(foreign_key=ForeignKey(entry="struct_ID", ref_table=element_table.table_name, ref_entry="struct_ID")))}
        species_entry_dict = {species:(float, None) for species in species_list}
        entries_dict.update(species_entry_dict)
        super().__init__(table_name, entries_dict, abbreviation)

# -------------------------------------------------------------------------------------
# KEY CLASSES
# -------------------------------------------------------------------------------------

class Key:
    def __init__(self):
        pass
    
    def __str__(self):
        raise Exception("Must be implemented by sub-classes")


class ForeignKey(Key):
    def __init__(self, entry:str, ref_table:str, ref_entry:str):
        super().__init__()
        self.entry = entry
        self.ref_table = ref_table
        self.ref_entry = ref_entry

    def __str__(self):
        return f"FOREIGN KEY ({self.entry}) REFERENCES {self.ref_table}({self.ref_entry})"
    
    @property
    def get_foreign_key_str(self):
        return str(self)

class PrimaryKey(Key):
    def __init__(self, add_auto_increment:bool=False):
        super().__init__()
        self.add_auto_increment = add_auto_increment

    def __str__(self):
        if self.add_auto_increment:
            return "PRIMARY KEY AUTOINCREMENT"
        else:
            return "PRIMARY KEY"


class BothPrimaryForeignKey(PrimaryKey):
    def __init__(self, foreign_key:ForeignKey, add_auto_increment:bool=False):
        super().__init__(add_auto_increment)
        self.foreign_key = foreign_key

    def __str__(self):
        return super().__str__()
    
    @property
    def get_foreign_key_str(self):
        return str(self.foreign_key)


# -------------------------------------------------------------------------------------
# SELECTION CLASS
# -------------------------------------------------------------------------------------

class Selection:
    def __init__(self):
        pass


# -------------------------------------------------------------------------------------
# Database Class
# -------------------------------------------------------------------------------------

class Database:
    def __init__(
            self, 
            filename:str, 
            miller_index:str, 
            use_tempdir:bool=False, 
            pth_header:str=None, 
            include_bond_info:bool=False, 
            include_eform_info:bool=False
        ) -> None:
        self.use_tempdir = use_tempdir
        self.filename = filename
        self.pth_header = pth_header

        if use_tempdir is True:
            tempdir = tempfile.gettempdir()
            self.connection = sqlite3.connect(database=os.path.join(tempdir, filename))
        else:
            self.connection = sqlite3.connect(database=self.join_pth_header_filename)
        
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE;")
        
        self.cursor = self.connection.cursor()
        self.element_table = ElementTable(miller_index=miller_index)
        self.table_list = [self.element_table]
        if include_bond_info:
            self.table_list.append(BondInfoTable(element_table=self.element_table))
        if include_eform_info:
            self.table_list.append(EformInfoTable(element_table=self.element_table))

    @property
    def join_pth_header_filename(self):
        if self.pth_header is not None:
            return os.path.join(self.pth_header, self.filename)
        else:
            return self.filename

    @property
    def delete_db_file(self):
        filename = self.join_pth_header_filename
        if os.path.exists(filename):
            os.remove(filename)

    def get_file_info_from_connection(self, connection:Connection):
        _, db_filename, db_pth_header = connection.execute("PRAGMA database_list").fetchone()
        return db_filename, db_pth_header

    def initialize_db_file(self) -> Cursor:
        for table in self.table_list:
            self.cursor = table.create_table(cursor=self.cursor)
    
    def drop_tables_db_file(self) -> Cursor:
        for table in self.table_list:
            self.cursor = table.drop_table(cursor=self.cursor)
    
    def insert_data_to_db_file(self, data_dict:dict) -> Cursor:
        for table in self.table_list:
            self.cursor = table.insert_data_to_table(cursor=self.cursor, data_dict=data_dict)

    def get_joining_command(self, selection:str):
        join_attr = "struct_ID"
        element_table_abbrev = self.element_table.abbrev
        command = f"""SELECT * FROM {self.element_table.table_name} AS {element_table_abbrev} \n"""
        for table in self.table_list:
            abbrev = table.abbrev
            if table.table_name == self.element_table.table_name:
                pass
            else:
                command += f"""JOIN {table.table_name} AS {abbrev} ON {element_table_abbrev}.{join_attr}={abbrev}.{join_attr} \n"""
        if selection is not None:
            pass #Code something here
        return command

    def join_tables(self, selection:str=None):
        join_command = self.get_joining_command(selection=selection)
        self.cursor.execute(join_command)
    
    def convert_selection_to_data(self, selection:str=None):
        coloumn_splits = {}
        if selection is None:
            n_prev = 0
            for table in self.table_list:
                coloumn_splits[table.table_name] = (n_prev, table.n_coloums) #Have to look into this, because it is not right as of now
                n_prev+=table.n_coloums
        else:
            pass #Code something here
        result_list = []
        for data_row in self.cursor.fetchall():
            result_dict = {}
            for table in self.table_list:
                i,j = coloumn_splits[table.table_name]
                result_dict.update(table.convert_data(*data_row[i:j]))
            result_list.append(result_dict)
        return result_list

    def select_data_from_db(self, selection:str=None):
        self.join_tables(selection=selection)
        result_list = self.convert_selection_to_data(selection=selection)
        return result_list
    
    def write_data_to_tables(self, data_dicts:list, append:bool=True):
        if append is False:
            self.drop_tables_db_file()
        self.initialize_db_file()
        try:
            self.cursor.execute("BEGIN")
            for data_dict in data_dicts:
                self.insert_data_to_db_file(data_dict=data_dict)
            self.cursor.execute("COMMIT")
        except Exception:
            self.cursor.execute("ROLLBACK")
            raise
        if self.use_tempdir is True:
            self.copy_sqlite_db(filename=self.filename, pth_header=self.pth_header)
    
    def copy_sqlite_db(self, filename:str, pth_header:str=None):
        if pth_header is not None:
            filename = os.path.join(pth_header, filename)
        if os.path.exists(filename):
            os.remove(filename)
        self.cursor.execute(f"VACUUM INTO '{filename}';")

    def close_connection(self):
        self.cursor.close()
        self.connection.close()
        
    @staticmethod
    def establish_connection(
        filename:str, 
        miller_index:str, 
        pth_header:str=None, 
        use_tempdir:bool=False, 
        database_kwargs:dict={}
        ):
        database = Database(
            filename=filename, 
            miller_index=miller_index, 
            use_tempdir=use_tempdir, 
            pth_header=pth_header, 
            **database_kwargs
        )
        return database
    
    
def load_data_from_db(database:Database, selection:str=None):
    result_list = database.select_data_from_db(selection=selection)
    database.close_connection()
    return result_list

def get_atoms_list_db(database:Database, template_surface:Atoms, selection:str=None):
    result_list = load_data_from_db(database=database, selection=selection)
    atoms_list = []
    for result_dict in result_list:
        atoms = template_surface.copy()
        elements = result_dict["elements"]
        rate = result_dict["rate"]
        atoms.symbols = elements
        atoms.info["rate"] = rate
        atoms_list.append(atoms)
    return atoms_list


import sys
sys.path.insert(0, "../../")
from gen_catalyst_toolkit.db import Database, load_data_from_db
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    miller_index = "100"
    db_pth_header = f"results/Rh_Cu_Au/miller_index_{miller_index}"
    database = Database.establish_connection(miller_index=miller_index, filename="runID_0_results.db", pth_header=db_pth_header)    #databases = [connect(filename=file, pth_header=db_pth_header) for file in os.listdir(db_pth_header)]
    #print(os.path.join(db_pth_header, "runID_0_results.db"))
    data_dicts = load_data_from_db(database=database)
    print(len(data_dicts))
    #for data_dict in data_dicts:
    #    print(data_dict["elements"])
    #or data_dict in data_dicts:
    #    print(data_dict)
        #print(data_dict["elements"], f"rate:{data_dict['scores']['rate']}")



if __name__ == "__main__":
    main()
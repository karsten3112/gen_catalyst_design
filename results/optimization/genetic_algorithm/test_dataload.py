from gen_catalyst_design.db import Database, load_datadicts_from_db


def main():
    db = Database.establish_connection(
        "test_opt.db"
    )
    datadicts = load_datadicts_from_db(db)

    unique_structs = []
    for datadict in datadicts:
        elements = "".join(datadict["elements"])
        if elements not in unique_structs:
            unique_structs.append(elements)
    print(len(unique_structs)/len(datadicts))


if __name__ == "__main__":
    main()
import yaml


def main():
    search_alg_dict = {
        "random_search":"Random Search",
        "genetic_algorithm": "Genetic Algorithm",
        "annealing": "Annealing"
    }



    stat_measures_whole = [
        "mean",
        "median",
        "IQR",
        "max_val",
        "unique_freq"
    ]

    stat_measures_top_k = [
        "mean",
        "median",
        "IQR"
    ]

    with open("result_stats.yaml", "r") as fileobj:
        stat_data = yaml.safe_load(fileobj)

    for search_alg in search_alg_dict:
        stat_str = search_alg_dict[search_alg] + " & "
        statistics = stat_data[search_alg]
        for stat_measure in stat_measures_whole:
            stat_str += get_stat_string(
                value_dict=statistics["whole_distribution"][stat_measure] if stat_measure in statistics["whole_distribution"] else statistics[stat_measure],
                stat_measure=stat_measure
            )
            stat_str += " & "
        for stat_measure in stat_measures_top_k:
            stat_str += " & "
            stat_str += get_stat_string(
                value_dict=statistics["top_k_summary"][stat_measure]
            )
        stat_str += r"\\"
        print(stat_str)

    #print(stat_data) 


def get_stat_string(
        value_dict:dict,
        stat_measure:str=None
    ):

    mean = value_dict["mean"]
    if stat_measure == "unique_freq":
        mean = 1.0 - mean
    
    err = value_dict["err"]
    decimal_counter = 0
    while round(err, decimal_counter) < 1e-12:
        decimal_counter+=1
        if decimal_counter > 5:
            decimal_counter=2
            break
    
    return rf"${round(mean, decimal_counter)} \pm {round(err, decimal_counter)}$"




if __name__ == "__main__":
    main()
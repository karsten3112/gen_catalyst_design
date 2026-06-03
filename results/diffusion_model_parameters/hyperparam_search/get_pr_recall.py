from gen_catalyst_design.db import Database, load_datadicts_from_db
from ase_ml_models.yaml import write_to_yaml
from ase.io import read
from scipy.stats import iqr
import numpy as np
import yaml
import os
from typing import Any, Dict, Iterable, List, Sequence


def main():
    use_log = True
    dataset = "genetic_alg_dataset"
    tau_sweep = [1,3,5,7,10]
    models = [filename for filename in os.listdir(dataset) if "model" in filename]

    real_dataset = os.path.join("..", "datasets", f"genetic_algorithm_2000.traj")
    real_datadicts = [{"elements":atoms.get_chemical_symbols(), "rate":atoms.info["rate"]} for atoms in read(real_dataset, ":")]
    real_rates = np.array([datadict["rate"] for datadict in real_datadicts])
    if use_log:
        real_rates = np.log10(real_rates)
    
        rate_threshold = np.percentile(real_rates, 80.0) #4.0

    for model in models:
        print(model)
        samples_dir = os.path.join(dataset, model, "samples")
        sampling_params_path = os.path.join(samples_dir, "sampling_params.yaml")

        if not os.path.exists(sampling_params_path):
            print(f"Skipping {model}: no sampling_params.yaml found.")
            continue

        with open(sampling_params_path, "r") as fileobj:
            sampling_params = yaml.safe_load(fileobj)
            
        g_scales = sampling_params["guidance_scales"]

        generated_samples_dict = {}
        pr_recall_results = {}
        best_fracs = {}
        for g_scale in g_scales:
            do_analysis = True
            pth_header = os.path.join(samples_dir, f"g_scale_{g_scale}_result")
            condition_dbs = [f"condition_{i}.db" for i in [1,2,3,4]]
            conditions_stored = os.listdir(pth_header)
            for condition_db in condition_dbs:
                if condition_db not in conditions_stored:
                    do_analysis = False
            
            if do_analysis == True:
                generated_samples = []
                for condition_db in condition_dbs:
                    generated_samples += load_datadicts_from_database(
                        db_filename=condition_db,
                        pth_header=pth_header
                    )
            
                generated_samples_dict[f"g_scale_{g_scale}"] = generated_samples


                generated_positive = filter_positive_samples(
                    datadicts=generated_samples,
                    threshold=rate_threshold,
                    use_log=use_log,
                )

                real_positive = filter_positive_samples(
                    datadicts=real_datadicts,
                    threshold=rate_threshold,
                    use_log=use_log,
                )

                best_frac = get_samples_higher_than_best(
                    real_positive_datadicts=real_positive,
                    generated_datadicts=generated_samples,
                    use_log=use_log
                )

                pr_recall_dicts = []
                for tau in tau_sweep:
                    precision = compute_precision(
                        generated_datadicts=generated_samples,
                        threshold=rate_threshold,
                        use_log=use_log,
                    )
                    #why is recall here only over generated-positive, and not over all. They should be over all?
                    recall = compute_recall(
                        real_positive_datadicts=real_positive,
                        generated_positive_datadicts=generated_positive,
                        tau=tau,
                    )
                    novelty = compute_novelty(
                        real_positive_datadicts=real_positive,
                        generated_positive_datadicts=generated_positive,
                        tau=tau,
                    )
                    result_dict = {"tau":tau, "precision":precision, "recall":recall, "novelty":novelty}
                    pr_recall_dicts.append(result_dict)

                pr_recall_results[f"g_scale_{g_scale}"] = pr_recall_dicts
                best_fracs[f"g_scale_{g_scale}"] = best_frac
            else:
                pr_recall_results[f"g_scale_{g_scale}"] = None
                best_fracs[f"g_scale_{g_scale}"] = None
        
        write_to_yaml(
            filename=os.path.join(dataset, model, "pr_recall.yaml"),
            data=pr_recall_results
        )

        write_to_yaml(
            filename=os.path.join(dataset, model, "best_frac.yaml"),
            data=best_fracs
        )


def get_samples_higher_than_best(
        real_positive_datadicts:list,
        generated_datadicts:list,
        use_log:bool=True
    ):
    real_pos_rates = np.array([datadict["rate"] for datadict in real_positive_datadicts])
    if use_log:
        real_pos_rates = np.log10(real_pos_rates)

    max_real_rate = np.max(real_pos_rates)
    gen_rates = np.array([datadict["rate"] for datadict in generated_datadicts])
    if use_log:
        gen_rates = np.log10(gen_rates)

    mask = np.sum((gen_rates > max_real_rate).astype(int))
    frac_high_samples = mask/len(gen_rates)
    return frac_high_samples





def hamming_distance(x: Sequence[Any], y: Sequence[Any]) -> int:
    """Compute Hamming distance between two equal-length sequences."""
    if len(x) != len(y):
        raise ValueError(
            f"Hamming distance requires equal-length sequences, got {len(x)} and {len(y)}."
        )
    return sum(a != b for a, b in zip(x, y))


def get_assignment_vector(datadict: Dict[str, Any], label_key: str="elements") -> List[Any]:
    """
    Extract the per-node atom assignment vector from a datadict.

    Update `label_key` below to match your actual field name, e.g.
    - "atom_types"
    - "node_labels"
    - "atomic_numbers"
    - "symbols"

    The value should be an ordered per-node sequence.
    """
    if label_key not in datadict:
        raise KeyError(
            f"Could not find label_key='{label_key}' in datadict. "
            f"Available keys: {list(datadict.keys())}"
        )

    labels = datadict[label_key]
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    else:
        labels = list(labels)

    return labels


def extract_rates(datadicts: Iterable[Dict[str, Any]], use_log: bool) -> np.ndarray:
    """Extract rates and optionally log10-transform them."""
    rates = np.array([datadict["rate"] for datadict in datadicts], dtype=float)
    if use_log:
        if np.any(rates <= 0):
            raise ValueError("Cannot apply log10 to non-positive rates.")
        rates = np.log10(rates)
    return rates


def filter_positive_samples(
    datadicts: Iterable[Dict[str, Any]],
    threshold: float,
    use_log: bool,
) -> List[Dict[str, Any]]:
    """Keep only samples with rate > threshold, using the same log convention as evaluation."""
    positives: List[Dict[str, Any]] = []
    for datadict in datadicts:
        rate = float(datadict["rate"])
        if use_log:
            if rate <= 0:
                continue
            rate = float(np.log10(rate))
        if rate > threshold:
            positives.append(datadict)
    return positives


def compute_precision(
    generated_datadicts: List[Dict[str, Any]],
    threshold: float,
    use_log: bool,
) -> float:
    """Precision = fraction of generated samples with rate > threshold."""
    if len(generated_datadicts) == 0:
        return 0.0

    generated_positive = filter_positive_samples(
        datadicts=generated_datadicts,
        threshold=threshold,
        use_log=use_log,
    )
    return len(generated_positive) / len(generated_datadicts)


def compute_recall(
    real_positive_datadicts: List[Dict[str, Any]],
    generated_positive_datadicts: List[Dict[str, Any]],
    tau: int,
    label_key: str="elements",
) -> float:
    """
    Recall = fraction of real high-rate samples covered by at least one
    generated high-rate sample within Hamming distance <= tau.
    """
    if len(real_positive_datadicts) == 0:
        return 0.0
    if len(generated_positive_datadicts) == 0:
        return 0.0

    generated_vectors = [
        get_assignment_vector(datadict=g, label_key=label_key)
        for g in generated_positive_datadicts
    ]

    covered = 0
    for real_datadict in real_positive_datadicts:
        real_vec = get_assignment_vector(datadict=real_datadict, label_key=label_key)
        if any(hamming_distance(real_vec, gen_vec) <= tau for gen_vec in generated_vectors):
            covered += 1

    return covered / len(real_positive_datadicts)


def compute_novelty(
    real_positive_datadicts: List[Dict[str, Any]],
    generated_positive_datadicts: List[Dict[str, Any]],
    tau: int,
    label_key: str="elements",
) -> float:
    """
    Novelty = fraction of generated high-rate samples that are NOT close to
    any real high-rate sample, i.e. Hamming distance > tau for all real positives.
    """
    if len(generated_positive_datadicts) == 0:
        return 0.0
    if len(real_positive_datadicts) == 0:
        # If there are no real positives, novelty is not really meaningful.
        # Return 0.0 to avoid misleading results.
        return 0.0

    real_vectors = [
        get_assignment_vector(datadict=r, label_key=label_key)
        for r in real_positive_datadicts
    ]

    novel_count = 0
    for gen_datadict in generated_positive_datadicts:
        gen_vec = get_assignment_vector(datadict=gen_datadict, label_key=label_key)
        if all(hamming_distance(gen_vec, real_vec) > tau for real_vec in real_vectors):
            novel_count += 1

    return novel_count / len(generated_positive_datadicts)


def load_datadicts_from_database(db_filename: str, pth_header: str) -> List[Dict[str, Any]]:
    """Helper to keep DB loading in one place."""
    database = Database.establish_connection(
        filename=db_filename,
        pth_header=pth_header,
    )
    return load_datadicts_from_db(database=database)



if __name__ == "__main__":
    main()
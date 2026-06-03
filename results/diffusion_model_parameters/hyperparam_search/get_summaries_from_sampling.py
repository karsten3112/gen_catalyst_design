from gen_catalyst_design.db import Database, load_datadicts_from_db
from ase_ml_models.yaml import write_to_yaml
from ase.io import read
from scipy.stats import iqr
import numpy as np
import yaml
import os
from typing import Any, Dict, Iterable, List, Sequence


def hamming_distance(x: Sequence[Any], y: Sequence[Any]) -> int:
    """Compute Hamming distance between two equal-length sequences."""
    if len(x) != len(y):
        raise ValueError(
            f"Hamming distance requires equal-length sequences, got {len(x)} and {len(y)}."
        )
    return sum(a != b for a, b in zip(x, y))


def get_assignment_vector(datadict: Dict[str, Any], label_key: str) -> List[Any]:
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


def main():
    # -----------------------------
    # User-configurable parameters
    # -----------------------------
    use_log = True
    dataset = "genetic_alg_dataset"

    # High-rate threshold T in the same space as evaluation:
    # if use_log=True, this should be in log10(rate) space.

    real_dataset = os.path.join("..", "datasets", f"genetic_algorithm_2000.traj")
    real_datadicts = [{"elements":atoms.get_chemical_symbols(), "rate":atoms.info["rate"]} for atoms in read(real_dataset, ":")]
    real_rates = np.array([datadict["rate"] for datadict in real_datadicts])
    if use_log:
        real_rates = np.log10(real_rates)
    

    rate_threshold = np.percentile(real_rates, 80.0)
    print(rate_threshold)
    exit()
    # Hamming tolerance tau:
    # tau = 0 exact match, tau = 1 one node substitution allowed, etc.
    tau = 7

    # Replace with the actual per-node atom assignment key in your datadicts.
    label_key = "elements"    


    models = sorted([model for model in os.listdir(dataset) if "model" in model])
    result_dict: Dict[str, Any] = {}

    for model in models:
        samples_dir = os.path.join(dataset, model, "samples")
        sampling_params_path = os.path.join(samples_dir, "sampling_params.yaml")

        if not os.path.exists(sampling_params_path):
            print(f"Skipping {model}: no sampling_params.yaml found.")
            continue

        with open(sampling_params_path, "r") as fileobj:
            sampling_params = yaml.safe_load(fileobj)

        conditions_dict = sampling_params["conditions"]
        g_scales = sampling_params["guidance_scales"]

        result_dict[model] = {}

        for g_scale in g_scales:
            result_dict[model][f"g_scale_{g_scale}"] = {}

            for condition, condition_value in conditions_dict.items():
                # -----------------------------
                # Load generated samples
                # -----------------------------
                generated_pth_header = os.path.join(samples_dir, f"g_scale_{g_scale}_result")
                generated_db_filename = f"{condition}.db"
                generated_datadicts = load_datadicts_from_database(
                    db_filename=generated_db_filename,
                    pth_header=generated_pth_header,
                )

                # -----------------------------
                # Load real/reference samples
                # -----------------------------
                # Assumes real/reference DBs are organized like:
                # genetic_alg_dataset/reference_samples/{condition}.db
                # and that pth_header can be the directory itself.

                # -----------------------------
                # Existing summary stats
                # -----------------------------
                sampled_rates = extract_rates(
                    datadicts=generated_datadicts,
                    use_log=use_log,
                )

                mean = float(np.mean(sampled_rates))
                median = float(np.median(sampled_rates))
                dist_iqr = float(iqr(sampled_rates))
                mae_score = float(np.mean(np.abs(condition_value - sampled_rates)))

                # -----------------------------
                # New precision / recall / novelty
                # -----------------------------
                generated_positive = filter_positive_samples(
                    datadicts=generated_datadicts,
                    threshold=rate_threshold,
                    use_log=use_log,
                )
                real_positive = filter_positive_samples(
                    datadicts=real_datadicts,
                    threshold=rate_threshold,
                    use_log=use_log,
                )

                precision = compute_precision(
                    generated_datadicts=generated_datadicts,
                    threshold=rate_threshold,
                    use_log=use_log,
                )
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

                summary_dict = {
                    "mean": mean,
                    "median": median,
                    "IQR": dist_iqr,
                    "mae": mae_score,
                    "condition_value": float(condition_value),
                    "rate_threshold": float(rate_threshold),
                    "tau": int(tau),
                    "num_generated": len(generated_datadicts),
                    "num_generated_positive": len(generated_positive),
                    "num_real_positive": len(real_positive),
                    "precision": float(precision),
                    "recall": float(recall),
                    "novelty": float(novelty),
                }

                result_dict[model][f"g_scale_{g_scale}"][condition] = summary_dict

    write_to_yaml("model_params.yaml", result_dict)


if __name__ == "__main__":
    main()
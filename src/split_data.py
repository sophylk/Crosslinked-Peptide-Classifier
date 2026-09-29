import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def build_split_table(dataset: list[dict]) -> pd.DataFrame:
    if not isinstance(dataset, list):
        raise TypeError("dataset must be a list")
    if not dataset:
        raise ValueError("dataset is empty")

    rows = []
    for spectrum in dataset:
        if not isinstance(spectrum, dict):
            raise TypeError("spectrum must be a dict")

        needed_keys = {"run_id", "scan_id", "label", "annotation"}
        no_keys = needed_keys - set(spectrum)
        if no_keys:
            raise KeyError(f"no keys: {sorted(no_keys)}")

        run_id, scan_id, label, annotation  = spectrum["run_id"], spectrum["scan_id"], spectrum["label"], spectrum["annotation"]

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if run_id != run_id.strip():
            raise ValueError("run_id contains surrounding spaces")

        for name, value in [("scan_id", scan_id), ("label", label)]:
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer")

        if scan_id <= 0 or label not in (0, 1):
            raise ValueError("invalid scan_id or label")
        if not isinstance(annotation, dict):
            raise TypeError("annotation must be a dict")

        peptides = []
        for column in ["peptide_a", "peptide_b"]:
            peptide = annotation.get(column)

            if not isinstance(peptide, str) or not peptide.strip():
                raise ValueError(f"missing {column} for {(run_id, scan_id)}")
            peptides.append(peptide.strip())

        peptide_a, peptide_b = sorted(peptides)
        rows.append({
            "run_id": run_id,
            "scan_id": int(scan_id),
            "label": int(label),
            "peptide_a": peptide_a,
            "peptide_b": peptide_b,
        })

    split_table = pd.DataFrame(rows)
    split_table = split_table.sort_values(["run_id", "scan_id"]).reset_index(drop=True)

    if split_table.duplicated(["run_id", "scan_id"]).any():
        raise ValueError("duplicate spectrum keys")
    if set(split_table["label"]) != {0, 1}:
        raise ValueError("dataset must contain both classes")

    peptide_pairs = list(zip(split_table["peptide_a"], split_table["peptide_b"]))

    group_numbers = {}
    for number, pair in enumerate(sorted(set(peptide_pairs))):
        group_numbers[pair] = number

    group_ids = []
    for pair in peptide_pairs:
        group_ids.append(group_numbers[pair])

    split_table["group_id"] = group_ids

    return split_table.set_index(["run_id", "scan_id"])


def find_group_split(data_table: pd.DataFrame, holdout_size: float, random_state: int = 42, n_trials: int = 100, min_remaining_groups: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size must be between 0 and 1")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")

    splitter = GroupShuffleSplit(n_splits=n_trials, test_size=holdout_size, random_state=random_state)
    labels, groups = data_table["label"].to_numpy(), data_table["group_id"].to_numpy()
    best_score, best_indices = float("inf"), None


    for remaining_indices, holdout_indices in splitter.split(data_table, groups=groups):
        remaining_labels, holdout_labels = labels[remaining_indices], labels[holdout_indices]

        if set(remaining_labels) != {0, 1} or set(holdout_labels) != {0, 1}:
            continue
        enough_groups = True

        for label in (0, 1):
            class_indices = remaining_indices[remaining_labels == label]
            class_group_count = len(np.unique(groups[class_indices]))

            if class_group_count < min_remaining_groups:
                enough_groups = False

        if not enough_groups:
            continue

        actual_size = len(holdout_indices) / len(data_table)
        score = abs(actual_size - holdout_size)

        for label in (0, 1):
            class_size = np.sum(labels == label)
            holdout_class_size = np.sum(holdout_labels == label)
            class_fraction = holdout_class_size / class_size
            score += abs(class_fraction - holdout_size)

        if score < best_score:
            best_score = score
            best_indices = (remaining_indices.copy(), holdout_indices.copy())

    if best_indices is None:
        raise ValueError("no suitable group split found; check group counts and sizes or increase n_trials")

    return best_indices


def validate_split(split_table: pd.DataFrame) -> None:
    needed_columns = {"label", "group_id", "split"}
    no_columns = needed_columns - set(split_table.columns)

    if no_columns:
        raise ValueError(f"no columns: {sorted(no_columns)}")
    if split_table.empty:
        raise ValueError("split table is empty")
    if split_table.index.names != ["run_id", "scan_id"]:
        raise ValueError("index must contain run_id and scan_id")
    if not split_table.index.is_unique:
        raise ValueError("duplicate spectrum keys")
    if split_table[list(needed_columns)].isna().any().any():
        raise ValueError("split table contains missing values")
    if set(split_table["split"]) != {"train", "validation", "test"}:
        raise ValueError("train, validation and test are required")


    group_split_counts = (split_table.groupby("group_id")["split"].nunique())
    if group_split_counts.gt(1).any():
        raise ValueError("a peptide pair occurs in different splits")

    for split_name in ["train", "validation", "test"]:
        part = split_table.loc[split_table["split"].eq(split_name)]
        if set(part["label"]) != {0, 1}:
            raise ValueError(f"{split_name} must contain both classes")


def split_dataset(dataset: list[dict], validation_size: float = 0.15, test_size: float = 0.15, random_state: int = 42, n_trials: int = 100) -> pd.DataFrame:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("some data must remain for training")

    split_table = build_split_table(dataset)
    groups_per_class = split_table.groupby("label")["group_id"].nunique()

    if groups_per_class.lt(3).any():
        raise ValueError("at least three groups per class are needed")

    remaining_indices, test_indices = find_group_split(split_table, holdout_size=test_size, random_state=random_state, n_trials=n_trials, min_remaining_groups=2)
    remaining_table = split_table.iloc[remaining_indices]
    relative_validation_size = validation_size * len(split_table) / len(remaining_table)
    train_local, validation_local = find_group_split(remaining_table, holdout_size=relative_validation_size, random_state=random_state, n_trials=n_trials)

    train_keys = remaining_table.iloc[train_local].index
    validation_keys = remaining_table.iloc[validation_local].index
    test_keys = split_table.iloc[test_indices].index

    split_table["split"] = ""
    split_table.loc[train_keys, "split"] = "train"
    split_table.loc[validation_keys, "split"] = "validation"
    split_table.loc[test_keys, "split"] = "test"
    validate_split(split_table)

    return split_table
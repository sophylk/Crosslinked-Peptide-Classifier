import numpy as np
import pandas as pd


def validate_labels(labels: pd.DataFrame) -> None:
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a DataFrame")
    if labels.empty:
        raise ValueError("labels table is empty")

    needed_columns = {"run_id", "scan_id", "label"}
    no_columns = needed_columns - set(labels.columns)

    if no_columns:
        raise ValueError(f"no columns: {sorted(no_columns)}")

    for run_id in labels["run_id"]:
        if not isinstance(run_id, str) or run_id.strip() or run_id != run_id.strip():
            raise TypeError("check labels for run_id")

    if labels["scan_id"].isna().any():
        raise ValueError("there are empty values")
    if not labels["label"].isin([0, 1]).all():
        raise ValueError("label must be 0 or 1")

    label_counts = (labels.groupby(["run_id", "scan_id"])["label"].nunique())
    double_labels = label_counts[label_counts > 1]

    if not double_labels.empty:
        raise ValueError("different labels for the same scan: "f"{double_labels.index.tolist()[:10]}")

    duplicates = labels.duplicated(subset=["run_id", "scan_id"], keep=False)
    if duplicates.any():
        duplicate_keys = (labels.loc[duplicates, ["run_id", "scan_id"]].drop_duplicates())
        raise ValueError("duplicate label keys: "f"{duplicate_keys.head(10).to_dict('records')}")



def build_spectrum_index(spectra: list[dict]) -> dict:
    if not isinstance(spectra, list):
        raise TypeError("spectra must be a list")

    if not spectra:
        raise ValueError("spectra list is empty")

    needed_keys = {"run_id", "scan_id", "mz_array", "intensity_array"}
    spectrum_ind = {}

    for spectrum in spectra:
        if not isinstance(spectrum, dict):
            raise TypeError("spectrum must be a dict")
        
        no_keys = needed_keys - set(spectrum)
        if no_keys:
            raise KeyError(f"no keys: {sorted(no_keys)}")

        run_id, scan_id  = spectrum["run_id"], spectrum["scan_id"]
        if not isinstance(run_id, str) or not run_id.strip():
            raise TypeError("run_id must be a string and not empty")


        if run_id != run_id.strip():
            raise ValueError(f"run_id contains surrounding spaces: {run_id!r}")
        if not isinstance(scan_id, (int, np.integer)) or isinstance(scan_id, (bool, np.bool_)):
            raise TypeError("scan_id must be an integer")

        key = (run_id, int(scan_id))
        if key in spectrum_ind:
            raise ValueError(f"duplicate spectrum key: {key}")
        spectrum_ind[key] = spectrum

    return spectrum_ind


def attach_labels(spectra: list[dict], labels: pd.DataFrame) -> list[dict]:
    validate_labels(labels)
    spectrum_index = build_spectrum_index(spectra)

    for column in labels.columns:
        if column not in {"run_id", "scan_id", "label"}:
            annotation_columns = [column]


    labeled_spectra, missing_keys  = [], []
    for _, row in labels.iterrows():
        key = (row["run_id"], int(row["scan_id"]))

        if key not in spectrum_index:
            missing_keys.append(key)
            continue

        spectrum = spectrum_index[key]
        if "label" in spectrum or "annotation" in spectrum:
            raise ValueError(f"spectrum already contains label or annotation: {key}")

        new_spectrum = spectrum.copy()
        new_spectrum["label"] = int(row["label"])

        annotation = {}
        for column in annotation_columns:
            annotation[column] = row[column]

        new_spectrum["annotation"] = annotation
        labeled_spectra.append(new_spectrum)

    if missing_keys:
        raise ValueError(f"no spectra for {len(missing_keys)} labels: {missing_keys[:10]}")

    return labeled_spectra


def validate_dataset(dataset: list[dict]) -> None:
    if not isinstance(dataset, list):
        raise TypeError("dataset must be a list")
    if not dataset:
        raise ValueError("dataset is empty")

    label_rows = []
    for spectrum in dataset:
        if not isinstance(spectrum, dict):
            raise TypeError("dataset elements must be dicts")

        needed_keys = {"run_id", "scan_id", "label"}
        no_keys = needed_keys - set(spectrum)

        if no_keys:
            raise KeyError(f"no keys: {sorted(no_keys)}")

        label_rows.append({
            "run_id": spectrum["run_id"],
            "scan_id": spectrum["scan_id"],
            "label": spectrum["label"],
        })

    labels_table = pd.DataFrame(label_rows)
    validate_labels(labels_table)
    build_spectrum_index(dataset)

    if set(labels_table["label"]) != {0, 1}:
        raise ValueError("dataset must contain both classes: 0 and 1")


def build_dataset(crosslinked_spectra: list[dict], positive_labels: pd.DataFrame, chimeric_spectra: list[dict], chimera_labels: pd.DataFrame) -> list[dict]:
    validate_labels(positive_labels)
    validate_labels(chimera_labels)

    if not positive_labels["label"].eq(1).all() or not chimera_labels["label"].eq(0).all():
        raise ValueError("positive_labels must contain only label 1 and chimera_labels must contain only label 0")
    
    labeled_crosslinks = attach_labels(crosslinked_spectra, positive_labels)
    labeled_chimeras = attach_labels(chimeric_spectra, chimera_labels)
    dataset = labeled_crosslinks + labeled_chimeras

    validate_dataset(dataset)
    return dataset
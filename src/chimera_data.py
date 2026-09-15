from pathlib import Path
import pandas as pd

def read_chimera_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".tsv":
            raise ValueError("tsv file needed")
    if not path.exists():
        raise FileNotFoundError("no file")
    
    data_table = pd.read_csv(path, sep='\t')
    if data_table.empty:
        raise ValueError(f"table is empty")

    data_table.columns = data_table.columns.str.strip()

    needed_col = {"Spectrum", "Peptide", "Probability", "Protein"}
    no_col = needed_col - set(data_table.columns)

    if no_col:
         raise ValueError(f"no columns: {no_col}")
    

    return data_table


def parse_chimera_spectrum_id(spectrum_id: str) -> tuple[str, int]:
    if not isinstance(spectrum_id, str):
        raise TypeError("spectrum_id must be a string")

    spectrum_id = spectrum_id.strip()

    if not spectrum_id:
        raise ValueError("spectrum_id is empty")

    parts = spectrum_id.rsplit(".", maxsplit=3)

    if len(parts) != 4:
        raise ValueError(f"invalid spectrum identifier: {spectrum_id}")

    run_id, first_scan_text, last_scan_text, charge_text = parts
    run_id = run_id.strip()

    if not run_id:
        raise ValueError(f"Empty run_id: {spectrum_id}")

    try:
        first_scan = int(first_scan_text)
        last_scan = int(last_scan_text)
        charge = int(charge_text)
    except ValueError:
        raise ValueError(f"Invalid scan number or charge: {spectrum_id}")

    
    if first_scan <= 0 or last_scan <= 0 or charge <= 0:
        raise ValueError(f"scan numbers and charge are negative: {spectrum_id}")
    if first_scan != last_scan:
        raise ValueError( f"first and last scan must match: {spectrum_id}")


    return run_id, first_scan


def prepare_chimera_results(data_table: pd.DataFrame) -> pd.DataFrame:
    columns_rename = {"Spectrum": "spectrum_id", "Peptide": "peptide", "Probability": "psm_probability", "Protein": "protein"}
    prepared_data = data_table.rename(columns=columns_rename).copy()
    text_columns = ["spectrum_id", "peptide", "protein"]

    for column in text_columns:
        prepared_data[column] = (prepared_data[column].astype("string").str.strip())

    prepared_data["psm_probability"] = pd.to_numeric( prepared_data["psm_probability"], errors="coerce")
    parsed_ids = prepared_data["spectrum_id"].apply(parse_chimera_spectrum_id)

    prepared_data["run_id"] = [run_id for run_id, scan_id in parsed_ids]
    prepared_data["scan_id"] = [scan_id for run_id, scan_id in parsed_ids]


    return prepared_data


def filter_confident_chimera_psms(data_table: pd.DataFrame, min_probability: float = 0.99, decoy_prefixes: tuple[str, ...] = ("rev_", "decoy_", "reverse_")) -> pd.DataFrame:
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be [0;1]")
    if not decoy_prefixes or any(not prefix.strip() for prefix in decoy_prefixes):
        raise ValueError("decoy prefixes are empty")

    normalized_prefixes = tuple(prefix.strip().lower() for prefix in decoy_prefixes)
    peptide_present = data_table["peptide"].notna() & data_table["peptide"].str.len().gt(0)
    protein_present = data_table["protein"].notna() & data_table["protein"].str.len().gt(0)
    probability_valid = data_table["psm_probability"].between(min_probability, 1.0, inclusive="both")
    is_decoy = data_table["protein"].str.lower().str.startswith(normalized_prefixes, na=False)

    mask = (peptide_present & protein_present & probability_valid & ~is_decoy).fillna(False)

    return data_table.loc[mask].copy().reset_index(drop=True)


def remove_duplicate_peptide_matches(data_table: pd.DataFrame) -> pd.DataFrame:
    sorted_data = data_table.sort_values(by="psm_probability", ascending=False, kind="stable")
    unique_matches = sorted_data.drop_duplicates(subset=["run_id", "scan_id", "peptide"], keep="first")

    return unique_matches.reset_index(drop=True)


def build_chimera_labels(data_table: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = data_table.duplicated(subset=["run_id", "scan_id", "peptide"])
    if duplicate_mask.any():
        raise ValueError("there're duplicates")
    
    label_rows = []
    grouped_data = data_table.groupby(["run_id", "scan_id"], sort=False)

    for (run_id, scan_id), group in grouped_data:
        if group["peptide"].nunique() != 2:
            continue

        ordered_group = group.sort_values(by="peptide", kind="stable")
        first_peptide = ordered_group.iloc[0]
        second_peptide = ordered_group.iloc[1]

        label_rows.append({
            "run_id": run_id,
            "scan_id": int(scan_id),
            "label": 0,
            "peptide_a": first_peptide["peptide"],
            "peptide_b": second_peptide["peptide"],
            "probability_a": first_peptide["psm_probability"],
            "probability_b": second_peptide["psm_probability"],
            "identified_peptide_count": 2,
        })

    columns = ["run_id", "scan_id", "label", "peptide_a", "peptide_b", "probability_a", "probability_b", "identified_peptide_count"]


    return pd.DataFrame(label_rows, columns=columns)


def load_chimera_labels(path: str | Path, min_probability: float = 0.99, decoy_prefixes: tuple[str, ...] = ("rev_", "decoy_", "reverse_")) -> pd.DataFrame:
    raw_results = read_chimera_results(path)
    prepared_results = prepare_chimera_results(raw_results)
    confident_results = filter_confident_chimera_psms(prepared_results, min_probability=min_probability, decoy_prefixes=decoy_prefixes)
    unique_results = remove_duplicate_peptide_matches(confident_results)

    labels = build_chimera_labels(unique_results)

    if labels.empty:
        raise ValueError("no scans")

    return labels
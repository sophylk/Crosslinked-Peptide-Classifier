import numpy as np
import pandas as pd

from .preprocessing import check_spectrum


DSSO_SIGNATURE_MASS_DIFFERENCE_DA = 31.9721
DSSO_SIGNATURE_MASS_TOLERANCE_DA = 0.02
DSSO_MAX_FRAGMENT_CHARGE = 3


def create_bin_edges(min_mz: float = 100.0, max_mz: float = 2000.0, bin_width: float = 1.0) -> np.ndarray:
    settings = np.array([min_mz, max_mz, bin_width], dtype=np.float64)

    if min_mz < 0 or max_mz <= min_mz:
        raise ValueError("max_mz must be greater than min_mz and not negative")

    bin_count_float = (max_mz - min_mz) / bin_width
    bin_count = int(round(bin_count_float))

    if bin_count < 1:
        raise ValueError("at least one bin is needed")
    if not np.isclose(bin_count_float, bin_count, rtol=0, atol=1e-8):
        raise ValueError("m/z range must contain a whole number of bins")

    bin_edges = np.linspace(min_mz, max_mz, bin_count + 1)

    return bin_edges


def bin_spectrum(mz_array: np.ndarray, intensity_array: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    mz = np.asarray(mz_array, dtype=np.float64)
    intensities = np.asarray(intensity_array, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)

    if mz.ndim != 1 or intensities.ndim != 1 or mz.size == 0 or intensities.size == 0 or  mz.size != intensities.size:
        raise ValueError("check peak arrays data")
    if not np.isfinite(mz).all():
        raise ValueError("mz_array contains non-finite values")
    if not np.isfinite(intensities).all():
        raise ValueError("intensity_array contains non-finite values")
    if not (np.diff(edges) > 0).all():
        raise ValueError("bin edges must be strictly increasing")

    binned_intensities, _ = np.histogram(mz, bins=edges, weights=intensities)
    return binned_intensities


def extract_dsso_features(mz_array: np.ndarray, intensity_array: np.ndarray, precursor_charge: float | None = None, mass_difference_da: float = DSSO_SIGNATURE_MASS_DIFFERENCE_DA, mass_tolerance_da: float = DSSO_SIGNATURE_MASS_TOLERANCE_DA, max_fragment_charge: int = DSSO_MAX_FRAGMENT_CHARGE) -> dict:
    mz = np.asarray(mz_array, dtype=np.float64)
    intensities = np.asarray(intensity_array, dtype=np.float64)

    if mz.ndim != 1 or intensities.ndim != 1 or mz.size == 0 or mz.size != intensities.size:
        raise ValueError("check peak arrays data")
    if not np.isfinite(mz).all() or not np.isfinite(intensities).all():
        raise ValueError("peak arrays must contain finite values and positive")
    if (mz <= 0).any() or (intensities <= 0).any():
        raise ValueError("DSSO features require positive m/z and intensity values")
    if not isinstance(max_fragment_charge, (int, np.integer)) or isinstance(max_fragment_charge, (bool, np.bool_)) or max_fragment_charge < 1:
        raise ValueError("max_fragment_charge must be a positive integer")

    charge_limit = int(max_fragment_charge)
    if precursor_charge is not None and not pd.isna(precursor_charge):
        if isinstance(precursor_charge, (bool, np.bool_)):
            raise TypeError("precursor_charge must be numeric, not bool")
        precursor_charge_value = float(precursor_charge)
        if not np.isfinite(precursor_charge_value) or precursor_charge_value <= 0 or not precursor_charge_value.is_integer():
            raise ValueError("precursor_charge must be a positive integer")
        charge_limit = min(charge_limit, int(precursor_charge_value))

    order = np.argsort(mz, kind="stable")
    sorted_mz = mz[order]
    sorted_intensities = intensities[order]
    peak_indices = np.arange(sorted_mz.size)

    counts_by_charge = {charge: 0 for charge in range(1, int(max_fragment_charge) + 1)}
    matched_peak_indices = set()
    neutral_mass_errors = []

    for fragment_charge in range(1, charge_limit + 1):
        expected_spacing = mass_difference_da / fragment_charge
        targets = sorted_mz + expected_spacing
        insertion_points = np.searchsorted(sorted_mz, targets, side="left")
        candidates = []

        for left_index, insertion_point in zip(peak_indices, insertion_points):
            best_candidate = None
            for right_index in (insertion_point - 1, insertion_point):
                if right_index <= left_index or right_index >= sorted_mz.size:
                    continue
                observed_difference = (sorted_mz[right_index] - sorted_mz[left_index]) * fragment_charge
                neutral_mass_error = abs(observed_difference - mass_difference_da)
                candidate = (neutral_mass_error, int(left_index), int(right_index))
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

            if best_candidate is not None and best_candidate[0] <= mass_tolerance_da:
                candidates.append(best_candidate)

        used_for_charge = set()
        for neutral_mass_error, left_index, right_index in sorted(candidates):
            if left_index in used_for_charge or right_index in used_for_charge:
                continue
            used_for_charge.update((left_index, right_index))
            matched_peak_indices.update((left_index, right_index))
            neutral_mass_errors.append(neutral_mass_error)
            counts_by_charge[fragment_charge] += 1

    total_doublet_count = sum(counts_by_charge.values())
    if matched_peak_indices:
        matched_indices = np.fromiter(sorted(matched_peak_indices), dtype=np.int64)
        matched_intensity = sorted_intensities[matched_indices].sum()
    else:
        matched_intensity = 0.0

    total_intensity = sorted_intensities.sum()
    features = {
        "dsso_doublet_count": int(total_doublet_count),
        "dsso_doublet_count_per_100_peaks": float(100.0 * total_doublet_count / sorted_mz.size),
        "dsso_doublet_peak_fraction": float(len(matched_peak_indices) / sorted_mz.size),
        "dsso_doublet_intensity_fraction": float(matched_intensity / total_intensity),
        "dsso_doublet_mean_abs_mass_error_da": float(np.mean(neutral_mass_errors)) if neutral_mass_errors else np.nan,
        "dsso_doublet_min_abs_mass_error_da": float(np.min(neutral_mass_errors)) if neutral_mass_errors else np.nan,
    }
    for fragment_charge, count in counts_by_charge.items():
        features[f"dsso_doublet_count_z{fragment_charge}"] = int(count)

    return features


def extract_spectrum_features(spectrum: dict, bin_edges: np.ndarray, dsso_mass_difference_da: float = DSSO_SIGNATURE_MASS_DIFFERENCE_DA, dsso_mass_tolerance_da: float = DSSO_SIGNATURE_MASS_TOLERANCE_DA, dsso_max_fragment_charge: int = DSSO_MAX_FRAGMENT_CHARGE) -> dict:
    check_spectrum(spectrum)

    if not np.isrealobj(spectrum["mz_array"]) or not np.isrealobj(spectrum["intensity_array"]):
        raise ValueError("peak arrays must contain real numbers")

    mz = np.asarray(spectrum["mz_array"], dtype=np.float64)
    intensities = np.asarray(spectrum["intensity_array"], dtype=np.float64)
    binned_intensities = bin_spectrum(mz, intensities, bin_edges)

    if not np.isclose(intensities.max(), 1.0):
        raise ValueError("normalize intensities before extracting features")
    features = {}

    for index, intensity in enumerate(binned_intensities):
        column_name = f"bin_{index:04d}"
        features[column_name] = float(intensity)

    sorted_intensities = np.sort(intensities)
    top_intensities = sorted_intensities[-10:]

    total_intensity = intensities.sum()
    top10_intensity_fraction = top_intensities.sum() / total_intensity
    
    features["peak_count"] = int(mz.size)
    features["intensity_mean"] = float(intensities.mean())
    features["intensity_median"] = float(np.median(intensities))
    features["intensity_std"] = float(intensities.std(ddof=0))
    features["top10_intensity_fraction"] = float(top10_intensity_fraction)

    for column in ["precursor_mz", "charge"]:
        value = spectrum.get(column)

        if pd.isna(value):
            features[column] = np.nan
            continue
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{column} must be numeric, not bool")

        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{column} must be finite and positive")
        if column == "charge" and not value.is_integer():
            raise ValueError("charge must be an integer")

        features[column] = value

    features.update(extract_dsso_features(mz, intensities, precursor_charge=features["charge"], mass_difference_da=dsso_mass_difference_da, mass_tolerance_da=dsso_mass_tolerance_da, max_fragment_charge=dsso_max_fragment_charge))

    return features


def build_feature_table(spectra: list[dict], min_mz: float = 100.0, max_mz: float = 2000.0, bin_width: float = 1.0, dsso_mass_difference_da: float = DSSO_SIGNATURE_MASS_DIFFERENCE_DA, dsso_mass_tolerance_da: float = DSSO_SIGNATURE_MASS_TOLERANCE_DA, dsso_max_fragment_charge: int = DSSO_MAX_FRAGMENT_CHARGE) -> pd.DataFrame:
    if not isinstance(spectra, list):
        raise TypeError("spectra must be a list")
    if not spectra:
        raise ValueError("spectra list is empty")

    bin_edges = create_bin_edges(min_mz=min_mz, max_mz=max_mz, bin_width=bin_width,)
    feature_rows, spectrum_keys, seen_keys = [], [], set()

    for spectrum in spectra:
        if not isinstance(spectrum, dict):
            raise TypeError("spectrum must be a dict")

        needed_keys = {"run_id", "scan_id"}
        no_keys = needed_keys - set(spectrum)

        if no_keys:
            raise KeyError(f"no keys: {sorted(no_keys)}")

        run_id = spectrum["run_id"]
        scan_id = spectrum["scan_id"]

        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        if not run_id.strip():
            raise ValueError("run_id is empty")
        if run_id != run_id.strip():
            raise ValueError("run_id contains surrounding spaces")

        if not isinstance(scan_id, (int, np.integer)) or isinstance(scan_id, (bool, np.bool_)):
            raise TypeError("scan_id must be an integer")
        if scan_id <= 0:
            raise ValueError("scan_id must be positive")

        key = (run_id, int(scan_id))
        if key in seen_keys:
            raise ValueError(f"duplicate spectrum key: {key}")

        try:
            features = extract_spectrum_features(spectrum, bin_edges, dsso_mass_difference_da=dsso_mass_difference_da, dsso_mass_tolerance_da=dsso_mass_tolerance_da, dsso_max_fragment_charge=dsso_max_fragment_charge)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"cannot extract features for spectrum {key}: {error}") from error

        feature_rows.append(features)
        spectrum_keys.append(key)
        seen_keys.add(key)

    feature_table = pd.DataFrame(feature_rows)
    feature_table.index = pd.MultiIndex.from_tuples(spectrum_keys, names=["run_id", "scan_id"])

    return feature_table

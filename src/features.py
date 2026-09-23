import numpy as np
import pandas as pd

from .preprocessing import check_spectrum


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


def extract_spectrum_features(spectrum: dict, bin_edges: np.ndarray) -> dict:
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

    return features


def build_feature_table(spectra: list[dict], min_mz: float = 100.0, max_mz: float = 2000.0, bin_width: float = 1.0) -> pd.DataFrame:
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
            features = extract_spectrum_features(spectrum, bin_edges)
        except (ValueError, TypeError, KeyError):
            raise ValueError(f"cannot extract features for spectrum {key}")

        feature_rows.append(features)
        spectrum_keys.append(key)
        seen_keys.add(key)

    feature_table = pd.DataFrame(feature_rows)
    feature_table.index = pd.MultiIndex.from_tuples(spectrum_keys, names=["run_id", "scan_id"])

    return feature_table
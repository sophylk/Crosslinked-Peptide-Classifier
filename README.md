# Logistic Regression Model for Identifying Cross-Linked Peptide Spectra

A binary classifier for tandem mass spectra (MS/MS):

- `1` — a spectrum from DSSO-cross-linked peptides;
- `0` — a chimeric spectrum in which exactly two distinct peptides remain after filtering.

The pipeline reads spectra from `mzML` files, creates labels from XlinkX and FragPipe/MSFragger-DDA+ results, cleans and normalizes peaks, builds tabular features, splits the data into train/validation/test sets, and trains a linear PyTorch model. The model is effectively logistic regression implemented as a single `nn.Linear` layer trained with `BCEWithLogitsLoss`.

The main entry point is the unified `run_pipeline.py` command-line interface.

## Pipeline overview

```text
XlinkX CSV ──> positive labels (label=1) ─┐
                                          ├─> dataset ─> group split ─> features ─> model
DDA+ psm.tsv ─> chimeric labels (label=0) ┘       ^
                                                  │
                            matching mzML MS2 ─────┘
```

Each spectrum is identified throughout the project by the `(run_id, scan_id)` pair. `run_id` is derived from the source filename without its extension, so run names in identification tables must match the corresponding `mzML` filenames.

## Quick start

The project was developed with Python 3.13.5. If the existing `.venv` environment is available, activate it with:

```bash
cd /Users/sph/Desktop/projects/peptide_log_reg_model
source .venv/bin/activate
```

Alternatively, create a clean environment and install the dependencies:

```bash
cd /Users/sph/Desktop/projects/peptide_log_reg_model

python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas pyteomics scikit-learn torch tensorboard joblib jupyter ipykernel psims
```

In VS Code, open the project root and select the interpreter from `.venv`. To work through Jupyter, run:

```bash
jupyter lab
```

The current local project environment includes `numpy 2.5.2`, `pandas 3.0.5`, `pyteomics 5.0.1`, `scikit-learn 1.9.1`, `torch 2.14.0`, `tensorboard 2.21.0`, and `joblib 1.6.0`. These versions are not pinned in a lock file.

## Running the complete pipeline

### CLI

From the project root, run:

```bash
python run_pipeline.py train
```

The command uses the default paths and parameters listed below, runs the complete pipeline, evaluates the model on the held-out test set, and saves a new run under `runs/logreg_<UTC timestamp>/`.

To start TensorBoard automatically after training:

```bash
python run_pipeline.py train --serve-tensorboard --open-browser
```

TensorBoard remains active until you press `Ctrl+C`. The `--open-browser` option is optional; without it, the dashboard URL is printed in the terminal.

To list all CLI options:

```bash
python run_pipeline.py train --help
```

Example with custom training parameters:

```bash
python run_pipeline.py train \
  --epochs 150 \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --validation-size 0.15 \
  --test-size 0.15 \
  --output-dir runs
```

Custom input paths can be supplied with `--positive-csv`, `--positive-mzml`, `--chimera-psm`, and `--chimera-mzml`. The stem of each `mzML` filename is used as its `run_id`. The CLI automatically selects labels for that run and stops with a clear error if no matching labels are found.


## Input data and class definitions

### Positive class: DSSO cross-links (`label = 1`)

The source is an XlinkX CSV file. Required columns:

```text
First Scan
XlinkX Score
Delta XlinkX Score
Sequence A
Sequence B
Is Decoy
Crosslinker
Crosslink Type
Spectrum file path
```

By default, a row is retained when:

- `XlinkX Score >= 20`;
- `Delta XlinkX Score >= 20`;
- the match is not a decoy;
- the cross-linker name contains `DSSO`;
- both peptide sequences are present.

If several rows share the same `(run_id, scan_id)`, the row with the highest `XlinkX Score` and `Delta XlinkX Score` is retained.

### Negative class: chimeric spectra (`label = 0`)

The source is a `psm.tsv` file produced by an MSFragger-DDA+/FragPipe search. Required columns:

```text
Spectrum
Peptide
Probability
Protein
```

By default, target PSMs with `Probability >= 0.99` are retained. Proteins with the `rev_`, `decoy_`, or `reverse_` prefix are removed, and duplicate matches of the same peptide sequence within a scan are collapsed. A `0` label is created only for scans that contain exactly two unique peptides after these operations.

This label means “two confidently identified peptides under the configured rules.” It does not guarantee the absence of a weak third component and is not an independent FDR estimate for chimeric scans.

## Features

Before feature extraction, peak arrays are checked for numeric values. Peaks containing `NaN`, infinite, or non-positive values are removed, the remaining peaks are sorted by `m/z`, and intensities are divided by the maximum intensity in the spectrum.

With the default settings `min_mz=100`, `max_mz=2000`, and `bin_width=1`, the pipeline creates 1,900 spectral bins and 7 summary features:

- peak count;
- mean, median, and standard deviation of intensity;
- fraction of total intensity contributed by the ten most intense peaks;
- `precursor_mz`;
- precursor charge.

The model therefore receives 1,907 features. Missing values in `precursor_mz` and `charge` are allowed: they are imputed with the training-set median before all features are standardized.

## Dataset splitting

`split_dataset()` creates train/validation/test sets with default proportions of `70/15/15`. Each group is an ordered-independent peptide pair `(peptide_a, peptide_b)`, so the same pair cannot appear in multiple splits.

This prevents leakage of identical peptide pairs, but it has several limitations:

- the same individual peptide may still appear in different splits;
- the split is not performed by complete experiment or `run_id`;
- every split must contain both classes;
- each class must contain at least three distinct groups.

The classification threshold is selected on the validation set by maximizing F1 for the positive class. The test set must only be used for final evaluation with the already selected threshold.

## Project structure and key functions

### `src/data.py`

- `read_spectra(path)` — reads an `mzML` file through Pyteomics, keeps MS2 spectra, and returns dictionaries containing peak arrays and metadata such as `run_id`, `scan_id`, precursor m/z, charge, and retention time.

### `src/labeling_data.py`

- `read_xlink_data(path)` — reads and validates an XlinkX CSV file.
- `prepare_data(data_table)` — converts column names and types to the internal format and derives `run_id`.
- `filter_crosslinks(...)` — applies score, decoy, and DSSO filters.
- `crosslinked_labels(data_table)` — creates unique class `1` labels.
- `load_positive_labels(path, ...)` — runs the complete positive-label pipeline.

### `src/chimera_data.py`

- `read_chimera_results(path)` — reads and validates `psm.tsv`.
- `parse_chimera_spectrum_id(spectrum_id)` — extracts `run_id` and `scan_id` from an identifier such as `run.101.101.2`.
- `prepare_chimera_results(data_table)` — standardizes columns and data types.
- `filter_confident_chimera_psms(...)` — applies the probability threshold and removes decoys.
- `remove_duplicate_peptide_matches(data_table)` — keeps the best duplicate match for one peptide within one scan.
- `build_chimera_labels(data_table)` — creates class `0` labels for scans containing exactly two peptides.
- `load_chimera_labels(path, ...)` — runs the complete negative-label pipeline.

### `src/preprocessing.py`

- `check_spectrum(spectrum)` — validates the spectrum structure and compatibility of its peak arrays.
- `clean_peaks(mz_array, intensity_array)` — removes invalid peaks and sorts the remaining peaks by `m/z`.
- `normalize_intensities(intensity_array)` — divides intensities by their maximum value.
- `preprocess_spectrum(spectrum)` / `preprocess_spectra(spectra)` — runs complete preprocessing for one spectrum or a list of spectra.

### `src/dataset.py`

- `validate_labels(labels)` — validates types, required columns, key uniqueness, and label values.
- `build_spectrum_index(spectra)` — builds a `{(run_id, scan_id): spectrum}` index and detects duplicate keys.
- `attach_labels(spectra, labels)` — attaches a label and additional annotations to spectra.
- `build_dataset(...)` — combines the positive and negative classes and validates the resulting dataset.
- `validate_dataset(dataset)` — validates key uniqueness and the presence of both classes.

### `src/split_data.py`

- `build_split_table(dataset)` — extracts peptide pairs and assigns their `group_id` values.
- `find_group_split(...)` — evaluates `GroupShuffleSplit` candidates and selects the candidate closest to the requested size and class balance.
- `split_dataset(...)` — creates train/validation/test sets without overlap between identical peptide pairs.
- `validate_split(split_table)` — validates split composition and checks for group leakage.

### `src/features.py`

- `create_bin_edges(...)` — creates boundaries for uniform `m/z` bins.
- `bin_spectrum(...)` — sums peak intensities within the bins.
- `extract_spectrum_features(...)` — extracts binned and summary features from one spectrum.
- `build_feature_table(spectra, ...)` — creates a feature `DataFrame` indexed by `(run_id, scan_id)`.

### `model/train.py`

- `prepare_features(X_train, X_validation)` — fits the median imputer and scaler on the training set only.
- `train_one_epoch(...)` — runs one optimization epoch with L2 regularization.
- `evaluate_epoch(...)` — calculates loss and metrics for the supplied dataset part.
- `predict_probabilities(model, preprocessor, features)` — returns class `1` probabilities after validating feature names and order.
- `train_model(...)` — trains the model with optional class weights, performs early stopping on validation average precision, selects the decision threshold, and saves run artifacts.

### `model/evaluate.py`

- `validate_predictions(...)` — validates labels and probabilities.
- `calculate_metrics(...)` — returns precision, recall, F1, average precision, and the confusion matrix.
- `find_best_threshold(...)` — selects the threshold that maximizes F1 on the precision-recall curve.

### `check_work/`

- `01_data_audit.ipynb` — audits cross-link data and spectrum preprocessing.
- `02_chimera_audit.ipynb` — audits chimeric PSMs and labels.
- `03_dataset_audit.ipynb` — validates the complete dataset, features, split, and training workflow.
- `test_linear_data.ipynb` — a draft notebook for `src/linear_data.py`; that module is currently absent, so the notebook cannot run in the current project state.

### `run_pipeline.py`

- `run_training(args)` — connects label preparation, spectrum loading, preprocessing, splitting, feature extraction, training, and test evaluation in a single run.
- `load_labeled_spectra(...)` — selects labeled scans from an `mzML` file and preprocesses them while reporting missing keys.
- `make_train_validation_test(...)` — creates aligned `X` and `y` objects for all three splits.
- `save_test_artifacts(...)` — saves test metrics and predictions and writes diagnostics to TensorBoard.
- `serve_tensorboard(...)` — launches TensorBoard from the same CLI.

### `tests/test_run_pipeline.py`

Small CLI unit tests verify run selection, `X`/`y` alignment, final artifact creation, and TensorBoard event creation. Run them with:

```bash
python -m unittest discover -s tests -v
```

## Default training parameters

```text
epochs            = 100
batch_size        = 256
learning_rate     = 0.001
l2_strength       = 0.0001
patience          = 15
use_class_weights = True
random_state      = 42
device            = CPU
```

The best epoch is selected by `validation_average_precision`. Early stopping is triggered when the metric does not improve for `patience` consecutive epochs.

## Training artifacts

Each `train_model()` call creates a separate `logreg_<UTC timestamp>` directory inside `output_dir`:

```text
best_weights.pt             best model weights
config.json                 parameters, library versions, and feature names
preprocessor.joblib         fitted median imputer and StandardScaler
train_validation_split.csv  train/validation keys and labels
history.csv                 epoch-level metric history
metrics.json                best epoch, threshold, and validation metrics
test_metrics.json           final test metrics
test_predictions.csv        probabilities and predictions for test spectra
dataset_split_summary.csv   class sizes in train/validation/test
tensorboard/                TensorBoard event files
```

TensorBoard records:

- loss, precision, recall, F1, and average precision for train/validation over epochs;
- final precision, recall, F1, and average precision for the test set;
- the test precision-recall curve;
- predicted-probability distributions for both true classes;
- weights and bias of the best model;
- the test confusion matrix.

To launch TensorBoard for previously saved runs:

```bash
python run_pipeline.py tensorboard --logdir runs --open-browser
```

When `train_model()` is called directly, `runs` is created relative to the current working directory unless `output_dir` is supplied. The CLI always uses `runs` in the project root unless it is overridden with `--output-dir`.


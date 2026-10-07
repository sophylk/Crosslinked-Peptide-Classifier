# DSSO Cross-Linked vs Chimeric Peptide Spectrum Classifier

## Purpose

This project contains a binary logistic regression model that classifies peptide-pair MS/MS spectra as either DSSO-cross-linked or chimeric. The model is intended specifically for spectra that contain evidence of two peptides and were generated using the same type of experimental and identification workflow as the training data.

The model has two output classes:

- `label = 1`: a spectrum assigned to a DSSO-cross-linked peptide pair;
- `label = 0`: a chimeric spectrum assigned to exactly two distinct peptide sequences.

The model does not have a separate class for a spectrum produced by a single linear peptide. If such a spectrum is provided, the binary model will still assign it to class `0` or class `1`, but that prediction is outside the model's intended scope.

## Training Classes and Label Sources

### Class 1: DSSO-cross-linked spectra

Positive labels are created from XlinkX cross-link search results. A peptide-spectrum match is retained when:

- `XlinkX Score >= 20`;
- `Delta XlinkX Score >= 20`;
- the match is not marked as a decoy;
- the cross-linker field contains `DSSO`;
- both peptide sequences are present.

If several XlinkX rows refer to the same spectrum, the row with the highest XlinkX and delta scores is retained. The corresponding MS/MS spectrum receives `label = 1`.

This class teaches the model what spectra identified as DSSO-cross-linked peptide pairs look like.

### Class 0: two-peptide chimeric spectra

Negative labels are created from MSFragger-DDA+/FragPipe `psm.tsv` results. A peptide-spectrum match is retained when:

- its PSM probability is at least `0.99`;
- the peptide and protein fields are present;
- the protein is not marked as a decoy.

Duplicate matches of the same peptide sequence in the same scan are removed. A spectrum receives `label = 0` only when exactly two distinct peptide sequences remain after filtering.

This class teaches the model what a spectrum containing two co-isolated and co-fragmented peptides looks like when the peptides are not identified as a covalently linked DSSO pair.

### How the search results are used

XlinkX and MSFragger/FragPipe results are used to select spectra and assign their training labels. The peptide sequences, XlinkX scores, delta scores, and PSM probabilities are not used as model input features.

The classifier learns from the features extracted from the corresponding MS/MS spectra. Therefore, it learns spectral differences between the two labeled classes rather than reproducing the database-search score thresholds directly.

## Spectrum Preprocessing

Before feature extraction, each spectrum is processed as follows:

1. The `m/z` and intensity arrays are validated.
2. Peaks with non-finite, zero, or negative values are removed.
3. The remaining peaks are sorted by `m/z`.
4. Intensities are divided by the maximum intensity in the spectrum.

This normalization preserves the relative intensity pattern within each spectrum while reducing differences caused only by absolute signal scale.

## Model Features

With the default settings, each spectrum is represented by 1,916 numerical features:

```text
1,900 binned-intensity features
    7 general spectrum features
    9 DSSO-specific features
--------------------------------
1,916 total features
```

### Binned-intensity features

The `m/z` range from `100` to `2000` is divided into bins with a default width of `1 m/z`. The normalized peak intensities within each bin are summed, producing 1,900 features named `bin_0000` through `bin_1899`.

These features allow the model to learn which regions and intensity patterns of an MS/MS spectrum are associated with each class.

### General spectrum features

Seven general features describe the overall spectrum and precursor:

| Feature | Meaning |
|---|---|
| `peak_count` | Number of valid peaks in the spectrum. |
| `intensity_mean` | Mean normalized peak intensity. |
| `intensity_median` | Median normalized peak intensity. |
| `intensity_std` | Standard deviation of normalized peak intensities. |
| `top10_intensity_fraction` | Fraction of total intensity contributed by the ten most intense peaks. |
| `precursor_mz` | Precursor mass-to-charge ratio. |
| `charge` | Precursor charge. |

### DSSO-specific features

DSSO, or disuccinimidyl sulfoxide, is an MS-cleavable cross-linker. Cleavage of its collision-labile C-S bonds can produce alkene (`A`), sulfenic acid (`S`), and thiol (`T`) remnants attached to the linked peptides. The frequently observed alkene/thiol (`A/T`) products form a characteristic doublet with a neutral-mass difference of approximately `31.9721 Da`.

For a fragment charge `z`, the expected separation in the observed spectrum is:

```text
expected m/z separation = 31.9721 / z
```

| Fragment charge | Expected m/z separation |
|---:|---:|
| `z = 1` | `31.972100` |
| `z = 2` | `15.986050` |
| `z = 3` | `10.657367` |

The feature extractor searches for peak pairs with these charge-dependent separations. A candidate pair is accepted when its neutral-mass error is no greater than `0.02 Da` by default. A peak cannot be reused within the same fragment-charge hypothesis.

Nine DSSO-specific features are calculated:

| Feature | Meaning |
|---|---|
| `dsso_doublet_count` | Total number of candidate DSSO doublets across the tested fragment charges. |
| `dsso_doublet_count_z1` | Candidate doublets detected for fragment charge `1`. |
| `dsso_doublet_count_z2` | Candidate doublets detected for fragment charge `2`. |
| `dsso_doublet_count_z3` | Candidate doublets detected for fragment charge `3`. |
| `dsso_doublet_count_per_100_peaks` | Total doublet count normalized per 100 spectrum peaks. |
| `dsso_doublet_peak_fraction` | Fraction of spectrum peaks participating in at least one candidate doublet. |
| `dsso_doublet_intensity_fraction` | Fraction of total spectrum intensity contributed by peaks in candidate doublets. |
| `dsso_doublet_mean_abs_mass_error_da` | Mean absolute neutral-mass error of the accepted doublets. |
| `dsso_doublet_min_abs_mass_error_da` | Smallest absolute neutral-mass error among the accepted doublets. |

If no candidate doublet is detected, the count and fraction features are set to zero. The two mass-error features are represented as missing values and are imputed during preprocessing.

These features give the model explicit information about DSSO-like fragmentation. However, a matching peak separation is supporting evidence rather than definitive proof of a cross-link, because unrelated peaks can produce the same spacing by chance.

## How the Model Learns to Distinguish the Classes

The labeled spectra are divided into training, validation, and test sets. Spectra with the same peptide pair are kept in the same split so that the model is not evaluated on a peptide pair it has already seen during training.

Before training:

1. Missing feature values are replaced with median values learned from the training set.
2. Each feature is standardized using the mean and standard deviation learned from the training set.

The classifier is a linear logistic regression model. For a spectrum with features `x1, x2, ..., xn`, it learns one coefficient for each feature and a bias term:

```text
linear score = w1*x1 + w2*x2 + ... + wn*xn + bias
probability of class 1 = sigmoid(linear score)
```

Features with positive learned coefficients increase the predicted probability of the DSSO-cross-linked class. Features with negative coefficients move the prediction toward the chimeric class. The model considers all features together; it does not classify a spectrum using only the presence or absence of a DSSO doublet.

Training minimizes weighted binary cross-entropy with L2 regularization. Class weighting compensates for unequal numbers of positive and negative training examples, while L2 regularization discourages excessively large coefficients. The best model state is selected using average precision on the validation set, and the final classification threshold is selected from validation predictions using the class-1 F1 score.

For a new spectrum, the model produces a probability for `label = 1`:

- a probability at or above the selected threshold is classified as DSSO-cross-linked;
- a probability below the threshold is classified as chimeric.

## What the Model Can Distinguish

The model is designed to distinguish between two types of peptide-pair spectra represented in its training data:

1. DSSO-cross-linked peptide pairs identified by XlinkX;
2. two-peptide chimeric spectra identified by MSFragger-DDA+/FragPipe.

It bases this distinction on the complete spectral representation: binned fragment intensities, general spectrum characteristics, precursor information, and explicit DSSO-doublet evidence.

The model should not be interpreted as a general detector of every possible cross-linked spectrum. In particular, its output is not reliable for single-peptide spectra, other cross-linker chemistries, or data generated under substantially different experimental conditions unless those cases are represented during training.

## References

1. Kao, A. et al. *Development of a Novel Cross-linking Strategy for Fast and Accurate Identification of Cross-linked Peptides of Protein Complexes.* Molecular & Cellular Proteomics (2011). [https://pmc.ncbi.nlm.nih.gov/articles/PMC3013449/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3013449/)
2. Kolbowski, L. et al. *Improved Peptide Backbone Fragmentation Is the Primary Advantage of MS-Cleavable Crosslinkers.* Analytical Chemistry (2022). [https://doi.org/10.1021/acs.analchem.1c05266](https://doi.org/10.1021/acs.analchem.1c05266)
3. Zhu, Y. et al. *Cross-link Assisted Spatial Proteomics to Map Sub-organelle Proteomes and Membrane Protein Topologies.* Nature Communications (2024). [https://doi.org/10.1038/s41467-024-47569-x](https://doi.org/10.1038/s41467-024-47569-x)

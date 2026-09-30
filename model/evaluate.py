import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score



def validate_predictions(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if labels.size == 0 or probabilities.size == 0:
        raise ValueError("labels and probabilities must not be empty")
    if labels.size != probabilities.size:
        raise ValueError("labels and probabilities must have the same length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if set(labels) != {0, 1}:
        raise ValueError("evaluation data must contain both classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")

    return labels.astype(np.int64), probabilities


def calculate_metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    labels, probabilities = validate_predictions(y_true, probabilities)
    threshold = float(threshold)

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    metrics = {
        "precision": float(precision_score(labels,predictions,pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels,predictions,pos_label=1,zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities, pos_label=1)),
        "confusion_matrix": matrix.tolist(),
        "threshold": threshold,
    }

    return metrics


def find_best_threshold(y_true, probabilities) -> float:
    labels, probabilities = validate_predictions(y_true, probabilities)
    precision_values, recall_values, thresholds = precision_recall_curve(labels,probabilities, pos_label=1)
    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]

    f1_values = []

    for precision, recall in zip(precision_values, recall_values):
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_values.append(f1)

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    return best_threshold

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .evaluate import calculate_metrics, find_best_threshold


def prepare_features(X_train: pd.DataFrame, X_validation: pd.DataFrame) -> tuple[Pipeline, torch.Tensor, torch.Tensor]:
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    train_values = preprocessor.fit_transform(X_train)
    validation_values = preprocessor.transform(X_validation)
    train_tensor = torch.tensor(train_values, dtype=torch.float32)
    validation_tensor = torch.tensor(validation_values, dtype=torch.float32)

    if not torch.isfinite(train_tensor).all() or not torch.isfinite(validation_tensor).all():
        raise ValueError("preprocessed features must be finite")

    return preprocessor, train_tensor, validation_tensor


def train_one_epoch(model: nn.Linear, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.BCEWithLogitsLoss, l2_strength: float) -> float:
    model.train()
    total_objective = 0.0
    total_examples = 0

    for features, labels in train_loader:
        optimizer.zero_grad()
        logits = model(features).squeeze(-1)
        penalty = 0.5 * l2_strength * model.weight.square().sum()
        loss = criterion(logits, labels) + penalty

        if not torch.isfinite(loss):
            raise ValueError("training loss is not finite; check data and learning_rate")

        loss.backward()
        optimizer.step()

        total_objective += loss.item() * len(labels)
        total_examples += len(labels)

    return total_objective / total_examples


def evaluate_epoch(model: nn.Linear, features: torch.Tensor, labels: torch.Tensor) -> tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(features).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probabilities = torch.sigmoid(logits).numpy()

    if not torch.isfinite(loss):
        raise ValueError("evaluation loss is not finite")

    metrics = calculate_metrics(labels.numpy(), probabilities, threshold=0.5)
    metrics["loss"] = float(loss.item())
    return metrics, probabilities


def predict_probabilities(model: nn.Linear, preprocessor: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a nonempty DataFrame")
    if list(features.columns) != list(preprocessor.feature_names_in_):
        raise ValueError("feature names and order must match training")

    values = preprocessor.transform(features)
    tensor = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("preprocessed features must be finite")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor).squeeze(-1)).numpy()

    return probabilities


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series, output_dir: str | Path = "runs", epochs: int = 100, batch_size: int = 256, learning_rate: float = 0.001, l2_strength: float = 0.0001, patience: int = 15, use_class_weights: bool = True, random_state: int = 42) -> dict: 
    for name, value in [("epochs", epochs), ("batch_size", batch_size), ("patience", patience)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(l2_strength) or l2_strength < 0:
        raise ValueError("l2_strength must be nonnegative and finite")
    if not isinstance(random_state, int) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1")
    if not isinstance(use_class_weights, bool):
        raise TypeError("use_class_weights must be True or False")

    torch.manual_seed(random_state)
    preprocessor, train_features, validation_features = prepare_features(X_train, X_validation)
    train_labels = torch.tensor(y_train.to_numpy(dtype=np.float32))
    validation_labels = torch.tensor(y_validation.to_numpy(dtype=np.float32))


    model = nn.Linear(X_train.shape[1], 1, device="cpu")
    negative_count = int(y_train.eq(0).sum())
    positive_count = int(y_train.eq(1).sum())
    positive_weight = negative_count / positive_count if use_class_weights else 1.0

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_dir).expanduser().resolve() / f"logreg_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model": "logistic_regression", "device": "cpu",
        "feature_names": X_train.columns.tolist(),
        "n_features": int(X_train.shape[1]),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "l2_strength": l2_strength,
        "patience": patience, "random_state": random_state,
        "use_class_weights": use_class_weights, "positive_weight": positive_weight,
        "train_class_counts": {"0": negative_count, "1": positive_count},
        "validation_size": len(y_validation),
        "selection_metric": "validation_average_precision",
        "threshold_metric": "validation_f1_class_1",
        "torch_version": str(torch.__version__), "sklearn_version": sklearn.__version__,
    }

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")

    train_rows = y_train.rename("label").to_frame().assign(split="train")
    validation_rows = y_validation.rename("label").to_frame().assign(split="validation")
    pd.concat([train_rows, validation_rows]).to_csv(run_dir / "train_validation_split.csv")

    print(f"Features: {config['n_features']}; parameters: {config['n_parameters']}")
    print(f"Positive weight: {positive_weight:.3f}; results: {run_dir}")

    best_ap = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history_rows = []
    metric_names = ["loss", "precision", "recall", "f1", "average_precision"]

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    try:
        for epoch in range(1, epochs + 1):
            objective = train_one_epoch(model, train_loader, optimizer, criterion, l2_strength)
            train_metrics, _ = evaluate_epoch(model, train_features, train_labels)
            validation_metrics, _ = evaluate_epoch(model, validation_features, validation_labels)

            row = {"epoch": epoch, "training_objective": objective}
            writer.add_scalar("objective/train_weighted_bce_plus_l2", objective, epoch)
            for part_name, metrics in [("train", train_metrics), ("validation", validation_metrics)]:
                for name in metric_names:
                    row[f"{part_name}_{name}"] = metrics[name]
                    writer.add_scalar(f"{name}/{part_name}", metrics[name], epoch)
            history_rows.append(row)

            current_ap = validation_metrics["average_precision"]
            print(f"Epoch {epoch:03d}: train loss={train_metrics['loss']:.4f}, "
                  f"validation loss={validation_metrics['loss']:.4f}, AP={current_ap:.4f}")

            if current_ap > best_ap:
                best_ap = current_ap
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
                torch.save(best_state, run_dir / "best_weights.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping: validation AP did not improve for {patience} epochs")
                break

        model.load_state_dict(best_state)
        _, validation_probabilities = evaluate_epoch(model, validation_features, validation_labels)
        threshold = find_best_threshold(y_validation, validation_probabilities)
        final_metrics = calculate_metrics(y_validation, validation_probabilities, threshold=threshold)

        result_info = {
            "best_epoch": best_epoch,
            "epochs_completed": len(history_rows),
            "threshold": threshold,
            "validation_metrics": final_metrics,
            "confusion_matrix_order": [0, 1],
            "confusion_matrix_axes": "rows=true, columns=predicted",
        }

        (run_dir / "metrics.json").write_text(json.dumps(result_info, indent=2), encoding="utf-8")
        writer.add_text("selection", json.dumps(result_info, indent=2), len(history_rows))
    finally:
        writer.close()
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

    print(f"Best epoch: {best_epoch}; threshold: {threshold:.4f}")
    return {"model": model, "preprocessor": preprocessor, "threshold": threshold, "best_epoch": best_epoch, "validation_metrics": final_metrics, "history": pd.DataFrame(history_rows), "run_dir": run_dir}

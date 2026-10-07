#!/usr/bin/env python3
"""Command-line entry point for training and inspecting the peptide classifier."""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from model.evaluate import calculate_metrics
from model.train import predict_probabilities, train_model
from src.chimera_data import load_chimera_labels
from src.data import read_spectra
from src.dataset import build_dataset, build_spectrum_index
from src.features import build_feature_table
from src.labeling_data import load_positive_labels
from src.preprocessing import preprocess_spectrum
from src.split_data import split_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POSITIVE_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw_files"
    / "stepps_comparison"
    / "Lumos_RSLC3_stepped-HCD_comparison_1.csv"
)
DEFAULT_POSITIVE_MZML = (
    PROJECT_ROOT
    / "data"
    / "raw_files"
    / "Lumos_RSLC3_stepped-HCD_comparison_1.mzML"
)
DEFAULT_CHIMERA_PSM = (
    PROJECT_ROOT
    / "data"
    / "raw_files"
    / "PXD027242"
    / "fragpipe_ddaplus"
    / "trypsin_hcd"
    / "psm.tsv"
)
DEFAULT_CHIMERA_MZML = (
    PROJECT_ROOT / "data" / "raw_files" / "PXD027242" / "Y20210522-02.mzML"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"


def existing_file(value: str) -> Path:
    """Return a resolved input path or raise an argparse-friendly error."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"файл не найден: {path}")
    return path


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("значение должно быть положительным целым числом")
    return number


def random_seed(value: str) -> int:
    """Parse a seed accepted by NumPy, scikit-learn and PyTorch."""
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("random-state должен быть в диапазоне [0, 2**32)")
    return number


def probability(value: str) -> float:
    """Parse a probability in the closed interval [0, 1]."""
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("значение должно находиться в диапазоне [0, 1]")
    return number


def fraction(value: str) -> float:
    """Parse a fraction in the open interval (0, 1)."""
    number = float(value)
    if not np.isfinite(number) or not 0 < number < 1:
        raise argparse.ArgumentTypeError("значение должно находиться в диапазоне (0, 1)")
    return number


def finite_positive_float(value: str) -> float:
    """Parse a finite positive floating-point value."""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным конечным числом")
    return number


def finite_nonnegative_float(value: str) -> float:
    """Parse a finite non-negative floating-point value."""
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("значение должно быть неотрицательным конечным числом")
    return number


def select_run_labels(labels: pd.DataFrame, run_id: str, source_name: str) -> pd.DataFrame:
    """Select one mzML run and fail with a useful list of available run IDs."""
    selected = labels.loc[labels["run_id"].eq(run_id)].copy().reset_index(drop=True)
    if selected.empty:
        available = sorted(labels["run_id"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"Для {source_name} не найдены метки run_id={run_id!r}. "
            f"Доступные run_id: {available[:20]}"
        )
    return selected


def load_labeled_spectra(path: Path, labels: pd.DataFrame, source_name: str) -> list[dict]:
    """Read one mzML file, select labeled scans and preprocess their peaks."""
    print(f"[{source_name}] Чтение MS2 из {path}")
    all_spectra = read_spectra(path)
    spectrum_index = build_spectrum_index(all_spectra)
    label_keys = [
        (row.run_id, int(row.scan_id))
        for row in labels[["run_id", "scan_id"]].itertuples(index=False)
    ]
    missing_keys = [key for key in label_keys if key not in spectrum_index]
    if missing_keys:
        raise ValueError(
            f"Для {source_name} отсутствуют {len(missing_keys)} размеченных спектров: "
            f"{missing_keys[:10]}"
        )

    processed_spectra = []
    for key in label_keys:
        try:
            processed_spectra.append(preprocess_spectrum(spectrum_index[key]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Ошибка предобработки спектра {key}: {error}") from error

    print(
        f"[{source_name}] MS2 в файле: {len(all_spectra)}; "
        f"выбрано и обработано: {len(processed_spectra)}"
    )
    return processed_spectra


def make_train_validation_test(
    feature_table: pd.DataFrame,
    split_table: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Align the feature matrix and labels for all three dataset parts."""
    parts = {}
    for split_name in ("train", "validation", "test"):
        rows = split_table.loc[split_table["split"].eq(split_name)]
        features = feature_table.loc[rows.index].copy()
        labels = rows["label"].astype(int).copy()
        if not features.index.equals(labels.index):
            raise ValueError(f"Нарушено соответствие X и y для части {split_name}")
        parts[split_name] = (features, labels)
    return parts


def save_test_artifacts(
    result: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    metrics: dict,
    split_table: pd.DataFrame,
) -> None:
    """Save held-out predictions and append test diagnostics to TensorBoard."""
    run_dir = Path(result["run_dir"])
    threshold = float(result["threshold"])
    predictions = (probabilities >= threshold).astype(np.int64)

    metrics_payload = {
        "best_epoch": int(result["best_epoch"]),
        "threshold_selected_on": "validation_f1_class_1",
        "test_size": int(len(y_test)),
        "test_metrics": metrics,
        "confusion_matrix_order": [0, 1],
        "confusion_matrix_axes": "rows=true, columns=predicted",
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    prediction_table = pd.DataFrame(
        {
            "label": y_test.to_numpy(dtype=np.int64),
            "probability_class_1": probabilities,
            "prediction": predictions,
            "threshold": threshold,
        },
        index=X_test.index,
    )
    prediction_table.to_csv(run_dir / "test_predictions.csv")

    split_summary = (
        pd.crosstab(split_table["split"], split_table["label"])
        .reindex(index=["train", "validation", "test"], columns=[0, 1], fill_value=0)
    )
    split_summary.columns = ["class_0", "class_1"]
    split_summary["total"] = split_summary.sum(axis=1)
    split_summary.to_csv(run_dir / "dataset_split_summary.csv")

    step = int(result["best_epoch"])
    labels_tensor = torch.tensor(y_test.to_numpy(dtype=np.int64))
    probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
    confusion_matrix = metrics["confusion_matrix"]
    confusion_markdown = (
        "| true \\ predicted | 0 | 1 |\n"
        "|---|---:|---:|\n"
        f"| 0 | {confusion_matrix[0][0]} | {confusion_matrix[0][1]} |\n"
        f"| 1 | {confusion_matrix[1][0]} | {confusion_matrix[1][1]} |"
    )

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    try:
        for name in ("precision", "recall", "f1", "average_precision"):
            writer.add_scalar(f"test/{name}", metrics[name], step)
        writer.add_scalar("test/threshold", threshold, step)
        writer.add_pr_curve(
            "test/precision_recall_curve",
            labels_tensor,
            probabilities_tensor,
            global_step=step,
        )
        for label in (0, 1):
            class_probabilities = probabilities_tensor[labels_tensor.eq(label)]
            writer.add_histogram(
                f"test/probability_true_class_{label}",
                class_probabilities,
                global_step=step,
            )
        dsso_columns = [column for column in X_test.columns if column.startswith("dsso_")]
        for column in dsso_columns:
            for label in (0, 1):
                class_values = X_test.loc[y_test.eq(label), column].to_numpy(dtype=np.float64)
                class_values = class_values[np.isfinite(class_values)]
                if class_values.size == 0:
                    continue
                writer.add_histogram(
                    f"features/test_true_class_{label}/{column}",
                    class_values,
                    global_step=step,
                )
                writer.add_scalar(
                    f"feature_means/test_true_class_{label}/{column}",
                    float(class_values.mean()),
                    global_step=step,
                )
        writer.add_histogram("model/weights", result["model"].weight, global_step=step)
        writer.add_histogram("model/bias", result["model"].bias, global_step=step)
        writer.add_text("test/confusion_matrix", confusion_markdown, global_step=step)
        writer.flush()
    finally:
        writer.close()


def save_pipeline_config(
    result: dict,
    args: argparse.Namespace,
    dataset_size: int,
    feature_table: pd.DataFrame,
) -> None:
    """Save data and feature-extraction settings required to reproduce a run."""
    dsso_feature_names = [
        column for column in feature_table.columns if column.startswith("dsso_")
    ]
    payload = {
        "input_files": {
            "positive_csv": str(args.positive_csv),
            "positive_mzml": str(args.positive_mzml),
            "chimera_psm": str(args.chimera_psm),
            "chimera_mzml": str(args.chimera_mzml),
        },
        "dataset_size": int(dataset_size),
        "feature_count": int(feature_table.shape[1]),
        "binning": {
            "min_mz": args.min_mz,
            "max_mz": args.max_mz,
            "bin_width": args.bin_width,
        },
        "dsso_signature": {
            "mass_difference_da": args.dsso_mass_difference,
            "neutral_mass_tolerance_da": args.dsso_mass_tolerance,
            "max_fragment_charge": args.dsso_max_fragment_charge,
            "expected_mz_spacing_by_charge": {
                str(charge): args.dsso_mass_difference / charge
                for charge in range(1, args.dsso_max_fragment_charge + 1)
            },
            "feature_names": dsso_feature_names,
        },
    }
    run_dir = Path(result["run_dir"])
    (run_dir / "pipeline_config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def print_metrics(title: str, metrics: dict) -> None:
    """Print scalar metrics and a confusion matrix in a compact form."""
    print(f"\n{title}")
    for name in ("precision", "recall", "f1", "average_precision", "threshold"):
        print(f"  {name}: {metrics[name]:.6f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")


def serve_tensorboard(logdir: Path, host: str, port: int, open_browser: bool) -> None:
    """Start TensorBoard and keep the process alive until Ctrl+C."""
    logdir = logdir.expanduser().resolve()
    if not logdir.is_dir():
        raise FileNotFoundError(f"Папка с логами не найдена: {logdir}")

    tensorboard = program.TensorBoard()
    tensorboard.configure(
        argv=[
            None,
            "--logdir",
            str(logdir),
            "--host",
            host,
            "--port",
            str(port),
        ]
    )
    url = tensorboard.launch()
    print(f"TensorBoard запущен: {url}")
    print(f"Логи: {logdir}")
    print("Для остановки нажмите Ctrl+C.")
    if open_browser:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTensorBoard остановлен.")


def run_training(args: argparse.Namespace) -> Path:
    """Execute the complete data, feature, training and evaluation pipeline."""
    input_paths = {
        "positive-csv": args.positive_csv,
        "positive-mzml": args.positive_mzml,
        "chimera-psm": args.chimera_psm,
        "chimera-mzml": args.chimera_mzml,
    }
    missing_paths = [f"{name}={path}" for name, path in input_paths.items() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Не найдены входные файлы: " + ", ".join(missing_paths))
    if args.validation_size + args.test_size >= 1:
        raise ValueError("Сумма validation-size и test-size должна быть меньше 1")
    if args.max_mz <= args.min_mz:
        raise ValueError("max-mz должен быть больше min-mz")

    positive_run_id = args.positive_mzml.stem
    chimera_run_id = args.chimera_mzml.stem

    print("[1/6] Подготовка меток")
    all_positive_labels = load_positive_labels(
        args.positive_csv,
        min_score=args.min_xlink_score,
        min_delta_score=args.min_delta_score,
    )
    all_chimera_labels = load_chimera_labels(
        args.chimera_psm,
        min_probability=args.min_probability,
        decoy_prefixes=tuple(args.decoy_prefix or ["rev_", "decoy_", "reverse_"]),
    )
    positive_labels = select_run_labels(
        all_positive_labels,
        positive_run_id,
        "положительного класса",
    )
    chimera_labels = select_run_labels(
        all_chimera_labels,
        chimera_run_id,
        "химерного класса",
    )
    print(f"  label=1: {len(positive_labels)}; label=0: {len(chimera_labels)}")

    print("[2/6] Загрузка и предобработка спектров")
    positive_spectra = load_labeled_spectra(
        args.positive_mzml,
        positive_labels,
        "label=1",
    )
    chimera_spectra = load_labeled_spectra(
        args.chimera_mzml,
        chimera_labels,
        "label=0",
    )

    print("[3/6] Сборка и разделение датасета")
    dataset = build_dataset(
        crosslinked_spectra=positive_spectra,
        positive_labels=positive_labels,
        chimeric_spectra=chimera_spectra,
        chimera_labels=chimera_labels,
    )
    split_table = split_dataset(
        dataset,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        n_trials=args.split_trials,
    )
    split_counts = pd.crosstab(split_table["split"], split_table["label"])
    print(split_counts.to_string())

    print("[4/6] Извлечение признаков")
    feature_table = build_feature_table(
        dataset,
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bin_width=args.bin_width,
        dsso_mass_difference_da=args.dsso_mass_difference,
        dsso_mass_tolerance_da=args.dsso_mass_tolerance,
        dsso_max_fragment_charge=args.dsso_max_fragment_charge,
    )
    parts = make_train_validation_test(feature_table, split_table)
    X_train, y_train = parts["train"]
    X_validation, y_validation = parts["validation"]
    X_test, y_test = parts["test"]
    print(f"  Спектров: {len(dataset)}; признаков: {feature_table.shape[1]}")

    print("[5/6] Обучение модели")
    result = train_model(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        patience=args.patience,
        use_class_weights=args.use_class_weights,
        random_state=args.random_state,
    )
    save_pipeline_config(result, args, len(dataset), feature_table)

    print("[6/6] Итоговая оценка на test")
    test_probabilities = predict_probabilities(result["model"], result["preprocessor"], X_test)
    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold=result["threshold"],
    )
    save_test_artifacts(
        result,
        X_test,
        y_test,
        test_probabilities,
        test_metrics,
        split_table,
    )

    print_metrics("Validation metrics", result["validation_metrics"])
    print_metrics("Test metrics", test_metrics)
    run_dir = Path(result["run_dir"])
    print(f"\nГотово. Артефакты запуска: {run_dir}")
    print(f"TensorBoard: python {Path(__file__).name} tensorboard --logdir {args.output_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Обучение классификатора cross-link/химера и просмотр TensorBoard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="запустить полный пайплайн подготовки данных, обучения и test-оценки",
    )
    train_parser.add_argument(
        "--positive-csv",
        type=existing_file,
        default=DEFAULT_POSITIVE_CSV,
        help=f"CSV XlinkX (по умолчанию: {DEFAULT_POSITIVE_CSV})",
    )
    train_parser.add_argument(
        "--positive-mzml",
        type=existing_file,
        default=DEFAULT_POSITIVE_MZML,
        help=f"mzML положительного класса (по умолчанию: {DEFAULT_POSITIVE_MZML})",
    )
    train_parser.add_argument(
        "--chimera-psm",
        type=existing_file,
        default=DEFAULT_CHIMERA_PSM,
        help=f"DDA+ psm.tsv (по умолчанию: {DEFAULT_CHIMERA_PSM})",
    )
    train_parser.add_argument(
        "--chimera-mzml",
        type=existing_file,
        default=DEFAULT_CHIMERA_MZML,
        help=f"mzML химерного класса (по умолчанию: {DEFAULT_CHIMERA_MZML})",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"корневая папка запусков (по умолчанию: {DEFAULT_OUTPUT_DIR})",
    )

    data_group = train_parser.add_argument_group("фильтрация и признаки")
    data_group.add_argument("--min-xlink-score", type=finite_nonnegative_float, default=20.0)
    data_group.add_argument("--min-delta-score", type=finite_nonnegative_float, default=20.0)
    data_group.add_argument("--min-probability", type=probability, default=0.99)
    data_group.add_argument(
        "--decoy-prefix",
        action="append",
        default=None,
        help="префикс decoy; можно повторить (по умолчанию: rev_, decoy_, reverse_)",
    )
    data_group.add_argument("--min-mz", type=finite_nonnegative_float, default=100.0)
    data_group.add_argument("--max-mz", type=finite_positive_float, default=2000.0)
    data_group.add_argument("--bin-width", type=finite_positive_float, default=1.0)
    data_group.add_argument(
        "--dsso-mass-difference",
        type=finite_positive_float,
        default=31.9721,
        help="DSSO A/T neutral-mass difference in Da (default: 31.9721)",
    )
    data_group.add_argument(
        "--dsso-mass-tolerance",
        type=finite_positive_float,
        default=0.02,
        help="allowed neutral-mass error for a candidate doublet in Da (default: 0.02)",
    )
    data_group.add_argument(
        "--dsso-max-fragment-charge",
        type=positive_integer,
        default=3,
        help="largest fragment-charge hypothesis used for DSSO doublets (default: 3)",
    )

    split_group = train_parser.add_argument_group("разделение")
    split_group.add_argument("--validation-size", type=fraction, default=0.15)
    split_group.add_argument("--test-size", type=fraction, default=0.15)
    split_group.add_argument("--split-trials", type=positive_integer, default=100)
    split_group.add_argument("--random-state", type=random_seed, default=42)

    model_group = train_parser.add_argument_group("обучение")
    model_group.add_argument("--epochs", type=positive_integer, default=100)
    model_group.add_argument("--batch-size", type=positive_integer, default=256)
    model_group.add_argument("--learning-rate", type=finite_positive_float, default=0.001)
    model_group.add_argument("--l2-strength", type=finite_nonnegative_float, default=0.0001)
    model_group.add_argument("--patience", type=positive_integer, default=15)
    model_group.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        help="не компенсировать дисбаланс классов через pos_weight",
    )
    train_parser.set_defaults(use_class_weights=True)
    train_parser.add_argument(
        "--serve-tensorboard",
        action="store_true",
        help="после обучения запустить TensorBoard и ждать Ctrl+C",
    )
    train_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="открыть TensorBoard в браузере (вместе с --serve-tensorboard)",
    )
    train_parser.add_argument("--tensorboard-host", default="127.0.0.1")
    train_parser.add_argument("--tensorboard-port", type=positive_integer, default=6006)

    tensorboard_parser = subparsers.add_parser(
        "tensorboard",
        help="запустить TensorBoard для сохранённых запусков",
    )
    tensorboard_parser.add_argument("--logdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    tensorboard_parser.add_argument("--host", default="127.0.0.1")
    tensorboard_parser.add_argument("--port", type=positive_integer, default=6006)
    tensorboard_parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "tensorboard":
        serve_tensorboard(args.logdir, args.host, args.port, args.open_browser)
        return 0

    if args.decoy_prefix is None:
        args.decoy_prefix = ["rev_", "decoy_", "reverse_"]
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.open_browser and not args.serve_tensorboard:
        parser.error("--open-browser для train требует --serve-tensorboard")

    run_training(args)
    if args.serve_tensorboard:
        serve_tensorboard(
            args.output_dir,
            args.tensorboard_host,
            args.tensorboard_port,
            args.open_browser,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

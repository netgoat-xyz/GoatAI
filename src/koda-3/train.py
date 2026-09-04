"""Train and evaluate Koda-3 in one streaming pass over a CSV file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from model import Koda3Classifier, normalize_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def discover_schema(path: Path, label_column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        header = next(csv.reader(file))
    labels = {label_column.lower(), "label"}
    return [name for name in header if name.lower() not in labels]


def parse_row(row: dict[str, str], features: list[str], label_column: str) -> tuple[list[float], int] | None:
    try:
        values = [float(row[name]) for name in features]
        if not all(value == value and abs(value) != float("inf") for value in values):
            return None
        return values, normalize_label(row[label_column])
    except (KeyError, TypeError, ValueError):
        return None


def is_validation_row(index: int, seed: int, fraction: float) -> bool:
    digest = hashlib.blake2s(f"{seed}:{index}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") / 2**32 < fraction


def iter_rows(path: Path, features: list[str], label_column: str):
    with path.open(newline="", encoding="utf-8-sig") as file:
        for index, row in enumerate(csv.DictReader(file)):
            yield index, parse_row(row, features, label_column)


def metrics(truth: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(actual == predicted == 1 for actual, predicted in zip(truth, predictions))
    tn = sum(actual == predicted == 0 for actual, predicted in zip(truth, predictions))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(truth, predictions))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(truth, predictions))
    total = max(len(truth), 1)
    return {
        "rows": len(truth),
        "accuracy": (tp + tn) / total,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tune_threshold(model: Koda3Classifier, rows: list[list[float]], truth: list[int]) -> float:
    probabilities = [model.predict_probability(row) for row in rows]
    best = (float("-inf"), 0.5)
    for step in range(1, 100):
        threshold = step / 100
        candidate = metrics(truth, [int(value >= threshold) for value in probabilities])
        balanced_accuracy = (float(candidate["recall"]) + float(candidate["specificity"])) / 2
        score = balanced_accuracy - abs(threshold - 0.5) * 1e-9
        if score > best[0]:
            best = (score, threshold)
    return best[1]


def evaluate_dataset(model: Koda3Classifier, path: Path, label_column: str = "Label") -> dict:
    schema = discover_schema(path, label_column)
    if schema != model.feature_names:
        raise ValueError(
            f"Test schema does not match model schema: expected {model.feature_names}, received {schema}"
        )
    rows: list[list[float]] = []
    truth: list[int] = []
    skipped = 0
    for _, parsed in iter_rows(path, schema, label_column):
        if parsed is None:
            skipped += 1
            continue
        values, label = parsed
        rows.append(values)
        truth.append(label)
    result = metrics(truth, model.predict_many(rows))
    result["skipped_rows"] = skipped
    result["path"] = str(path)
    return result


def train_model(
    data_path: Path,
    model_path: Path,
    label_column: str = "Label",
    validation_fraction: float = 0.2,
    seed: int = 42,
    test_data_path: Path | None = None,
) -> dict:
    started = time.perf_counter()
    features = discover_schema(data_path, label_column)
    model = Koda3Classifier(features)
    validation_rows: list[list[float]] = []
    validation_truth: list[int] = []
    skipped = 0

    for index, parsed in iter_rows(data_path, features, label_column):
        if parsed is None:
            skipped += 1
            continue
        values, label = parsed
        if validation_fraction and is_validation_row(index, seed, validation_fraction):
            validation_rows.append(values)
            validation_truth.append(label)
        else:
            model.update(values, label)

    model.finalize()
    if validation_rows:
        model.threshold = tune_threshold(model, validation_rows, validation_truth)
        evaluation = metrics(validation_truth, model.predict_many(validation_rows))
    else:
        evaluation = {"rows": 0}
    model.save(model_path)

    report = {
        "model": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "features": features,
        "training_rows": sum(item.count for item in model.classes),
        "class_counts": [item.count for item in model.classes],
        "skipped_rows": skipped,
        "threshold": model.threshold,
        "training_seconds": time.perf_counter() - started,
        "validation": evaluation,
    }
    if test_data_path is not None:
        report["test"] = evaluate_dataset(model, test_data_path, label_column)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact streaming Koda-3 DDoS classifier.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "koda-3.pkl")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-data",
        type=Path,
        default=PROJECT_ROOT / "data" / "koda-3-test.csv",
        help="External labeled CSV used only after fitting and threshold selection.",
    )
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")
    report = train_model(
        args.data,
        args.model,
        args.label_column,
        args.validation_fraction,
        args.seed,
        args.test_data if args.test_data.exists() else None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

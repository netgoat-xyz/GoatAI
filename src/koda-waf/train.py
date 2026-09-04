"""Train Koda-WAF's compact learned token model from labeled request CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from engine import KodaWAF, RequestView, WAFModel, extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("method", "path", "query", "headers", "body")


def request_from_row(row: dict[str, str]) -> RequestView:
    return RequestView(*(row.get(field, "") for field in FIELDS))


def malicious_label(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "attack", "malicious"}


def validation_row(index: int, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2s(f"{seed}:{index}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") / 2**32 < fraction


def classification_metrics(truth: list[bool], predictions: list[bool]) -> dict:
    tp = sum(actual and predicted for actual, predicted in zip(truth, predictions))
    tn = sum(not actual and not predicted for actual, predicted in zip(truth, predictions))
    fp = sum(not actual and predicted for actual, predicted in zip(truth, predictions))
    fn = sum(actual and not predicted for actual, predicted in zip(truth, predictions))
    return {
        "rows": len(truth),
        "accuracy": (tp + tn) / max(len(truth), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(scores: list[float], truth: list[bool]) -> float:
    positives = sum(truth)
    negatives = len(truth) - positives
    tp = fp = 0
    best = (float("-inf"), 0.5)
    ranked = sorted(zip(scores, truth), reverse=True)
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / max(positives, 1)
        specificity = (negatives - fp) / max(negatives, 1)
        candidate = (recall + specificity) / 2 - abs(threshold - 0.5) * 1e-12
        if candidate > best[0]:
            best = (candidate, threshold)
    return min(max(best[1], 1e-9), 1 - 1e-9)


def train_model(
    data_path: Path,
    output_path: Path,
    validation_fraction: float = 0.1,
    seed: int = 42,
    minimum_count: int = 2,
    maximum_features: int = 60_000,
    hard_negative_weight: float = 3.0,
) -> dict:
    started = time.perf_counter()
    counts = [Counter(), Counter()]
    class_rows = [0, 0]
    effective_class_rows = [0.0, 0.0]
    validation_requests: list[RequestView] = []
    validation_truth: list[bool] = []

    with data_path.open(newline="", encoding="utf-8-sig") as file:
        for index, row in enumerate(csv.DictReader(file)):
            request = request_from_row(row)
            label = malicious_label(row.get("label", ""))
            if validation_row(index, validation_fraction, seed):
                validation_requests.append(request)
                validation_truth.append(label)
                continue
            class_index = int(label)
            class_rows[class_index] += 1
            source = row.get("source", "").lower()
            sample_weight = (
                hard_negative_weight
                if not label and ("hard_negative" in source or "false_positive" in source)
                else 1.0
            )
            effective_class_rows[class_index] += sample_weight
            for feature in extract_features(request):
                counts[class_index][feature] += (
                    1.0 if feature.startswith("signature") else sample_weight
                )

    if min(class_rows) == 0:
        raise ValueError("Training data must contain benign and malicious rows")
    alpha = 1.0
    candidates = set(counts[0]) | set(counts[1])
    ranked: list[tuple[float, str, float]] = []
    for feature in candidates:
        total = counts[0][feature] + counts[1][feature]
        if total < minimum_count:
            continue
        denominators = class_rows if feature.startswith("signature") else effective_class_rows
        benign_probability = (counts[0][feature] + alpha) / (denominators[0] + 2 * alpha)
        malicious_probability = (counts[1][feature] + alpha) / (denominators[1] + 2 * alpha)
        weight = math.log(malicious_probability / benign_probability)
        importance = abs(weight) * math.log1p(total)
        ranked.append((importance, feature, weight))
    ranked.sort(reverse=True)
    weights = {feature: weight for _, feature, weight in ranked[:maximum_features]}
    bias = math.log((class_rows[1] + alpha) / (class_rows[0] + alpha))
    model = WAFModel(weights, bias)

    validation_scores = [model.probability(request) for request in validation_requests]
    model.threshold = choose_threshold(validation_scores, validation_truth)
    validation_predictions = [score >= model.threshold for score in validation_scores]
    model.save(output_path)
    return {
        "model": str(output_path),
        "model_size_bytes": output_path.stat().st_size,
        "features": len(weights),
        "training_rows": sum(class_rows),
        "class_rows": {"benign": class_rows[0], "malicious": class_rows[1]},
        "effective_class_rows": {
            "benign": effective_class_rows[0],
            "malicious": effective_class_rows[1],
        },
        "hard_negative_weight": hard_negative_weight,
        "threshold": model.threshold,
        "training_seconds": time.perf_counter() - started,
        "validation": classification_metrics(validation_truth, validation_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the binary Koda-WAF model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "koda-waf.pkl")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-count", type=int, default=2)
    parser.add_argument("--maximum-features", type=int, default=60_000)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    args = parser.parse_args()
    report = train_model(
        args.data, args.model, args.validation_fraction,
        minimum_count=args.minimum_count, maximum_features=args.maximum_features,
        hard_negative_weight=args.hard_negative_weight,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

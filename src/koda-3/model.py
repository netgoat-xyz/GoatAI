"""Compact streaming Gaussian classifier used by Koda-3."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def normalize_label(value: object) -> int:
    text = str(value).strip().lower()
    return 0 if text in {"0", "0.0", "benign", "normal", "false"} else 1


@dataclass
class RunningStats:
    count: int
    mean: list[float]
    m2: list[float]

    @classmethod
    def empty(cls, dimensions: int) -> "RunningStats":
        return cls(0, [0.0] * dimensions, [0.0] * dimensions)

    def update(self, values: Sequence[float]) -> None:
        self.count += 1
        for index, value in enumerate(values):
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count
            self.m2[index] += delta * (value - self.mean[index])

    def variances(self) -> list[float]:
        denominator = max(self.count - 1, 1)
        return [value / denominator for value in self.m2]


class Koda3Classifier:
    """Diagonal Gaussian classifier with O(features) inference cost."""

    format_version = 1

    def __init__(self, feature_names: Sequence[str]) -> None:
        if not feature_names:
            raise ValueError("Koda-3 requires at least one feature")
        self.feature_names = list(feature_names)
        self.classes = [RunningStats.empty(len(self.feature_names)) for _ in range(2)]
        self.threshold = 0.5
        self._variances: list[list[float]] | None = None

    def update(self, values: Sequence[float], label: int) -> None:
        if len(values) != len(self.feature_names):
            raise ValueError("Feature count does not match the model schema")
        self.classes[int(label)].update(values)
        self._variances = None

    def finalize(self) -> None:
        if any(stats.count < 2 for stats in self.classes):
            raise ValueError("Koda-3 requires at least two valid rows for each class")

        raw = [stats.variances() for stats in self.classes]
        pooled = []
        for index in range(len(self.feature_names)):
            class_means = [stats.mean[index] for stats in self.classes]
            scale = max(raw[0][index], raw[1][index], (class_means[0] - class_means[1]) ** 2, 1.0)
            pooled.append(scale * 1e-8)
        self._variances = [
            [max(value, pooled[index]) for index, value in enumerate(class_variance)]
            for class_variance in raw
        ]

    def _log_likelihood(self, values: Sequence[float], label: int) -> float:
        if self._variances is None:
            self.finalize()
        assert self._variances is not None
        stats = self.classes[label]
        total_count = self.classes[0].count + self.classes[1].count
        score = math.log(stats.count / total_count)
        for value, mean, variance in zip(values, stats.mean, self._variances[label]):
            score -= 0.5 * (math.log(2.0 * math.pi * variance) + ((value - mean) ** 2 / variance))
        return score

    def predict_probability(self, values: Sequence[float]) -> float:
        benign = self._log_likelihood(values, 0)
        attack = self._log_likelihood(values, 1)
        difference = max(min(attack - benign, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-difference))

    def predict(self, values: Sequence[float]) -> int:
        return int(self.predict_probability(values) >= self.threshold)

    def predict_many(self, rows: Iterable[Sequence[float]]) -> list[int]:
        return [self.predict(row) for row in rows]

    def to_dict(self) -> dict:
        self.finalize()
        return {
            "format": "koda-3",
            "version": self.format_version,
            "features": self.feature_names,
            "threshold": self.threshold,
            "classes": [
                {"count": stats.count, "mean": stats.mean, "m2": stats.m2}
                for stats in self.classes
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self.to_dict(), file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "Koda3Classifier":
        with path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("format") != "koda-3" or payload.get("version") != cls.format_version:
            raise ValueError("Unsupported Koda-3 model format")
        model = cls(payload["features"])
        model.threshold = float(payload.get("threshold", 0.5))
        model.classes = [
            RunningStats(int(item["count"]), list(map(float, item["mean"])), list(map(float, item["m2"])))
            for item in payload["classes"]
        ]
        model.finalize()
        return model

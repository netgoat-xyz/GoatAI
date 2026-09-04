from __future__ import annotations

import csv
import random
import tempfile
import time
import unittest
from pathlib import Path

from model import Koda3Classifier
from make_test_set import generate
from train import evaluate_dataset, train_model


FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Flow IAT Mean",
    "Fwd Flag Count",
]


def write_dataset(path: Path, rows: int = 20_000) -> None:
    rng = random.Random(42)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*FEATURES, "Label"])
        for index in range(rows):
            if index % 2 == 0:
                fwd = rng.randint(10, 99)
                writer.writerow([
                    rng.randint(50_000, 60_000_000), fwd, fwd + rng.randint(5, 49),
                    abs(rng.gauss(500, 200)), abs(rng.gauss(100_000, 50_000)),
                    int(rng.random() >= 0.95), 0,
                ])
            else:
                writer.writerow([
                    rng.randint(100, 9_999), rng.randint(500, 49_999), rng.randint(0, 4),
                    abs(rng.gauss(1200, 10)), rng.expovariate(1 / 100),
                    int(rng.random() >= 0.1), 1,
                ])


class Koda3Tests(unittest.TestCase):
    def test_training_accuracy_size_and_inference_speed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset.csv"
            artifact = root / "koda-3.pkl"
            write_dataset(data)
            report = train_model(data, artifact)
            self.assertGreaterEqual(report["validation"]["accuracy"], 0.99)
            self.assertLess(artifact.stat().st_size, 10_000)

            model = Koda3Classifier.load(artifact)
            sample = [5000, 1500, 1, 1200, 50, 1]
            started = time.perf_counter()
            for _ in range(20_000):
                model.predict(sample)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertEqual(model.predict(sample), 1)

    def test_dedicated_test_set_is_external_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_data = root / "train.csv"
            first_test = root / "test-a.csv"
            second_test = root / "test-b.csv"
            artifact = root / "koda-3.pkl"
            write_dataset(training_data, 5_000)
            generate(first_test, rows=1_000)
            generate(second_test, rows=1_000)
            self.assertEqual(first_test.read_bytes(), second_test.read_bytes())
            self.assertNotEqual(training_data.read_bytes(), first_test.read_bytes())

            report = train_model(training_data, artifact, test_data_path=first_test)
            self.assertEqual(report["test"]["rows"], 1_000)
            self.assertLessEqual(report["test"]["accuracy"], report["validation"]["accuracy"])
            model = Koda3Classifier.load(artifact)
            self.assertEqual(evaluate_dataset(model, first_test)["rows"], 1_000)


if __name__ == "__main__":
    unittest.main()

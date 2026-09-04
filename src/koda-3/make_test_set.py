"""Build the frozen, distribution-shifted Koda-3 challenge test set."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Flow IAT Mean",
    "Fwd Flag Count",
]


def clipped_gauss(rng: random.Random, mean: float, deviation: float, minimum: float = 0.0) -> float:
    return max(minimum, rng.gauss(mean, deviation))


def benign_flow(rng: random.Random, subtype: int) -> list[float | int]:
    if subtype == 0:  # Ordinary bidirectional traffic with shifted means.
        forward = rng.randint(8, 180)
        return [
            rng.randint(40_000, 75_000_000),
            forward,
            max(1, forward + rng.randint(-15, 80)),
            clipped_gauss(rng, 620, 280),
            clipped_gauss(rng, 125_000, 70_000),
            int(rng.random() < 0.08),
        ]
    # Legitimate burst traffic deliberately overlaps packet-count attack features.
    forward = rng.randint(180, 2_000)
    return [
        rng.randint(15_000, 8_000_000),
        forward,
        max(20, int(forward * rng.uniform(0.45, 1.3))),
        clipped_gauss(rng, 780, 240),
        clipped_gauss(rng, 16_000, 12_000),
        int(rng.random() < 0.12),
    ]


def attack_flow(rng: random.Random, subtype: int) -> list[float | int]:
    if subtype == 0:  # Volumetric flood, shifted from the training generator.
        return [
            rng.randint(80, 30_000),
            rng.randint(350, 65_000),
            rng.randint(0, 12),
            clipped_gauss(rng, 1_080, 170),
            rng.expovariate(1 / 180),
            int(rng.random() < 0.82),
        ]
    if subtype == 1:  # Smaller SYN flood.
        return [
            rng.randint(2_000, 2_000_000),
            rng.randint(120, 6_000),
            rng.randint(0, 35),
            clipped_gauss(rng, 520, 260),
            rng.uniform(100, 12_000),
            1,
        ]
    # Low-rate attack: intentionally difficult for a model trained only on floods.
    return [
        rng.randint(1_000_000, 45_000_000),
        rng.randint(90, 900),
        rng.randint(0, 28),
        clipped_gauss(rng, 650, 260),
        rng.uniform(2_000, 80_000),
        int(rng.random() < 0.55),
    ]


def generate(path: Path, rows: int = 10_000, seed: int = 20260903) -> None:
    if rows < 20:
        raise ValueError("The challenge set requires at least 20 rows")
    rng = random.Random(seed)
    samples: list[list[float | int]] = []
    for index in range(rows // 2):
        samples.append([*benign_flow(rng, int(index % 5 == 0)), 0])
        samples.append([*attack_flow(rng, index % 3), 1])
    if rows % 2:
        samples.append([*benign_flow(rng, 0), 0])
    rng.shuffle(samples)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*FEATURES, "Label"])
        writer.writerows(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen Koda-3 challenge test set.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "koda-3-test.csv")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    generate(args.output, args.rows, args.seed)
    print(f"Wrote {args.rows:,} challenge rows to {args.output}")


if __name__ == "__main__":
    main()


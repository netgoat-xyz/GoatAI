"""Low-overhead Koda-3 command-line inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model import Koda3Classifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one flow with Koda-3.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "koda-3.pkl")
    parser.add_argument("--input", help="JSON object of feature names to numeric values; reads stdin when omitted.")
    args = parser.parse_args()

    model = Koda3Classifier.load(args.model)
    payload = json.loads(args.input if args.input is not None else sys.stdin.read())
    values = [float(payload[name]) for name in model.feature_names]
    probability = model.predict_probability(values)
    print(json.dumps({
        "label": "attack" if probability >= model.threshold else "benign",
        "attack_probability": probability,
        "threshold": model.threshold,
    }))


if __name__ == "__main__":
    main()

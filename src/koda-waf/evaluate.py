"""Evaluate Koda-WAF against a frozen JSONL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import KodaWAF, RequestView

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate(test_path: Path, model_path: Path) -> dict:
    engine = KodaWAF.from_model(model_path)
    tp = tn = fp = fn = 0
    failures = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        decision = engine.inspect(RequestView(
            method=item.get("method", "GET"),
            path=item.get("path", "/"),
            query=item.get("query", ""),
            headers=item.get("headers", ""),
            body=item.get("body", ""),
        ))
        actual = item["label"] == "malicious"
        if actual and decision.blocked:
            tp += 1
        elif not actual and not decision.blocked:
            tn += 1
        elif actual:
            fn += 1
        else:
            fp += 1
        if actual != decision.blocked:
            failures.append({"id": item["id"], "label": item["label"], "reasons": decision.reasons})
    total = tp + tn + fp + fn
    return {
        "rows": total,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Koda-WAF on a frozen JSONL corpus.")
    parser.add_argument("--test-data", type=Path, default=PROJECT_ROOT / "data" / "koda-waf-test.jsonl")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "koda-waf.pkl")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.test_data, args.model), indent=2))


if __name__ == "__main__":
    main()

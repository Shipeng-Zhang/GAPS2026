#!/usr/bin/env python3
"""Reproduce the direct-application bias experiment from Eqs. (19)--(21)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    counts = [1, 2, 4, 8, 16, 32, 64, 128]
    mu, sigma = 0.46, 0.24
    report = {
        "distribution": {"name": "normal", "mean": mu, "stddev": sigma},
        "realizations": args.realizations,
        "smooth_square": [],
        "discontinuous_step": [],
    }
    for count in counts:
        means = rng.normal(mu, sigma / np.sqrt(count), size=args.realizations)
        square = means**2
        measured_bias = float(square.mean() - mu**2)
        predicted_bias = sigma**2 / count
        report["smooth_square"].append({
            "inner_samples": count,
            "measured_bias": measured_bias,
            "equation_21_bias": predicted_bias,
            "absolute_error": abs(measured_bias - predicted_bias),
        })
        # A step models a two-band cel function around its boundary. Its
        # derivative does not exist, so Eq. (21) is reported as inapplicable.
        step_estimates = (means >= 0.5).astype(np.float64)
        true_step = float(mu >= 0.5)
        report["discontinuous_step"].append({
            "inner_samples": count,
            "expected_estimate": float(step_estimates.mean()),
            "true_value": true_step,
            "bias": float(step_estimates.mean() - true_step),
            "equation_21_bias": None,
        })
    output = ROOT / "outputs/bias_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


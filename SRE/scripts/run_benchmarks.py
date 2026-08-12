#!/usr/bin/env python3
"""Render the paper's estimator ablations on the SRE Cornell box."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mitsuba as mi


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spp", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("cuda_ad_rgb", "scalar_rgb"),
        default="cuda_ad_rgb",
    )
    args = parser.parse_args()
    mi.set_variant(args.variant)
    from sre.render import render_scene

    names = ["identity", "cel_1", "cel_8", "cel_32", "cel_polynomial",
             "gamma_unbiased", "telescoping_saturation", "combined_gu",
             "crosshatch", "recursive_saturation"]
    if args.quick:
        names = ["identity", "cel_1", "cel_8", "gamma_unbiased", "crosshatch"]
    reports = []
    for name in names:
        print(f"Rendering {name}...", flush=True)
        reports.append(render_scene(
            ROOT / "scenes/sre_cbox.xml",
            ROOT / f"configs/{name}.json",
            ROOT / f"outputs/{name}.exr",
            spp=args.spp,
            resolution=args.resolution,
            max_depth=args.max_depth,
            seed=args.seed,
        ))
    report_path = ROOT / "outputs/benchmark_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

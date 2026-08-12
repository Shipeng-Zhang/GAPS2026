#!/usr/bin/env python3
"""Validate identity SRE against an analytic Lambertian environment result."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mitsuba as mi
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render(integrator: str, spp: int, seed: int):
    scene = mi.load_file(
        str(ROOT / "scenes/validation_environment.xml"),
        integrator=integrator,
        max_depth="2",
        spp=str(spp),
    )
    start = time.perf_counter()
    image = np.asarray(mi.render(scene, spp=spp, seed=seed))[0, 0, :3]
    return image, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spp", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--variant",
        choices=("cuda_ad_rgb", "scalar_rgb"),
        default="cuda_ad_rgb",
    )
    args = parser.parse_args()
    mi.set_variant(args.variant)
    if args.variant == "cuda_ad_rgb":
        from sre.cuda_integrator import register_cuda_integrator
        register_cuda_integrator()
        integrator = "sre_cuda"
    else:
        from sre.integrator import register_sre_integrator
        register_sre_integrator()
        integrator = "sre"
    sre_image, sre_seconds = render(integrator, args.spp, args.seed)
    path_image, path_seconds = render("path", args.spp, args.seed)
    expected = np.repeat(0.5, 3)
    report = {
        "spp": args.spp,
        "variant": args.variant,
        "analytic_rgb": expected.tolist(),
        "sre_rgb": sre_image.tolist(),
        "path_rgb": path_image.tolist(),
        "sre_absolute_error": np.abs(sre_image - expected).tolist(),
        "path_absolute_error": np.abs(path_image - expected).tolist(),
        "sre_seconds": sre_seconds,
        "path_seconds": path_seconds,
    }
    output = ROOT / "outputs/identity_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if np.max(np.abs(sre_image - expected)) > 0.03:
        raise SystemExit("SRE identity validation exceeded 3% absolute error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

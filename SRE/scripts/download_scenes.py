#!/usr/bin/env python3
"""Re-download the minimal official Mitsuba Cornell Box asset set."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-data/master/scenes/cbox"
FILES = {
    "cbox-rgb.xml": 299,
    "cbox-spectral.xml": 317,
    "cbox.xml": 6114,
    "fragments/base.xml": 1135,
    "fragments/bsdfs-rgb.xml": 624,
    "fragments/bsdfs-spectral.xml": 3852,
    "fragments/bsdfs-white.xml": 515,
    "fragments/shapes.xml": 1190,
    "meshes/cbox_back.obj": 82,
    "meshes/cbox_ceiling.obj": 172,
    "meshes/cbox_floor.obj": 272,
    "meshes/cbox_greenwall.obj": 74,
    "meshes/cbox_largebox.obj": 1414,
    "meshes/cbox_luminaire.obj": 90,
    "meshes/cbox_redwall.obj": 82,
    "meshes/cbox_smallbox.obj": 1402,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "scenes/official_cbox"
    for relative, expected_size in FILES.items():
        target = destination / relative
        if target.exists() and target.stat().st_size == expected_size and not args.force:
            print(f"verified {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{relative}", timeout=60) as response:
            content = response.read()
        if len(content) != expected_size:
            raise RuntimeError(
                f"Unexpected size for {relative}: {len(content)} != {expected_size}"
            )
        target.write_bytes(content)
        print(f"downloaded {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


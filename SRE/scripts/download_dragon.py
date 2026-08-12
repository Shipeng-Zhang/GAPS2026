#!/usr/bin/env python3
"""Download and safely extract the official Stanford Dragon reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = "https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"
SHA256 = "74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f"
MESH_SHA256 = "fea87ff48f2aba22fb53e7b67c3ff3f7b8c2a3b3a0653af62c48bba67c6d5744"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "scenes/stanford_dragon"
    archive = destination / "source/dragon_recon.tar.gz"
    mesh = destination / "dragon_recon/dragon_vrip.ply"
    destination.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    if args.force or not archive.exists() or digest(archive) != SHA256:
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    actual = digest(archive)
    if actual != SHA256:
        raise RuntimeError(f"Dragon archive SHA-256 mismatch: {actual}")
    print(f"verified {archive.relative_to(ROOT)}")

    if args.force or not mesh.exists() or digest(mesh) != MESH_SHA256:
        safe_extract(archive, destination)
        print(f"extracted {mesh.relative_to(ROOT)}")
    actual_mesh = digest(mesh)
    if actual_mesh != MESH_SHA256:
        raise RuntimeError(f"Dragon mesh SHA-256 mismatch: {actual_mesh}")
    print(f"verified {mesh.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

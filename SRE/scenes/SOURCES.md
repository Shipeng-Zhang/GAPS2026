# Scene Asset Sources

The `official_cbox/` directory is the minimal Cornell Box data set downloaded
from the official Mitsuba data repository on 2026-07-23:

- Repository: `https://github.com/mitsuba-renderer/mitsuba-data`
- Source path: `scenes/cbox/`
- Branch: `master`
- Recursive tree used for discovery: `17a3c030d6f4be117612e584aa36c8ba42561786`

`scripts/download_scenes.py` contains the exact raw URLs and expected byte sizes
for all 16 files. It verifies existing files and downloads only missing or
size-mismatched assets. The custom `sre_cbox.xml` reuses the official geometry,
adds stable material and shape IDs for SRE configuration, and adds a mirror ball
to exercise path-dependent material stylization.

The paper's authored production scenes and supplemental package were not found
in public GitHub repository search by exact title or DOI (`10.1145/3658161`).
The included scene is therefore a reproducible public benchmark, not a claim
that the original proprietary scene package was recovered.

## Stanford Dragon

The paper explicitly credits the Stanford Computer Graphics Laboratory for the
dragon used in Figs. 7--12. The official VRIP reconstruction is stored under
`stanford_dragon/`; its URL, checksums, mesh variants, research-use terms, and
the distinction between the official mesh and our paper-inspired Mitsuba scene
are recorded in `stanford_dragon/README.md`. Run
`scripts/download_dragon.py` to verify or restore these files.

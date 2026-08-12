# Stanford Dragon Assets

The paper credits the Stanford Computer Graphics Laboratory for the dragon in
Figs. 7--12. This directory contains the official VRIP reconstruction archive,
downloaded from the Stanford 3D Scanning Repository on 2026-07-25.

- Repository page: `https://graphics.stanford.edu/data/3Dscanrep/`
- Archive: `https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz`
- Archive SHA-256: `74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f`
- Full mesh: `dragon_recon/dragon_vrip.ply`
- Full mesh SHA-256: `fea87ff48f2aba22fb53e7b67c3ff3f7b8c2a3b3a0653af62c48bba67c6d5744`
- Full mesh contents: 437,645 vertices and 871,414 triangular faces

`dragon_vrip_res2.ply`, `dragon_vrip_res3.ply`, and
`dragon_vrip_res4.ply` are progressively smaller meshes supplied in the same
official archive. Stanford notes that these were produced by an old decimator
that does not necessarily preserve topology. Use `dragon_vrip.ply` for final
rendering and a reduced mesh only for interactive scene setup.

The Stanford repository permits research use and free redistribution with
credit to the Stanford Computer Graphics Laboratory, and disallows commercial
product use without permission. It also asks users to treat the dragon as a
symbol of Chinese culture and avoid inappropriate destructive depictions. See
the repository page for the complete terms and context.

The paper says that implementation and standardized-scene details are in its
supplemental material, but no author scene/code package was publicly available
through ACM metadata, OpenAlex, Unpaywall, Semantic Scholar, or the authors'
public GitHub accounts when checked. `../sre_dragon.xml` is therefore a
paper-inspired reproducible Mitsuba scene, not the authors' original scene.

From the repository root, render a quick geometry preview with the CPU
reference path:

```bash
PYTHON=/home/shiwuxuan/anaconda3/envs/gaps/bin/python
$PYTHON -m sre.render --scene scenes/sre_dragon.xml \
  --config configs/identity.json --output outputs/dragon_identity.png \
  --spp 1 --resolution 64 --max-depth 3 --variant scalar_rgb
```

Use `configs/dragon_cel_8.json` and `--variant cuda_ad_rgb` for the material-level
SRE starter setup. The complete official mesh is ASCII PLY and Mitsuba will
print a harmless slow-parsing warning while loading it.

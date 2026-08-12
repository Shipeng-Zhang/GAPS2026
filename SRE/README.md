# Stylized Rendering as a Function of Expectation

This directory contains an independent Mitsuba 3 reproduction of West and
Mukherjee, *Stylized Rendering as a Function of Expectation* (SIGGRAPH 2024,
DOI `10.1145/3658161`). It implements the recursive estimator, not an image-space
filter: every styled material evaluates a function of independently estimated
outgoing radiance, and styled vertices recursively form the tree in Algorithm 1.
The authors' official [supplemental material](Stylized%20Rendering%20as%20a%20Function%20of%20Expectation%20Supplemental.pdf)
is included locally; the Fig. 8 Cel preset follows its Section S5.2 parameters.

## Implemented Core

- Material-level SRE through Mitsuba BSDF IDs, with shape-level overrides.
- Recursive Eq. (23) tree sampling and depth/occurrence/path parameterization.
- Unbiased U-statistic recurrence for polynomial powers.
- Random-prefix power-series estimation (Eqs. 13 and 24).
- Random-prefix telescoping estimation (Eq. 15).
- Chebyshev finite polynomial approximation (Eq. 17).
- Direct application with controllable inner sample count (Eq. 19).
- Full RGB delta-method bias diagnostic based on Eq. (21), plus optional delta
  and jackknife bias correction for smooth functions.
- Group-unbiased addition, independent multiplication, and nested composition
  from Theorem 4.1.
- Color maps, cel shading, saturation, gamma, cross-hatching, halftone, Gooch,
  and cosine tie-dye functions.
- BSDF/emitter balance-MIS for each unbiased child-integral sample.
- Per-pixel AOVs: tree nodes, style evaluations, inner variance, and estimated
  smooth-function bias.
- Lifting Lines and Tone feature lines: image-space path parametrization,
  stratified auxiliary stencils, event-preserving half-vector transport,
  Levi--Civita parallel transport, curvature-aware finite differences, and
  stochastic multi-line composition under CPU and CUDA DFS traversal.
- Compact CUDA `AuxiliaryPathFrame` storage and live-lane-first continuation:
  invalid camera/path lanes are compressed before the next `N_aux` edges are
  traced, and auxiliary intersections retain only the fields required by the
  line estimator.

## Environment

The supplied environment is used directly because this machine's older Conda
cannot create `conda run` temporary files inside the read-only environment:

```bash
cd /data1/swx/Project/GAPS2026/SRE
PYTHON=/home/shiwuxuan/anaconda3/envs/gaps/bin/python
$PYTHON -c "import mitsuba; print(mitsuba.__version__)"
```

The tested versions are Python 3.10.20, Mitsuba 3.9.0, and Dr.Jit 1.4.0.
`cuda_ad_rgb` is the default renderer and uses Dr.Jit arrays for ray validity,
BSDF/PDF masks, recursive child batches, estimator algebra, style evaluation,
and AOV accumulation. `scalar_rgb` remains the compact NumPy reference and CPU
fallback. The launcher selects the Conda LLVM 15 runtime for both CUDA workers
and final scalar image assembly; a system LLVM 12 installation is not used.

## Reproduce

Verify or re-download the official public scene assets:

```bash
$PYTHON scripts/download_scenes.py
$PYTHON scripts/download_dragon.py
```

The Stanford Dragon scene used for paper-style experiments is
`scenes/sre_dragon.xml`; its matching starter configuration is
`configs/dragon_cel_8.json`. Asset provenance and the limits of matching the
authors' unpublished standardized scene are documented in
[`scenes/stanford_dragon/README.md`](scenes/stanford_dragon/README.md).

Render one material style (the EXR includes named diagnostic AOVs):

```bash
$PYTHON -m sre.render --scene scenes/sre_cbox.xml \
  --config configs/cel_8.json \
  --output outputs/cel_8.exr \
  --spp 8 --resolution 64 --max-depth 5 --seed 23 \
  --variant cuda_ad_rgb
```

Use independent dimensions for rectangular output. `--resolution` remains the
backward-compatible square default and supplies either dimension that is not
specified explicitly:

```bash
$PYTHON -m sre.render --scene scenes/sre_dragon.xml \
  --config configs/identity.json --output outputs/dragon_1k.exr \
  --width 1024 --height 796 --spp 256 --variant cuda_ad_rgb
```

The report and image array use `[height, width, channels]`, so this command
produces shape `[796, 1024, 7]`. Multi-GPU crop rendering uses the same width
and height without requiring scene XML changes.

### Reference-image MSE

MSE is an image-level metric, not an SRE AOV. A style config can declare a PNG
reference through `metadata.mse_reference`; `dragon_cel.json` points to
`outputs/dragon_cel_base.png`. Single- and multi-GPU renders then append `mse`
immediately after the AOV statistics in the JSON report. Both PNGs are decoded
from sRGB to linear RGB before evaluating the mean squared error over all RGB
pixels. The rendered image and reference must have identical dimensions.

After changing a style formula or scene, regenerate `dragon_cel_base.png`
before treating that MSE as convergence error. In particular, a base image
created with the former four-color palette is not a valid reference for the
supplemental S5.2 two-band RGB-scaling formula.

Use `--mse-reference path/to/reference.png` to override the config reference,
or `--skip-mse` for quick renders at a different resolution. MSE should be
computed from images rendered with the same scene, camera, dimensions, depth,
and style function; only sampling budgets and independent seeds should differ.

For a fast first check, use `--spp 1 --resolution 8 --max-depth 3`. CUDA JIT
compilation is one-time work for each estimator/depth/array-shape graph. Highly
branched estimators such as `cel_polynomial` can therefore take minutes on their
first invocation even at low `spp`; an identical rerun uses Dr.Jit's disk cache
and takes well under a second on the tested GPU. Changing `spp`, resolution,
depth, or estimator structure can generate a new graph. This is compilation
latency, not scalar execution or a stalled render.

### CUDA depth-first evaluation and multiple GPUs

CUDA evaluates the recursive estimator depth first. It traces one inner child,
accumulates its contribution into a material-local online statistic, calls
`dr.eval()`, and only then traces the next child. Direct application uses an
online mean and covariance, polynomial estimators keep only their elementary
symmetric recurrence, and gamma/telescoping estimators keep only their running
recurrence. Before descending, the current child's valid transport lanes are
stream-compacted; its result and AOVs are scattered back after that child
finishes. Rays that miss geometry terminate without further recursion.
Consequently, increasing a direct estimator from 1 to 32 inner samples
increases work but no longer allocates a 32-times-wider child wavefront at every
styled depth. The live estimator state is `O(1)` for direct application and
`O(degree)` for polynomial estimators, in addition to the compacted current
path, scene BVH, and JIT state.

Outer pixel samples are independently bounded with `--spp-per-pass`. Its
lowest-memory setting is the default, one SPP per render pass. Each completed
pass is copied to a CPU `float64` accumulation buffer and the CUDA allocation
cache is flushed before the next pass. For example:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 $PYTHON ./sre/render.py \
  --scene scenes/sre_dragon.xml --config configs/dragon_cel_8.json \
  --output outputs/dragon_cel.exr --width 1024 --height 796 \
  --spp 256 --spp-per-pass 1 --max-depth 4
```

`--spp-per-pass` only splits independent outer pixel samples. Inner samples
cannot be split into separate styled estimates: the implementation always
forms `g(sum(children) / inner_sample)` once, never an average of independently
styled chunks.

For LLaT feature lines, the same rule also keeps the camera wavefront from
being multiplied by outer SPP. `N_aux=16` auxiliary paths and `n=16` pair
comparisons are still evaluated for every outer sample, so lowering
`--spp-per-pass` does not change the feature-line estimator or line width. Use
`--spp-per-pass 1` for minimum memory; values 2--4 trade proportionally more
memory for fewer CPU/GPU pass boundaries. Avoid 16 at 1K resolution unless the
GPU has enough space for sixteen complete camera wavefronts.

CUDA renders are additionally split into bounded spatial tiles. The default
`--max-wavefront-size 1536` limits `tile_pixels * spp_per_pass` rather than
letting one GPU expand its entire image strip at once. Every tile runs in a
fresh process and CUDA context. After a tile completes, process exit destroys
all Mitsuba C++ objects, OptiX state, Dr.Jit graphs, and allocator caches before
that GPU receives another tile. Different GPUs remain concurrent. This changes
neither `spp`, `N_aux`, line comparisons, nor path transport. For unusually
large auxiliary budgets, lower the bound:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON ./sre/render.py \
  --scene scenes/fig11.xml --config configs/f11_lines.json \
  --output outputs/f11_lines.exr --width 1920 --height 640 \
  --spp 256 --spp-per-pass 1 --max-depth 2 \
  --max-wavefront-size 1536
```

Smaller bounds trade scene loading/JIT time for lower peak host and device
memory. Nested SRE, feature-line, and tone configurations never permit an
unbounded value: passing `0` is automatically restored to the safe default.

The temporary OptiX graph for the `N_aux` feature-line intersections is also
bounded by `feature_lines.cuda_auxiliary_batch_size` in the style JSON. Its
default is `4`; use `1` for the lowest peak or `8` when memory is ample and
kernel-launch throughput is more important. This setting does not alter the
samples, stencil tests, line priority, or output estimator.

Render the LLaT Fig. 13 canonical image-space hatching benchmark with four
GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON ./sre/render.py \
  --scene scenes/fig13.xml --config configs/f13_tone.json \
  --output outputs/f13_tone.png --width 960 --height 540 \
  --spp 256 --spp-per-pass 1 --max-depth 4 \
  --max-wavefront-size 1536
```

Use `configs/f13_halftone.json` with the same command for the halftone field.
Both presets use material-independent mirror anchors and local linear MLS, so
the 2D pattern retains its film-space scale/orientation through the planar
mirror and is blurred by the curved GGX reflector through transport rather
than by reparameterizing the dots or strokes.

When `CUDA_VISIBLE_DEVICES` names one or more devices, `sre.render` assigns
bounded crops dynamically. Each GPU slot runs at most one fresh tile process
at a time and receives another tile only after that process has exited:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 $PYTHON ./sre/render.py \
  --scene scenes/sre_dragon.xml \
  --config configs/dragon_crosshatch.json \
  --output outputs/dragon_crosshatch.exr \
  --spp 16 --resolution 512 --max-depth 5
```

Each crop keeps the SRE inner-sample count specified by `samples` in the JSON
configuration (`--spp` is the separate outer pixel-sample count), and the
parent process assembles one PNG or named seven-channel EXR. This divides
camera-wavefront and recursive dynamic memory approximately by the number of
GPUs. Each GPU must still hold its own scene/BVH and JIT state. Crop workers
receive decorrelated, deterministic seeds, so a multi-GPU result is
statistically equivalent but not bit-identical to a single-GPU render with the
same base seed. Tiny renders can be slower because every process pays scene
loading/JIT startup costs; the speedup applies to production-sized renders.

To force the original CPU path:

```bash
$PYTHON -m sre.render --spp 1 --resolution 8 --max-depth 3 \
  --variant scalar_rgb --output outputs/cpu_smoke.png
```

Run the estimator suite and analytic transport validation:

```bash
$PYTHON scripts/run_tests.py
$PYTHON scripts/validate_identity.py --spp 8192 --variant cuda_ad_rgb
$PYTHON scripts/analyze_bias.py --realizations 100000
$PYTHON scripts/run_benchmarks.py --spp 4 --resolution 64 --max-depth 5 \
  --seed 23 --variant cuda_ad_rgb
```

Use `--variant scalar_rgb` with either rendering script to exercise the CPU
reference. The optional renderer smoke test accepts the same selection through
`SRE_VARIANT=cuda_ad_rgb` (and is enabled with `SRE_MITSUBA_TEST=1`).

`outputs/benchmark_report.json`, `outputs/identity_validation.json`, and
`outputs/bias_analysis.json` contain machine-readable results. See
[`configs/README.md`](configs/README.md) for the material schema and
[`scenes/SOURCES.md`](scenes/SOURCES.md) for asset provenance.

## Bias and Expectation

For smooth functions, the direct estimator records the second-order bias
`0.5 * trace(H_g Cov[mean])`. `bias_correction: delta` subtracts this estimate;
`jackknife` removes the leading finite-sample bias without derivatives. These are
optional extensions: `bias_correction: none` is exactly Eq. (19).

The CUDA `none` and `delta` modes are fully streamed. Jackknife mathematically
needs all leave-one-out values and currently retains its inner samples as an
explicit higher-memory fallback; use `delta` when a smooth function needs bias
correction under a strict memory budget.

Discontinuous functions such as cel and hatching do not satisfy Eq. (21)'s
smoothness assumption. Their reliable paper-aligned controls are increasing the
inner sample count (`cel_1`, `cel_8`, `cel_32`) or using a finite polynomial
approximation (`cel_polynomial`). The numerical bias AOV can spike near a step
and must not be interpreted as a valid Taylor prediction there.

Power-series and telescoping estimators are unbiased under their convergence
conditions but can have high variance. `gamma_unbiased.json` uses an independent
pilot expansion point, random prefix, minimum expansion point, and 8x recurrence
oversampling, corresponding to the variance-reduction progression in Section 5.3.

## Scope

The formulation covers local functions of exitant radiance, conditional
feature-line lifting, and canonical tone lifting with locally approximated
inverse coordinates. Feature-line extraction remains stochastic
and does not produce editable vector strokes. Photon-mapping FTV comparison and
the paper's acknowledged open problems (unbiased discontinuous functions,
reciprocity/path-integral formulation, and general infinite recursive existence)
are not presented as solved. The original production scene package was not
publicly discoverable; the official Mitsuba Cornell Box and a custom mirror
extension provide the checked-in reproducible benchmark instead.

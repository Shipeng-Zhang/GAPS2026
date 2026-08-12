# Paper-to-Code Map

| Paper item | Implementation |
|---|---|
| Eq. (6), recursive `g_theta` | `sre/integrator.py::_radiance`, `sre/cuda_integrator.py::_radiance` |
| Eq. (9), group-unbiased contract | `Estimator.estimate(draw, rng, context, stats)` |
| Eq. (13), random power prefix | `GammaPowerSeriesEstimator` |
| Eq. (15), telescoping prefix | `TelescopingEstimator` |
| Eq. (17), finite approximation | `fit_polynomial`, `PolynomialEstimator` |
| Eq. (19), direct application | `DirectApplicationEstimator` |
| Eq. (21), local bias | `delta_method_bias` |
| Eqs. (22)-(23), complete estimator | `SREIntegrator.sample`, `CudaSREIntegrator.sample`, `_radiance` |
| Eq. (24), gamma coefficients | `GammaPowerSeriesEstimator.estimate` |
| Algorithm 1 | Estimator-controlled calls to recursive `sample_integrand` |
| Theorem 4.1 addition | `AdditionEstimator` with sequential independent draws |
| Theorem 4.1 multiplication | `MultiplicationEstimator` with disjoint draws |
| Theorem 4.1 composition | `CompositionEstimator` with nested independent estimates |

The NumPy/scalar classes are the readable reference implementation. The CUDA
variant dispatches the same estimator and style objects through
`cuda_backend.py`, while `cuda_integrator.py` carries per-lane Dr.Jit masks and
evaluates independent recursive children depth first.

## One Child Integral Sample

At a surface, `sample_integrand` estimates the quantity inside `g_theta`:

1. Evaluate emitted radiance at the current vertex.
2. Select BSDF or emitter-direction sampling with probability 1/2 when the BSDF
   has a smooth component; delta-only BSDFs use their own sampler.
3. Evaluate the balance-mixture PDF and BSDF-cosine numerator.
4. Trace the selected direction and recursively evaluate the next vertex SRE.
5. Add emission and weighted recursive transport.

The local estimator decides how many times this callback is invoked. Thus eight
inner samples create eight independent recursive children, rather than reusing a
single path or applying a style after rendering.

## CUDA Memory Schedule

Each invocation of the callback completes one recursive child before the next
invocation starts. `WavefrontEstimator._materialize` evaluates the online
accumulator and diagnostic AOV state after every child, cutting references to
the completed Dr.Jit graph. No list of child rays, intersections, or recursive
radiance values is retained. This is the depth-first schedule described around
Algorithm 1 and replaces the previous breadth-first concatenated wavefront.
Before descending one level, the implementation compresses the current child's
valid transport mask and seeds a compact sampler. The returned radiance and AOV
state are scattered back to the parent wavefront after the child completes.

Direct estimators use Welford updates for the RGB mean and covariance, so their
state does not depend on the inner sample count. Polynomial and random-prefix
estimators update their recurrence in place and retain `O(degree)` values.
At the image level, `render.py` bounds the camera wavefront using independent
`--spp-per-pass` passes and accumulates their weighted results on the CPU.
For CUDA, it also bounds the spatial wavefront with
`--max-wavefront-size`: each device processes several crops sequentially, so
peak memory is proportional to one crop instead of the complete image strip.
The feature-line auxiliary frame requests and stores only fields required by
the active dictionaries at that path depth. In particular, BSDF pointers,
positions, shape pointers, and shading frames are omitted when their metrics
and later parallel transport do not consume them.
CUDA auxiliary intersections are materialized in bounded groups controlled by
`feature_lines.cuda_auxiliary_batch_size` (default `4`). This prevents the
temporary OptiX/Dr.Jit graph for all `N_aux` intersections from being live at
once. Set it to `1` for the lowest peak memory, or increase it toward `N_aux`
when launch overhead matters more than memory; the rendered estimator is
unchanged.

## Canonical Tone Lifting and Local Inversion

`tone_mapping.py` and `cuda_tone_mapping.py` implement LLaT Section 5 and
Supplemental S2. A camera sample creates a deterministic multi-scale set of
film-space anchors. Every anchor is traced with perfect reflection about the
macro geometric normal at every surface, regardless of its actual BSDF. The
stored correspondence at one DFS level consists only of film coordinate,
incoming direction, position, geometric normal, shape prefix, and validity.

At an actual path vertex, candidates must match the current shape prefix and
an incoming-direction angular bound. They are projected into the plane through
the query normal to its incoming ray. An affine image coordinate is evaluated
with adaptive-Gaussian linear moving least squares. Centered normal equations
reduce the solve to 2x2 while remaining algebraically equivalent to
Supplemental S2.5's 3x3 affine basis. Sparse/degenerate neighborhoods use the
paper's affine, nearest, and first-edge fallback hierarchy.

The canonical child frame is independent of every stochastic BSDF draw, so it
is built once per SRE DFS vertex and reused by all inner samples. Compacted
recursive lanes gather the tone frame alongside their ray and sampler state.
BVH queries are evaluated in `cuda_anchor_batch_size` groups. Peak additional
device state is therefore `O(anchor_samples * DFS_depth)` and does not grow
with the estimator's `inner_sample` count or total image resolution.

## Minimum-Variance Polynomial Terms

For `n` independent samples, `unbiased_powers` updates elementary symmetric
polynomials in descending degree and divides degree `k` by `choose(n,k)`. The
result averages all products of `k` distinct samples, providing an unbiased
estimate of `I^k` while using every sample combination. Luminance projection is
linear, so the same construction remains unbiased for scalar-to-RGB color maps.

## Material and Path State

The integrator reads `si.bsdf().id()` at every vertex. A material binding is
therefore shared by every shape referencing that BSDF, while `shapes` can override
one instance. Each recursive branch carries its own immutable occurrence map;
`first_hit` consequently means first occurrence on that branch. This implements
the mirror/depth examples in Section 5.2 without screen-space buffers.

## Energy and Termination

`max_depth` bounds stylization recursion and guarantees that deeper evaluation
reduces to a terminal emitted-radiance estimate. Russian roulette begins at
`rr_depth` and divides surviving transport by the continuation probability.
Energy-increasing styles can still generate high variance or bright results, as
the paper notes; no silent radiance clamp is applied to the core estimator.

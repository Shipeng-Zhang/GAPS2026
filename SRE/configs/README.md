# Configuration Reference

Styles bind to Mitsuba BSDF IDs through `materials`, or to individual shape IDs
through the higher-priority `shapes` map. Unlisted surfaces use identity SRE.

Each binding selects an `estimator`:

| Estimator | Paper correspondence | Important fields |
|---|---|---|
| `identity` | Rendering equation special case | none |
| `constant` | Deterministic local paper/ink base `g(L)=c` | `value` or `color` |
| `direct` | Eq. (19), biased/consistent locally | `samples`, `function`, `bias_correction` |
| `polynomial` | Eq. (17), unbiased polynomial terms | `degree`, `sample_count`, `fit_interval`, `normalized_domain` |
| `power_series_gamma` | Eqs. (13), (24) | `continuation_probability`, `pilot_samples`, `oversampling` |
| `telescoping` | Eq. (15) | `base_samples`, `continuation_probability`, `function` |
| `addition` | Theorem 4.1 | `left`, `right` |
| `multiplication` | Theorem 4.1 | `left`, `right` |
| `composition` | Theorem 4.1 | `outer`, `inner` |

Available style functions are `identity`, `gamma`, `saturation`, `color_map`,
`color_map_nonlinear`, `cel`, `crosshatch`, `halftone`, `gooch`, and
`tie_dye`. `color_map` linearly interpolates each gradient segment;
`color_map_nonlinear` applies a cubic smoothstep transfer within each segment.

`cel` supports nonuniform normalized `thresholds`, explicit `band_values`, and
the `luminance` or `mean` `brightness_mode`. Supplemental Section S5.2 defines
the Fig. 8 GI style using mean RGB brightness, two target bands, and RGB scaling
instead of a fixed palette: `u=(R+G+B)/3`, `m(u)=0.4` below `0.75` and `0.95`
otherwise, followed by `RGB'=RGB*m(u)/u`. This naturally retains indirect-light
color bleed. `dragon_cel.json` implements that formula exactly and uses the
paper's highest plotted budget of 128 inner samples; the paper's reference used
4096 inner samples per evaluation. A `palette` and the older optional
`chroma_strength` controls remain available for non-paper presets.

`crosshatch` treats each entry in `directions` as the normal of a family of
parallel object-space slice planes. The vectors are normalized before use, so
their length cannot accidentally change stroke spacing. Corresponding entries
in `activation_thresholds`, `phase_offsets`, `scale_factors`, and
`family_widths` control when each darker-tone layer appears and prevent all
families from forming one uniform grid. `width` and `edge_softness` are measured
as fractions of a stroke period; `width_growth` thickens strokes in dark tones.
The Fig. 12 configuration uses the paper's direct application estimator because
the hatch activation boundaries are discontinuous.

`halftone` follows Fig. 12 by intersecting the surface with an oriented 3D
lattice of spheres instead of relying on mesh UVs. `scale`, `phase`, and
`orientation` place the global sphere grid. `min_radius`, `max_radius`, and
`radius_gamma` control how dot size grows with darkness; `dot_threshold` keeps
highlights clean. `min_ink_strength` makes small highlight dots slightly softer
while dark dots approach the configured `ink`, and `edge_softness` only
anti-aliases the sphere boundary. The direct estimator's inner variance still
produces the soft/hard tone-sphere edges described by the paper.

For high-degree polynomial fits, set `normalized_domain: true`. The fitter then
represents the polynomial in the normalized coordinate
`u = 2 * (x - low) / (high - low) - 1`, and the CPU/CUDA estimators apply the
same affine transform to every independent inner sample before the Eq. (17)
recurrence. This is the same polynomial estimator, but avoids the large power
basis coefficients and Float32 cancellation caused by evaluating degree-20
fits directly in the original interval. Following supplemental Section S5.5,
`dragon_tie_dye.json` fits the exact per-channel negative cosine waves with a
degree-20 Chebyshev polynomial on `[-1, 4]`, clamps out-of-range radiance
samples to that interval, and uses 32 inner samples per evaluation. Its
power-basis coefficients strongly cancel, so the CUDA recurrence uses
`evaluation_precision: float64`; this exchanges throughput for the color
stability of the paper's CPU implementation without increasing path storage.

The Stanford Dragon scene has ready-to-render material bindings in
`dragon_cel.json`, `dragon_linear_colormap.json`,
`dragon_nonlinear_colormap.json`, `dragon_crosshatch.json`,
`dragon_halftone.json`, and `dragon_tie_dye.json`. They target the `dragon` BSDF ID in
`scenes/sre_dragon.xml` and leave unlisted scene materials physically shaded.

The optional `when` object parameterizes the style by zero-based path vertex
depth and material occurrence. It supports `min_depth`, `max_depth`, `depths`,
`first_hit`, and `max_occurrences`. `first_hit` is per expansion branch, not a
screen-space approximation.

## Lifting Lines and Tone feature lines

An optional top-level `feature_lines` dictionary enables the conditional
lifting from Section 4 and Supplemental S1.4 of *Lifting Lines and Tone*.
`auxiliary_samples` image-offset paths are stratified over the largest active
line stencil and extended edge-by-edge with the base SRE path. Microfacet
half-vectors retain their tilt and are Levi--Civita transported in the paper's
view-oriented chart; diffuse events use the documented minimal-rotation
fallback. Auxiliary prefixes are stored only for the current DFS recursion,
so live state is `O(auxiliary_samples * max_depth)`.

Each entry in `types` accepts `measurement` (`depth`, `normal`, `curvature`,
`material_id`, `shape_id`, or `position`), `threshold`, `stencil` (`disk` or
`square`), `stencil_radius` in pixels, `comparisons`, `color`, and optional
`min_depth`/`max_depth`. Types are evaluated in listed priority order. A
finite-difference pair that crosses a visibility discontinuity always detects
a line; otherwise Eq. (21)'s slope is compared with the type's threshold.
The binary product estimator exits on the first successful search, matching
Eqs. (29)--(34).

`llat_feature_lines.json` uses the paper's published practical setting of 16
searches, a stratified disk stencil, and front-to-back DFS composition. The
supplement denotes the auxiliary-path count as `N_aux` without publishing a
single scene-independent value; the reproduction uses 16 auxiliary paths to
match the search budget. The paper likewise does not publish universal numeric
thresholds because they depend on the units of the chosen measurement field;
the preset therefore supplies normalized depth and angular-curvature
thresholds for the included LLaT scene.

`f10_lines.json` uses 16 auxiliary paths and an `n=32` finite-difference search
for the wider Fig. 10 line preset (Section 4.6 notes that wider lines can need
more searches than the usual 12--16). Pairs outside an individual line type's
stencil are ignored without consuming that type's search budget, as required
by Supplemental S1.4. Its depth, angular-curvature, and material-ID stencils
preserve the calibrated silhouette/interior-line hierarchy without treating
every Blender submesh as a semantic line. Diffuse surfaces use deterministic
paper white, while the glossy floor and smooth curved conductor remain
identity-bound so the four transport cases in Fig. 10 are produced by path
transport rather than image compositing.

## Lifting Lines and Tone: canonical tone fields

The optional top-level `tone_mapping` block implements Section 5 and
Supplemental S2. `anchor_samples` includes the central camera ray and controls
the number of compact perfect-mirror paths retained at one DFS level.
`min_radius`, `search_radius`, and `radial_rings` form the multi-scale camera
neighborhood. At 1920x1080 the paper uses a maximum 512 px search radius;
`f13_tone.json` uses the resolution-equivalent 256 px radius at 960x540.
`reference_width` makes both anchor radii and returned tone coordinates scale
with output resolution, so style spacing remains visually unchanged.

`min_candidates`, `sigma_scale`, `angular_limit_degrees`, and
`condition_epsilon` control the view-oriented linear MLS inverse. The runtime
falls back from MLS to a minimal affine fit, nearest neighbor, and finally the
originating first-edge coordinate. `cuda_anchor_batch_size` bounds the number
of unevaluated anchor BVH queries and changes memory/launch overhead only.

Use `tone_hatch` and `tone_halftone` to consume the reconstructed 2D
`StyleContext.tone_coordinate`. The existing `crosshatch` and `halftone`
functions remain world-space styles for the original SRE experiments.
`f13_tone.json` is the Fig. 13 line+hatch preset; `f13_halftone.json` uses the
same mapping with dots to validate the second tone family.

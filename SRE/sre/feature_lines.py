"""Feature-line lifting from image space into SRE path space.

This module contains renderer-independent configuration and geometry helpers
for Section 4 and Supplemental Section S1 of *Lifting Lines and Tone*.  The
Mitsuba-specific auxiliary-path implementations live next to the CPU and CUDA
integrators, while the equations below are shared and directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


RGB = np.ndarray


def _rgb(value: Sequence[float]) -> RGB:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("feature-line colors must contain three finite values")
    return result


def normalize(value: Sequence[float], epsilon: float = 1e-12) -> np.ndarray:
    """Return a numerically stable normalized 3-vector."""
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= epsilon:
        raise ValueError("cannot normalize a degenerate vector")
    return vector / length


def view_oriented_frame(
    normal: Sequence[float], view: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Construct Eqs. (17)--(18)'s view-oriented tangent frame."""
    normal = normalize(normal)
    view = normalize(view)
    tangent_1 = np.cross(normal, view)
    if np.linalg.norm(tangent_1) <= 1e-10:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(normal[0])) > 0.8:
            axis = np.array([0.0, 1.0, 0.0])
        tangent_1 = np.cross(normal, axis)
    tangent_1 = normalize(tangent_1)
    tangent_2 = normalize(np.cross(tangent_1, normal))
    return tangent_1, tangent_2


def parallel_transport_half_vector(
    half_vector: Sequence[float],
    base_normal: Sequence[float],
    auxiliary_normal: Sequence[float],
    base_view: Sequence[float],
    auxiliary_view: Sequence[float],
) -> np.ndarray:
    """Transport a microfacet realization using Eqs. (13)--(18).

    The cosine tilt and the azimuth in the incoming-direction-oriented chart
    are fixed.  Reconstructing the same components in the auxiliary chart is
    the discrete Levi--Civita transport used by the paper's implementation.
    """
    normal = normalize(base_normal)
    auxiliary_normal = normalize(auxiliary_normal)
    half_vector = normalize(half_vector)
    if float(np.dot(half_vector, normal)) < 0.0:
        half_vector = -half_vector
    cosine_tilt = float(np.clip(np.dot(half_vector, normal), -1.0, 1.0))
    sine_tilt = float(np.sqrt(max(0.0, 1.0 - cosine_tilt * cosine_tilt)))
    if sine_tilt <= 1e-10:
        return auxiliary_normal.copy()

    tangent = normalize(half_vector - cosine_tilt * normal)
    tangent_1, tangent_2 = view_oriented_frame(normal, base_view)
    angle = float(
        np.arctan2(np.dot(tangent, tangent_2), np.dot(tangent, tangent_1))
    )
    aux_tangent_1, aux_tangent_2 = view_oriented_frame(
        auxiliary_normal, auxiliary_view
    )
    transported = (
        np.cos(angle) * aux_tangent_1 + np.sin(angle) * aux_tangent_2
    )
    return normalize(
        cosine_tilt * auxiliary_normal + sine_tilt * transported
    )


def minimal_rotation(
    value: Sequence[float],
    source_normal: Sequence[float],
    target_normal: Sequence[float],
) -> np.ndarray:
    """Transport a direction by the minimal rotation between two normals."""
    value = normalize(value)
    source = normalize(source_normal)
    target = normalize(target_normal)
    axis = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return value.copy()
    if cosine < -1.0 + 1e-8:
        tangent_1, _ = view_oriented_frame(source, value)
        return normalize(2.0 * np.dot(value, tangent_1) * tangent_1 - value)
    rotated = (
        value
        + np.cross(axis, value)
        + np.cross(axis, np.cross(axis, value)) / (1.0 + cosine)
    )
    return normalize(rotated)


def finite_difference_slope(
    first: Sequence[float] | float,
    second: Sequence[float] | float,
    distance: float,
) -> float:
    """Two-point Lipschitz lower bound from Eq. (21)."""
    if not np.isfinite(distance) or distance <= 0.0:
        return 0.0
    difference = np.asarray(second, dtype=np.float64) - np.asarray(
        first, dtype=np.float64
    )
    return float(np.linalg.norm(difference.reshape(-1)) / distance)


def normal_finite_difference_slope(
    first: Sequence[float],
    second: Sequence[float],
    distance: float,
    orientation_invariant: bool = False,
) -> float:
    """Eq. (21) applied to a normalized surface-normal field.

    Separately exported mesh parts can encode the same tangent plane with
    opposite normal signs. ``orientation_invariant`` treats those signs as
    equivalent and prevents a false extremal gradient at the export seam.
    """
    if not np.isfinite(distance) or distance <= 0.0:
        return 0.0
    first_normal = normalize(first)
    second_normal = normalize(second)
    direct = float(np.linalg.norm(second_normal - first_normal))
    if orientation_invariant:
        direct = min(direct, float(np.linalg.norm(second_normal + first_normal)))
    return direct / distance


@dataclass(frozen=True)
class FeatureLineType:
    """One dictionary entry in the paper's multi-line composition."""

    name: str
    measurement: str = "normal"
    threshold: float = 0.2
    stencil_radius: float = 1.5
    comparisons: int = 16
    color: RGB = field(
        default_factory=lambda: np.array([0.015, 0.015, 0.015])
    )
    stencil: str = "disk"
    relative_depth: bool = True
    # Treat n and -n as the same tangent-plane orientation when measuring a
    # normal finite difference. Useful for Blender scenes assembled from many
    # independently exported meshes with inconsistent normal signs.
    normal_orientation_invariant: bool = False
    # Some independently exported face pieces are coplanar at a visible
    # decorative seam. Their geometric normals are identical, so a normal
    # finite difference cannot observe that seam. This opt-in supplement keeps
    # the normal metric primary and only closes such an explicitly configured
    # mesh boundary.
    normal_shape_boundary_fallback: bool = False
    # Internal-detail dictionaries can disable hit/miss silhouettes and leave
    # them to a later, wider outline dictionary. This preserves fine nested
    # rings and panel seams without thinning the object's outer contour.
    include_silhouette: bool = True
    min_depth: int = 0
    max_depth: int | None = None
    include_materials: tuple[str, ...] = ()
    exclude_materials: tuple[str, ...] = ()
    # Optional Mitsuba shape IDs restricting the dictionary to selected mesh
    # parts.  This is useful when Blender exports many overlapping PLYs that
    # share one BSDF (e.g. the face rings and side shell of Fig. 11 robot 1).
    include_shapes: tuple[str, ...] = ()
    depth_colors: tuple[tuple[int, RGB], ...] = ()
    # Optional per-dictionary override for the neutral glossy-path tint.  A
    # value of ``None`` preserves the global FeatureLineConfig setting.  This
    # is needed for Fig. 11's fourth robot, whose once-reflected strokes are
    # intentionally yellow rather than a desaturated version of its direct
    # pink strokes.
    glossy_mix_strength: float | None = None
    stencil_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    # Optional object-space modulation of the image-space stencil center.
    # This realizes the displaced/sketchy line style shown by the magenta
    # robot in Fig. 11 while preserving the same path-space edge estimator.
    # Each output component is
    #   amplitude[i] * sin(frequency[i] * dot(axis[i], p) + phase[i]).
    stencil_warp_amplitude: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    stencil_warp_frequency: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    stencil_warp_phase: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    stencil_warp_axes: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
    )
    # ``sin`` gives a regular displacement field. ``ripple`` keeps a strong
    # smooth fundamental and adds one weak harmonic, producing the broad
    # displaced/wavy contours of the magenta robot in Fig. 11. ``sketchy``
    # uses three incommensurate harmonics for a more irregular hand-drawn line.
    stencil_warp_profile: str = "sin"
    # Optional object-space dash/hatch modulation applied *after* a feature
    # has been detected. A zero scale preserves the ordinary solid line. This
    # is used by the fifth Fig. 11 robot: CrossHatch must texture only its
    # detected contours, not fill the complete material surface.
    line_hatch_scale: float = 0.0
    line_hatch_width: float = 0.1
    line_hatch_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.0], dtype=np.float64)
    )
    line_hatch_phase: float = 0.0
    line_hatch_edge_softness: float = 0.0

    def __post_init__(self) -> None:
        measurement = self.measurement.lower()
        aliases = {"normals": "normal", "material": "material_id", "shape": "shape_id"}
        measurement = aliases.get(measurement, measurement)
        allowed = {
            "depth", "normal", "curvature", "material_id", "shape_id",
            "position", "silhouette"
        }
        if measurement not in allowed:
            raise ValueError(
                f"unknown feature-line measurement {self.measurement!r}"
            )
        if self.stencil not in {"disk", "square"}:
            raise ValueError("feature-line stencil must be 'disk' or 'square'")
        warp_profile = str(self.stencil_warp_profile).lower()
        if warp_profile not in {"sin", "ripple", "sketchy"}:
            raise ValueError(
                "feature-line stencil_warp_profile must be 'sin', 'ripple', "
                "or 'sketchy'"
            )
        if not self.name:
            raise ValueError("feature-line types need a non-empty name")
        if not np.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("feature-line threshold must be finite and non-negative")
        if not np.isfinite(self.stencil_radius) or self.stencil_radius <= 0.0:
            raise ValueError("feature-line stencil_radius must be positive")
        if self.comparisons < 1:
            raise ValueError("feature-line comparisons must be positive")
        if self.min_depth < 0:
            raise ValueError("feature-line min_depth cannot be negative")
        if self.max_depth is not None and self.max_depth < self.min_depth:
            raise ValueError("feature-line max_depth cannot precede min_depth")
        materials = tuple(str(value) for value in self.include_materials)
        if any(not value for value in materials):
            raise ValueError("feature-line include_materials cannot contain empty IDs")
        excluded_materials = tuple(str(value) for value in self.exclude_materials)
        if any(not value for value in excluded_materials):
            raise ValueError("feature-line exclude_materials cannot contain empty IDs")
        if set(materials).intersection(excluded_materials):
            raise ValueError(
                "feature-line material filters cannot include and exclude the same ID"
            )
        shapes = tuple(str(value) for value in self.include_shapes)
        if any(not value for value in shapes):
            raise ValueError("feature-line include_shapes cannot contain empty IDs")
        raw_depth_colors = self.depth_colors
        if isinstance(raw_depth_colors, Mapping):
            raw_depth_colors = raw_depth_colors.items()
        parsed_depth_colors = tuple(
            (int(line_depth), _rgb(line_color))
            for line_depth, line_color in raw_depth_colors
        )
        if any(line_depth < 0 for line_depth, _ in parsed_depth_colors):
            raise ValueError("feature-line depth_colors cannot use negative depths")
        if len({line_depth for line_depth, _ in parsed_depth_colors}) != len(
            parsed_depth_colors
        ):
            raise ValueError("feature-line depth_colors cannot repeat a depth")
        glossy_mix_strength = self.glossy_mix_strength
        if glossy_mix_strength is not None:
            glossy_mix_strength = float(glossy_mix_strength)
            if (
                not np.isfinite(glossy_mix_strength)
                or not 0.0 <= glossy_mix_strength <= 1.0
            ):
                raise ValueError(
                    "feature-line glossy_mix_strength must lie in [0, 1]"
                )
        offset = np.asarray(self.stencil_offset, dtype=np.float64)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("feature-line stencil_offset must contain two finite values")
        warp_amplitude = np.asarray(
            self.stencil_warp_amplitude, dtype=np.float64
        )
        warp_frequency = np.asarray(
            self.stencil_warp_frequency, dtype=np.float64
        )
        warp_phase = np.asarray(self.stencil_warp_phase, dtype=np.float64)
        warp_axes = np.asarray(self.stencil_warp_axes, dtype=np.float64)
        for name, value in (
            ("stencil_warp_amplitude", warp_amplitude),
            ("stencil_warp_frequency", warp_frequency),
            ("stencil_warp_phase", warp_phase),
        ):
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"feature-line {name} must contain two finite values"
                )
        if np.any(warp_amplitude < 0.0):
            raise ValueError("feature-line stencil_warp_amplitude cannot be negative")
        if warp_axes.shape != (2, 3) or not np.all(np.isfinite(warp_axes)):
            raise ValueError(
                "feature-line stencil_warp_axes must contain two finite 3D axes"
            )
        axis_norms = np.linalg.norm(warp_axes, axis=1)
        if np.any((warp_amplitude > 0.0) & (axis_norms <= 1e-12)):
            raise ValueError("active feature-line stencil warp axes cannot be zero")
        warp_axes = warp_axes.copy()
        active_axes = axis_norms > 1e-12
        warp_axes[active_axes] /= axis_norms[active_axes, None]
        hatch_scale = float(self.line_hatch_scale)
        hatch_width = float(self.line_hatch_width)
        hatch_phase = float(self.line_hatch_phase) % 1.0
        hatch_softness = float(self.line_hatch_edge_softness)
        hatch_direction = np.asarray(
            self.line_hatch_direction, dtype=np.float64
        )
        if not np.isfinite(hatch_scale) or hatch_scale < 0.0:
            raise ValueError("feature-line hatch scale must be finite and non-negative")
        if not np.isfinite(hatch_width) or not 0.0 < hatch_width < 0.5:
            raise ValueError("feature-line hatch width must lie in (0, 0.5)")
        if not np.isfinite(hatch_softness) or hatch_softness < 0.0:
            raise ValueError("feature-line hatch softness must be finite and non-negative")
        if hatch_direction.shape != (3,) or not np.all(np.isfinite(hatch_direction)):
            raise ValueError("feature-line hatch direction must contain three finite values")
        hatch_norm = float(np.linalg.norm(hatch_direction))
        if hatch_scale > 0.0 and hatch_norm <= 1e-12:
            raise ValueError("active feature-line hatch direction cannot be zero")
        if hatch_norm > 1e-12:
            hatch_direction = hatch_direction / hatch_norm
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "include_silhouette", bool(self.include_silhouette))
        object.__setattr__(
            self,
            "normal_orientation_invariant",
            bool(self.normal_orientation_invariant),
        )
        object.__setattr__(
            self,
            "normal_shape_boundary_fallback",
            bool(self.normal_shape_boundary_fallback),
        )
        object.__setattr__(self, "color", _rgb(self.color))
        object.__setattr__(self, "include_materials", materials)
        object.__setattr__(self, "exclude_materials", excluded_materials)
        object.__setattr__(self, "include_shapes", shapes)
        object.__setattr__(
            self, "depth_colors", tuple(sorted(parsed_depth_colors, key=lambda item: item[0]))
        )
        object.__setattr__(self, "glossy_mix_strength", glossy_mix_strength)
        object.__setattr__(self, "stencil_offset", offset)
        object.__setattr__(self, "stencil_warp_amplitude", warp_amplitude)
        object.__setattr__(self, "stencil_warp_frequency", warp_frequency)
        object.__setattr__(self, "stencil_warp_phase", warp_phase)
        object.__setattr__(self, "stencil_warp_axes", warp_axes)
        object.__setattr__(self, "stencil_warp_profile", warp_profile)
        object.__setattr__(self, "line_hatch_scale", hatch_scale)
        object.__setattr__(self, "line_hatch_width", hatch_width)
        object.__setattr__(self, "line_hatch_direction", hatch_direction)
        object.__setattr__(self, "line_hatch_phase", hatch_phase)
        object.__setattr__(self, "line_hatch_edge_softness", hatch_softness)

    def active_at(self, depth: int) -> bool:
        if depth < self.min_depth:
            return False
        return self.max_depth is None or depth <= self.max_depth

    def color_at(self, depth: int) -> RGB:
        """Return the optional path-depth color override for this line type."""
        for line_depth, line_color in self.depth_colors:
            if line_depth == depth:
                return line_color
        return self.color

    def glossy_strength(self, default: float) -> float:
        """Resolve the glossy-path tint without changing other dictionaries."""
        if self.glossy_mix_strength is None:
            return float(default)
        return self.glossy_mix_strength

    def applies_to_material(self, material_id: str) -> bool:
        return (
            material_id not in self.exclude_materials
            and (not self.include_materials or material_id in self.include_materials)
        )

    def applies_to_shape(self, shape_id: str) -> bool:
        return not self.include_shapes or shape_id in self.include_shapes

    def line_hatch_weight(self, position: Sequence[float]) -> float:
        """Return object-space hatch coverage for an already detected line."""
        if self.line_hatch_scale <= 0.0:
            return 1.0
        point = np.asarray(position, dtype=np.float64).reshape(3)
        phase = np.mod(
            self.line_hatch_scale * np.dot(self.line_hatch_direction, point)
            + self.line_hatch_phase,
            1.0,
        )
        distance = min(float(phase), 1.0 - float(phase))
        if self.line_hatch_edge_softness <= 0.0:
            return float(distance < self.line_hatch_width)
        weight = np.clip(
            (self.line_hatch_width + self.line_hatch_edge_softness - distance)
            / (2.0 * self.line_hatch_edge_softness),
            0.0,
            1.0,
        )
        return float(weight * weight * (3.0 - 2.0 * weight))

    @property
    def centered_sampling_radius(self) -> float:
        """Bounding radius after factoring out the shared warp displacement."""
        local_radius = self.stencil_radius
        if self.stencil == "square":
            local_radius *= float(np.sqrt(2.0))
        return local_radius + float(np.linalg.norm(self.stencil_offset))

    @property
    def sampling_radius(self) -> float:
        """Radius of the origin-centered disk enclosing this line stencil."""
        # The two sinusoidal components can reach their extrema at the same
        # surface point. Include that worst-case displacement in the bounding
        # disk so warped stencils never read outside the sampled domain.
        maximum_center = np.abs(self.stencil_offset) + self.stencil_warp_amplitude
        local_radius = self.stencil_radius
        if self.stencil == "square":
            local_radius *= float(np.sqrt(2.0))
        return local_radius + float(np.linalg.norm(maximum_center))

    def stencil_warp(self, position: Sequence[float]) -> np.ndarray:
        """Evaluate only the shared continuous displacement component."""
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,):
            point = point.reshape(3)
        phase = (
            self.stencil_warp_frequency
            * (self.stencil_warp_axes @ point)
            + self.stencil_warp_phase
        )
        if self.stencil_warp_profile == "ripple":
            waveform = (
                0.82 * np.sin(phase)
                + 0.18 * np.sin(2.17 * phase + 0.55)
            )
        elif self.stencil_warp_profile == "sketchy":
            waveform = (
                0.60 * np.sin(phase)
                + 0.25 * np.sin(1.91 * phase + 0.73)
                + 0.15 * np.sin(3.17 * phase - 1.11)
            )
        else:
            waveform = np.sin(phase)
        return self.stencil_warp_amplitude * waveform

    def stencil_center(self, position: Sequence[float]) -> np.ndarray:
        """Evaluate the position-dependent stencil center for scalar paths."""
        return self.stencil_offset + self.stencil_warp(position)


@dataclass(frozen=True)
class FeatureLineConfig:
    """Global auxiliary-path construction parameters."""

    enabled: bool = False
    auxiliary_samples: int = 16
    cuda_auxiliary_batch_size: int = 4
    # Materialize the line-priority accumulator periodically instead of after
    # every dictionary. Larger intervals reduce CUDA launch/synchronization
    # overhead while keeping the temporary Dr.Jit graph bounded.
    cuda_line_eval_interval: int = 4
    # At a delta mirror, first transport the camera stencil to the reflected
    # surface, fit its local ray differential, and then draw a material-sized
    # stencil there. This avoids forcing every reflected material to share the
    # largest dictionary support while preserving the actual mirror path.
    resample_delta_reflections: bool = False
    # Also apply the material-local stencil after a conditionally specular
    # microfacet event. The distribution flag is preserved so outer samples
    # still integrate to a naturally blurred glossy reflection.
    resample_glossy_reflections: bool = False
    max_normal_angle_degrees: float = 75.0
    glossy_line_strength: float = 0.45
    glossy_line_color: RGB = field(
        default_factory=lambda: np.array([0.30, 0.30, 0.30])
    )
    types: tuple[FeatureLineType, ...] = ()

    def __post_init__(self) -> None:
        if self.auxiliary_samples < 2:
            raise ValueError("feature lines require at least two auxiliary samples")
        if self.cuda_auxiliary_batch_size < 1:
            raise ValueError("cuda_auxiliary_batch_size must be positive")
        if self.cuda_line_eval_interval < 1:
            raise ValueError("cuda_line_eval_interval must be positive")
        if not 0.0 < self.max_normal_angle_degrees < 180.0:
            raise ValueError("max_normal_angle_degrees must lie in (0, 180)")
        if not 0.0 <= self.glossy_line_strength <= 1.0:
            raise ValueError("glossy_line_strength must lie in [0, 1]")
        color = np.asarray(self.glossy_line_color, dtype=np.float64)
        if color.shape != (3,) or np.any(~np.isfinite(color)):
            raise ValueError("glossy_line_color must be a finite RGB triplet")
        object.__setattr__(self, "glossy_line_color", color)
        if self.enabled and not self.types:
            raise ValueError("enabled feature lines require at least one line type")

    @property
    def max_radius(self) -> float:
        return max(
            (line.sampling_radius for line in self.types),
            default=0.0,
        )

    @property
    def max_origin_radius(self) -> float:
        """Largest support for an origin-centred, unclassified reflector.

        Warped line dictionaries include their worst-case displacement in
        ``sampling_radius``.  A floor or mirror lane is not yet associated
        with that material, so using this expanded radius there wastes the
        auxiliary samples and blurs unrelated reflected contours.
        """
        radii = [
            line.sampling_radius
            for line in self.types
            if not np.any(line.stencil_warp_amplitude > 0.0)
        ]
        return max(radii, default=self.max_radius)

    @property
    def min_radius(self) -> float:
        """Smallest dictionary support used by the S1.4 mixture density."""
        return min(
            (line.sampling_radius for line in self.types),
            default=0.0,
        )

    @property
    def sampling_stencil(self) -> str:
        # Supplemental S1.4 uses one origin-centered bounding disk for all
        # currently active dictionaries. A square dictionary is enclosed by
        # that disk; it does not turn every other material's sampling domain
        # into a large square.
        return "disk"

    def can_apply_from(self, depth: int) -> bool:
        """Whether any configured line type can still apply at/after depth."""
        return self.enabled and any(
            line.max_depth is None or line.max_depth >= depth
            for line in self.types
        )

    def active_types(self, depth: int) -> tuple[FeatureLineType, ...]:
        """Return only line dictionaries that can affect this path vertex."""
        return tuple(line for line in self.types if line.active_at(depth))

    def needs_measurement(self, depth: int, *measurements: str) -> bool:
        """Whether an active dictionary reads one of the requested fields."""
        requested = set(measurements)
        return any(
            line.measurement in requested
            or (
                "shape_id" in requested
                and line.normal_shape_boundary_fallback
            )
            for line in self.active_types(depth)
        )


def build_feature_lines(
    spec: Mapping[str, Any] | None,
) -> FeatureLineConfig:
    """Parse the optional top-level ``feature_lines`` JSON object."""
    if not spec:
        return FeatureLineConfig()
    # JSON has no native comment syntax. Configuration files may therefore
    # place human-readable annotations in keys beginning with ``_``; these
    # fields are deliberately ignored by the runtime parser.
    values = {
        key: value for key, value in dict(spec).items()
        if not str(key).startswith("_")
    }
    raw_types = values.pop("types", ())
    line_types = []
    for index, raw in enumerate(raw_types):
        entry = {
            key: value for key, value in dict(raw).items()
            if not str(key).startswith("_")
        }
        entry.setdefault("name", f"line_{index}")
        line_types.append(FeatureLineType(**entry))
    values.setdefault("enabled", True)
    return FeatureLineConfig(types=tuple(line_types), **values)


__all__ = [
    "FeatureLineConfig",
    "FeatureLineType",
    "build_feature_lines",
    "finite_difference_slope",
    "minimal_rotation",
    "normalize",
    "parallel_transport_half_vector",
    "view_oriented_frame",
]

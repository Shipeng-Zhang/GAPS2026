"""Canonical image-space tone lifting for path-space stylization.

This module implements Section 5 and Supplemental Sections S2.2--S2.5 of
*Lifting Lines and Tone*.  A small set of camera rays is propagated using
perfect reflection about the macro geometry, independently of the scene
materials.  Their path vertices carry their originating film coordinates and
provide the samples for a local inverse of the canonical mapping.

The scalar implementation deliberately stores only the fields used by the
inverse (position, normal, incoming direction, shape and film coordinate).
The CUDA equivalent in :mod:`sre.cuda_tone_mapping` follows the same layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ToneMappingConfig:
    """Parameters of the material-independent mirror-anchor mapping."""

    enabled: bool = False
    anchor_samples: int = 16
    max_depth: int = 4
    search_radius: float = 256.0
    min_radius: float = 1.5
    reference_width: float = 0.0
    radial_rings: int = 4
    min_candidates: int = 4
    sigma_scale: float = 1.0
    angular_limit_degrees: float = 35.0
    condition_epsilon: float = 1e-6
    cuda_anchor_batch_size: int = 2

    def __post_init__(self) -> None:
        if self.anchor_samples < 4:
            raise ValueError("tone_mapping.anchor_samples must be at least four")
        if self.max_depth < 1:
            raise ValueError("tone_mapping.max_depth must be positive")
        if not 0.0 < self.min_radius <= self.search_radius:
            raise ValueError(
                "tone_mapping radii must satisfy 0 < min_radius <= search_radius"
            )
        if self.reference_width < 0.0:
            raise ValueError("tone_mapping.reference_width cannot be negative")
        if self.radial_rings < 1:
            raise ValueError("tone_mapping.radial_rings must be positive")
        if not 3 <= self.min_candidates <= self.anchor_samples:
            raise ValueError(
                "tone_mapping.min_candidates must lie in [3, anchor_samples]"
            )
        if self.sigma_scale <= 0.0:
            raise ValueError("tone_mapping.sigma_scale must be positive")
        if not 0.0 < self.angular_limit_degrees <= 180.0:
            raise ValueError(
                "tone_mapping.angular_limit_degrees must lie in (0, 180]"
            )
        if self.condition_epsilon <= 0.0:
            raise ValueError("tone_mapping.condition_epsilon must be positive")
        if self.cuda_anchor_batch_size < 1:
            raise ValueError(
                "tone_mapping.cuda_anchor_batch_size must be positive"
            )

    def active_at(self, depth: int) -> bool:
        return self.enabled and 0 <= depth < self.max_depth

    def can_extend_from(self, depth: int) -> bool:
        return self.enabled and depth + 1 < self.max_depth


def build_tone_mapping(
    spec: Mapping[str, Any] | None,
) -> ToneMappingConfig:
    values = {
        key: value
        for key, value in dict(spec or {}).items()
        if not str(key).startswith("_")
    }
    return ToneMappingConfig(**values)


def multiscale_anchor_offsets(config: ToneMappingConfig) -> tuple[np.ndarray, ...]:
    """Return a deterministic center sample plus concentric pixel rings.

    The supplemental implementation queries a global cache out to 512 pixels
    at 1920x1080.  A global cache is undesirable on a GPU, so this project uses
    a streaming, camera-sample-local equivalent.  Geometrically spaced rings
    retain both the local samples needed for a well-conditioned fit and sparse
    long-range samples needed by glossy transport.
    """

    remaining = config.anchor_samples - 1
    ring_count = min(config.radial_rings, max(1, remaining // 3))
    radii = np.geomspace(config.min_radius, config.search_radius, ring_count)
    base_count, extra = divmod(remaining, ring_count)
    offsets: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    for ring_index, radius in enumerate(radii):
        count = base_count + (1 if ring_index < extra else 0)
        phase = (ring_index * 0.3819660112501051) % 1.0
        for index in range(count):
            angle = 2.0 * pi * (index / count + phase)
            offsets.append(
                float(radius)
                * np.array([cos(angle), sin(angle)], dtype=np.float64)
            )
    return tuple(offsets)


def _stable_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-20)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(normal[0])) > 0.8:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    first = np.cross(normal, axis)
    first /= max(float(np.linalg.norm(first)), 1e-20)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-20)
    return first, second


@dataclass(frozen=True)
class ToneInverseResult:
    coordinate: np.ndarray
    valid: bool
    confidence: float
    method: str


def linear_mls_inverse(
    points: Sequence[Sequence[float]],
    coordinates: Sequence[Sequence[float]],
    query: Sequence[float],
    incoming_direction: Sequence[float],
    *,
    min_candidates: int = 4,
    sigma_scale: float = 1.0,
    condition_epsilon: float = 1e-6,
    fallback: Sequence[float] = (0.0, 0.0),
) -> ToneInverseResult:
    """Evaluate Supplemental S2.5's local linear MLS inverse.

    A view-oriented plane through the query is used.  Changing the in-plane
    basis is an affine coordinate change and therefore does not alter the
    fitted value at the query.  Centered normal equations are used instead of
    explicitly inverting the paper's 3x3 matrix, which improves conditioning
    and reduces the CUDA implementation to a 2x2 solve.
    """

    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    coordinate_array = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    fallback_array = np.asarray(fallback, dtype=np.float64).reshape(2)
    finite = np.all(np.isfinite(points_array), axis=1) & np.all(
        np.isfinite(coordinate_array), axis=1
    )
    points_array = points_array[finite]
    coordinate_array = coordinate_array[finite]
    if len(points_array) == 0:
        return ToneInverseResult(fallback_array.copy(), False, 0.0, "first_edge")

    query_array = np.asarray(query, dtype=np.float64).reshape(3)
    incoming = np.asarray(incoming_direction, dtype=np.float64).reshape(3)
    plane_normal = -incoming / max(float(np.linalg.norm(incoming)), 1e-20)
    first, second = _stable_plane_basis(plane_normal)
    displacement = points_array - query_array
    projected = np.stack((displacement @ first, displacement @ second), axis=1)
    distances = np.linalg.norm(projected, axis=1)

    nearest = int(np.argmin(distances))
    if len(points_array) < min_candidates:
        return ToneInverseResult(
            coordinate_array[nearest].copy(), True,
            float(len(points_array)) / min_candidates, "nearest",
        )

    sigma = sigma_scale * max(float(np.mean(distances)), 1e-8)
    weights = np.exp(-np.square(distances / sigma))
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
        return ToneInverseResult(
            coordinate_array[nearest].copy(), True, 0.1, "nearest"
        )

    mean_p = np.sum(weights[:, None] * projected, axis=0) / weight_sum
    mean_s = np.sum(weights[:, None] * coordinate_array, axis=0) / weight_sum
    centered_p = projected - mean_p
    centered_s = coordinate_array - mean_s
    covariance = (centered_p * weights[:, None]).T @ centered_p
    cross = (centered_p * weights[:, None]).T @ centered_s
    determinant = float(np.linalg.det(covariance))
    scale = max(float(np.trace(covariance)) ** 2, 1e-20)
    relative_determinant = determinant / scale
    if not np.isfinite(relative_determinant) or relative_determinant <= condition_epsilon:
        # The affine/barycentric minimal-set fallback from S2.3 is equivalent
        # to an unweighted affine solve.  Least-squares handles both exactly
        # determined and mildly overdetermined sparse neighborhoods.
        design = np.column_stack((np.ones(len(projected)), projected))
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(
                design, coordinate_array, rcond=condition_epsilon
            )
            if rank == 3:
                coordinate = np.array([1.0, 0.0, 0.0]) @ coefficients
                return ToneInverseResult(coordinate, True, 0.25, "affine")
        except np.linalg.LinAlgError:
            pass
        return ToneInverseResult(
            coordinate_array[nearest].copy(), True, 0.1, "nearest"
        )

    gradient = np.linalg.solve(covariance, cross)
    coordinate = mean_s - mean_p @ gradient
    confidence = min(
        1.0,
        (len(points_array) / max(min_candidates, 1))
        * np.sqrt(max(relative_determinant, 0.0) / condition_epsilon),
    )
    return ToneInverseResult(coordinate, True, float(confidence), "linear_mls")


@dataclass
class ScalarToneAnchor:
    coordinate: np.ndarray
    direction: np.ndarray
    position: np.ndarray
    geometric_normal: np.ndarray
    shape: Any
    valid: bool


@dataclass
class ScalarToneFrame:
    anchors: list[ScalarToneAnchor] = field(default_factory=list)

    @property
    def first_edge_coordinate(self) -> np.ndarray:
        if not self.anchors:
            return np.zeros(2, dtype=np.float64)
        return self.anchors[0].coordinate


class ScalarToneMapper:
    """Low-memory scalar reference implementation of canonical tone lifting."""

    def __init__(self, config: ToneMappingConfig) -> None:
        self.config = config
        self.offsets = multiscale_anchor_offsets(config)
        self._sensor: Any = None
        self._to_camera: Any = None
        self._to_world: Any = None
        self._focus_distance: float = 1.0
        self._pixel_scale: float = 1.0

    def prepare(self, scene: Any) -> None:
        if self._sensor is not None:
            return
        sensors = scene.sensors()
        if not sensors:
            raise ValueError("tone mapping requires a projective camera")
        self._sensor = sensors[0]
        self._to_world = self._sensor.world_transform()
        self._to_camera = self._to_world.inverse()
        self._focus_distance = float(self._sensor.focus_distance())
        film_width = float(self._sensor.film().size()[0])
        self._pixel_scale = (
            film_width / self.config.reference_width
            if self.config.reference_width > 0.0 else 1.0
        )

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float64).reshape(-1)[:3]

    @staticmethod
    def _signed_nonzero(value: float) -> float:
        if abs(value) > 1e-20:
            return value
        return -1e-20 if value < 0.0 else 1e-20

    def _focus_point(self, origin: Any, direction: Any) -> np.ndarray:
        local_origin = self._array(self._to_camera @ origin)
        local_direction = self._array(self._to_camera @ direction)
        distance = (
            self._focus_distance - local_origin[2]
        ) / self._signed_nonzero(float(local_direction[2]))
        return local_origin + distance * local_direction

    def primary_coordinate(self, ray: Any) -> np.ndarray:
        focus = self._focus_point(ray.o, ray.d)
        focus_x = self._focus_point(ray.o_x, ray.d_x)
        focus_y = self._focus_point(ray.o_y, ray.d_y)
        delta_x = focus_x[0] / self._focus_distance - focus[0] / self._focus_distance
        delta_y = focus_y[1] / self._focus_distance - focus[1] / self._focus_distance
        film_size = np.asarray(self._sensor.film().size(), dtype=np.float64)
        return np.array(
            [
                0.5 * film_size[0]
                + (focus[0] / self._focus_distance)
                / self._signed_nonzero(float(delta_x)),
                0.5 * film_size[1]
                + (focus[1] / self._focus_distance)
                / self._signed_nonzero(float(delta_y)),
            ],
            dtype=np.float64,
        ) / self._pixel_scale

    def spawn_primary(self, scene: Any, ray: Any, active: bool = True) -> ScalarToneFrame:
        import mitsuba as mi

        self.prepare(scene)
        base_coordinate = self.primary_coordinate(ray)
        local_origin = self._array(self._to_camera @ ray.o)
        focus = self._focus_point(ray.o, ray.d)
        focus_x = self._focus_point(ray.o_x, ray.d_x)
        focus_y = self._focus_point(ray.o_y, ray.d_y)
        delta_x = focus_x - focus
        delta_y = focus_y - focus
        anchors: list[ScalarToneAnchor] = []
        for offset in self.offsets:
            ray_offset = self._pixel_scale * offset
            target = focus + ray_offset[0] * delta_x + ray_offset[1] * delta_y
            local_direction = target - local_origin
            local_direction /= max(float(np.linalg.norm(local_direction)), 1e-20)
            world_direction = self._array(
                self._to_world @ mi.Vector3f(local_direction)
            )
            world_direction /= max(float(np.linalg.norm(world_direction)), 1e-20)
            auxiliary = mi.Ray3f(ray)
            auxiliary.o = mi.Point3f(ray.o)
            auxiliary.d = mi.Vector3f(world_direction)
            interaction = scene.ray_intersect(
                auxiliary, int(mi.RayFlags.Minimal), False, bool(active)
            )
            valid = bool(active) and bool(interaction.is_valid())
            anchors.append(
                ScalarToneAnchor(
                    coordinate=base_coordinate + offset,
                    direction=world_direction,
                    position=self._array(interaction.p),
                    geometric_normal=self._array(interaction.n),
                    shape=interaction.shape,
                    valid=valid,
                )
            )
        return ScalarToneFrame(anchors)

    def query(
        self,
        frame: ScalarToneFrame,
        interaction: Any,
        ray: Any,
        depth: int,
        analytic: bool = False,
    ) -> ToneInverseResult:
        central_compatible = (
            bool(frame.anchors)
            and frame.anchors[0].valid
            and frame.anchors[0].shape == interaction.shape
        )
        if depth == 0 or (analytic and central_compatible):
            return ToneInverseResult(
                frame.first_edge_coordinate.copy(), True, 1.0, "analytic"
            )
        incoming = self._array(ray.d)
        incoming /= max(float(np.linalg.norm(incoming)), 1e-20)
        cosine_limit = cos(np.radians(self.config.angular_limit_degrees))
        candidates = [
            anchor
            for anchor in frame.anchors
            if anchor.valid
            and anchor.shape == interaction.shape
            and float(np.dot(incoming, anchor.direction)) >= cosine_limit
        ]
        return linear_mls_inverse(
            [anchor.position for anchor in candidates],
            [anchor.coordinate for anchor in candidates],
            self._array(interaction.p),
            incoming,
            min_candidates=self.config.min_candidates,
            sigma_scale=self.config.sigma_scale,
            condition_epsilon=self.config.condition_epsilon,
            fallback=frame.first_edge_coordinate,
        )

    def extend(
        self,
        scene: Any,
        frame: ScalarToneFrame,
        interaction: Any,
        active: bool = True,
    ) -> ScalarToneFrame:
        import mitsuba as mi

        children: list[ScalarToneAnchor] = []
        for anchor in frame.anchors:
            prefix = bool(active) and anchor.valid and anchor.shape == interaction.shape
            direction = anchor.direction
            normal = anchor.geometric_normal
            normal /= max(float(np.linalg.norm(normal)), 1e-20)
            reflected = direction - 2.0 * float(np.dot(direction, normal)) * normal
            reflected /= max(float(np.linalg.norm(reflected)), 1e-20)
            source = mi.Interaction3f()
            source.p = mi.Point3f(anchor.position)
            source.n = mi.Normal3f(anchor.geometric_normal)
            child_ray = source.spawn_ray(mi.Vector3f(reflected))
            child = scene.ray_intersect(
                child_ray, int(mi.RayFlags.Minimal), False, prefix
            )
            valid = prefix and bool(child.is_valid())
            children.append(
                ScalarToneAnchor(
                    coordinate=anchor.coordinate,
                    direction=reflected,
                    position=self._array(child.p),
                    geometric_normal=self._array(child.n),
                    shape=child.shape,
                    valid=valid,
                )
            )
        return ScalarToneFrame(children)


__all__ = [
    "ScalarToneAnchor",
    "ScalarToneFrame",
    "ScalarToneMapper",
    "ToneInverseResult",
    "ToneMappingConfig",
    "build_tone_mapping",
    "linear_mls_inverse",
    "multiscale_anchor_offsets",
]

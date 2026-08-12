"""CUDA auxiliary-path transport for image-space feature lines."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, gcd, pi, radians, sqrt
from typing import Any

import drjit as dr
import mitsuba as mi

try:
    from .cuda_backend import gather_lanes
    from .feature_lines import FeatureLineConfig, FeatureLineType
except ImportError:
    from cuda_backend import gather_lanes
    from feature_lines import FeatureLineConfig, FeatureLineType


def _finite_vector(value: Any) -> Any:
    return dr.isfinite(value[0]) & dr.isfinite(value[1]) & dr.isfinite(value[2])


def _normalize(value: Any) -> Any:
    inverse = dr.rsqrt(dr.maximum(dr.squared_norm(value), 1e-20))
    return value * inverse


def _view_frame(normal: Any, view: Any) -> tuple[Any, Any]:
    normal = _normalize(normal)
    view = _normalize(view)
    first = dr.cross(normal, view)
    fallback_axis = dr.select(
        dr.abs(normal[0]) < 0.8,
        mi.Vector3f(1.0, 0.0, 0.0),
        mi.Vector3f(0.0, 1.0, 0.0),
    )
    fallback = dr.cross(normal, fallback_axis)
    first = dr.select(dr.squared_norm(first) > 1e-16, first, fallback)
    first = _normalize(first)
    second = _normalize(dr.cross(first, normal))
    return first, second


def _transport_half_vector(
    half_vector: Any,
    normal: Any,
    auxiliary_normal: Any,
    view: Any,
    auxiliary_view: Any,
) -> Any:
    normal = _normalize(normal)
    auxiliary_normal = _normalize(auxiliary_normal)
    half_vector = _normalize(half_vector)
    half_vector = dr.select(
        dr.dot(half_vector, normal) < 0.0, -half_vector, half_vector
    )
    cosine_tilt = dr.clamp(dr.dot(half_vector, normal), -1.0, 1.0)
    sine_tilt = dr.safe_sqrt(1.0 - dr.square(cosine_tilt))
    tangent = _normalize(half_vector - cosine_tilt * normal)
    first, second = _view_frame(normal, view)
    angle = dr.atan2(dr.dot(tangent, second), dr.dot(tangent, first))
    aux_first, aux_second = _view_frame(auxiliary_normal, auxiliary_view)
    aux_tangent = dr.cos(angle) * aux_first + dr.sin(angle) * aux_second
    transported = _normalize(
        cosine_tilt * auxiliary_normal + sine_tilt * aux_tangent
    )
    return dr.select(sine_tilt <= 1e-8, auxiliary_normal, transported)


def _minimal_rotation(value: Any, source: Any, target: Any) -> Any:
    value = _normalize(value)
    source = _normalize(source)
    target = _normalize(target)
    axis = dr.cross(source, target)
    cosine_angle = dr.clamp(dr.dot(source, target), -1.0, 1.0)
    regular = (
        value
        + dr.cross(axis, value)
        + dr.cross(axis, dr.cross(axis, value))
        / dr.maximum(1.0 + cosine_angle, 1e-8)
    )
    tangent, _ = _view_frame(source, value)
    antipodal = 2.0 * dr.dot(value, tangent) * tangent - value
    result = dr.select(
        cosine_angle > 1.0 - 1e-7,
        value,
        dr.select(cosine_angle < -1.0 + 1e-6, antipodal, regular),
    )
    return _normalize(result)


def _concentric_disk(first: Any, second: Any) -> tuple[Any, Any]:
    x = 2.0 * first - 1.0
    y = 2.0 * second - 1.0
    x_major = dr.abs(x) > dr.abs(y)
    radius = dr.select(x_major, x, y)
    angle = dr.select(
        x_major,
        (pi / 4.0) * y / dr.select(x != 0.0, x, 1.0),
        (pi / 2.0) - (pi / 4.0) * x / dr.select(y != 0.0, y, 1.0),
    )
    origin = (x == 0.0) & (y == 0.0)
    return (
        dr.select(origin, 0.0, radius * dr.cos(angle)),
        dr.select(origin, 0.0, radius * dr.sin(angle)),
    )


@dataclass
class CudaAuxiliaryFrame:
    """Compact state for one DFS level (Supplemental S1.4/Fig. 1).

    A full ``SurfaceInteraction3f`` contains UV derivatives, tangent
    derivatives, primitive indices, and other fields that feature-line
    continuation never reads.  Keeping sixteen of those records for every
    camera lane was the dominant memory cost.  This frame stores only the
    sufficient statistics needed by Eqs. (13)--(21).
    """

    offsets: list[tuple[Any, Any]]
    directions: list[Any]
    ray_origins: list[Any]
    positions: list[Any]
    geometric_normals: list[Any]
    shading_normals: list[Any]
    depths: list[Any]
    shapes: list[Any]
    materials: list[Any]
    prefix_valid: list[Any]
    # True when this auxiliary path has crossed at least one non-delta
    # glossy event. Glossy transport should soften line contrast, while a
    # perfect mirror preserves the original black feature line.
    distributional: Any
    time: Any
    wavelengths: Any

    def lane_width(self) -> int:
        """Return the parent wavefront width, ignoring broadcast fields."""
        values: list[Any] = [self.time]
        for first, second in self.offsets:
            values.extend((first, second))
        for arrays in (
            self.directions,
            self.ray_origins,
            self.positions,
            self.geometric_normals,
            self.shading_normals,
            self.depths,
            self.shapes,
            self.materials,
            self.prefix_valid,
        ):
            values.extend(arrays)
        return max((dr.width(value) for value in values), default=1)

    def eval(self) -> "CudaAuxiliaryFrame":
        """Materialize the frame and sever transient intersection graphs.

        This mirrors the stored ``AuxiliaryPathFrame`` in Supplemental S1.4.
        Without this boundary Dr.Jit may retain the full construction graph of
        every compact field, which defeats the memory benefit of the compact
        representation.
        """
        values: list[Any] = [self.time]
        for first, second in self.offsets:
            values.extend((first, second))
        for arrays in (
            self.directions,
            self.ray_origins,
            self.positions,
            self.geometric_normals,
            self.shading_normals,
            self.depths,
            self.shapes,
            self.materials,
            self.prefix_valid,
        ):
            values.extend(arrays)
        dr.eval(*values)
        return self

    def gather(self, indices: Any) -> "CudaAuxiliaryFrame":
        parent_width = self.lane_width()
        return CudaAuxiliaryFrame(
            offsets=[
                (
                    gather_lanes(mi.Float, offset[0], indices, parent_width,
                                 "auxiliary offset x"),
                    gather_lanes(mi.Float, offset[1], indices, parent_width,
                                 "auxiliary offset y"),
                )
                for offset in self.offsets
            ],
            directions=[
                gather_lanes(mi.Vector3f, direction, indices, parent_width,
                             "auxiliary direction")
                for direction in self.directions
            ],
            ray_origins=[
                gather_lanes(mi.Point3f, origin, indices, parent_width,
                             "auxiliary ray origin")
                for origin in self.ray_origins
            ],
            positions=[
                gather_lanes(mi.Point3f, position, indices, parent_width,
                             "auxiliary position")
                for position in self.positions
            ],
            geometric_normals=[
                gather_lanes(mi.Normal3f, normal, indices, parent_width,
                             "auxiliary geometric normal")
                for normal in self.geometric_normals
            ],
            shading_normals=[
                gather_lanes(mi.Normal3f, normal, indices, parent_width,
                             "auxiliary shading normal")
                for normal in self.shading_normals
            ],
            depths=[
                gather_lanes(mi.Float, depth, indices, parent_width,
                             "auxiliary depth") for depth in self.depths
            ],
            shapes=[
                gather_lanes(mi.ShapePtr, shape, indices, parent_width,
                             "auxiliary shape") for shape in self.shapes
            ],
            materials=[
                gather_lanes(mi.BSDFPtr, material, indices, parent_width,
                             "auxiliary material")
                for material in self.materials
            ],
            prefix_valid=[
                gather_lanes(mi.Bool, valid, indices, parent_width,
                             "auxiliary prefix mask")
                for valid in self.prefix_valid
            ],
            distributional=gather_lanes(
                mi.Bool, self.distributional, indices, parent_width,
                "auxiliary distributional mask"
            ),
            time=gather_lanes(
                mi.Float, self.time, indices, parent_width, "auxiliary time"
            ),
            # cuda_ad_rgb has a zero-width wavelength packet. It is shared by
            # every compacted lane and therefore needs no gather.
            wavelengths=self.wavelengths,
        )


class CudaFeatureLineTracer:
    """Paper-aligned conditional lifting and extremal-gradient search."""

    def __init__(self, config: FeatureLineConfig) -> None:
        self.config = config
        # Keep only a small number of unevaluated OptiX intersections alive.
        # This is an execution detail: it does not change auxiliary samples or
        # their random numbers. A value of one minimizes memory, while the
        # default four amortizes CUDA launch overhead.
        self._trace_batch_size = min(
            config.cuda_auxiliary_batch_size, config.auxiliary_samples
        )
        self.normal_cosine_limit = cos(radians(config.max_normal_angle_degrees))
        sample_count = config.auxiliary_samples + 1
        all_pairs = [
            (first, second)
            for first in range(sample_count)
            for second in range(first + 1, sample_count)
        ]
        self._pair_orders: list[list[tuple[int, int]]] = []
        self._line_materials: list[list[Any]] = [
            [] for _ in config.types
        ]
        self._line_shapes: list[list[Any]] = [[] for _ in config.types]
        for line in config.types:
            start = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(line.name)
            ) % len(all_pairs)
            stride = 53
            while gcd(stride, len(all_pairs)) != 1:
                stride += 2
            self._pair_orders.append([
                all_pairs[(start + index * stride) % len(all_pairs)]
                for index in range(len(all_pairs))
            ])
        # A material/shape restricted line only needs to cover the primary
        # sampling domain of dictionaries that can coexist on the same lane.
        # The old global check compared against the largest stencil of every
        # robot in Fig. 11, so even a 2.2 px blue outline scanned all 136
        # auxiliary pairs although its sixteen primary samples were already
        # inside that outline. This conservative per-filter test preserves the
        # exact samples/result and lets those dictionaries stop after the
        # requested 16 comparisons.
        self._primary_domain_covered = [
            self._line_covers_filtered_primary_domain(index)
            for index in range(len(config.types))
        ]
        self._intersector: Any = None

    @staticmethod
    def _filters_overlap(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
        return not first or not second or bool(set(first).intersection(second))

    def _line_covers_filtered_primary_domain(self, line_index: int) -> bool:
        line = self.config.types[line_index]
        if (
            not line.active_at(0)
            or line.stencil != self.config.sampling_stencil
            or bool(line.stencil_offset[0])
            or bool(line.stencil_offset[1])
            or any(float(value) > 0.0 for value in line.stencil_warp_amplitude)
        ):
            return False
        for other in self.config.types:
            if not other.active_at(0):
                continue
            if not self._filters_overlap(
                line.include_materials, other.include_materials
            ):
                continue
            if line.exclude_materials or other.exclude_materials:
                return False
            if not self._filters_overlap(line.include_shapes, other.include_shapes):
                continue
            if float(other.sampling_radius) > float(line.stencil_radius) + 1e-9:
                return False
        return True

    def set_intersector(self, intersector: Any) -> None:
        self._intersector = intersector

    def _ray_intersect(
        self, scene: Any, ray: Any, flags: int, active: Any
    ) -> Any:
        if self._intersector is not None:
            return self._intersector.ray_intersect(
                ray, flags, False, active
            )
        return scene.ray_intersect(ray, flags, False, active)

    def prepare(self, scene: Any) -> None:
        """Resolve configured Mitsuba material IDs to inexpensive BSDF pointers."""
        by_id: dict[str, Any] = {}
        shape_by_id: dict[str, Any] = {}
        shape_pointer = getattr(mi, "ShapePtr", lambda value: value)
        for shape in scene.shapes():
            if shape.id():
                shape_by_id.setdefault(shape.id(), shape_pointer(shape))
            bsdf = shape.bsdf()
            if bsdf is not None and bsdf.id():
                by_id.setdefault(bsdf.id(), mi.BSDFPtr(bsdf))
        self._line_materials = [
            [by_id[material_id] for material_id in line.include_materials
             if material_id in by_id]
            for line in self.config.types
        ]
        self._line_excluded_materials = [
            [by_id[material_id] for material_id in line.exclude_materials
             if material_id in by_id]
            for line in self.config.types
        ]
        self._line_shapes = [
            [shape_by_id[shape_id] for shape_id in line.include_shapes
             if shape_id in shape_by_id]
            for line in self.config.types
        ]

    def _material_active(self, line_index: int, material: Any, active: Any) -> Any:
        line = self.config.types[line_index]
        if not line.include_materials:
            matches = mi.Bool(active)
        else:
            matches = mi.Bool(False)
            for expected in self._line_materials[line_index]:
                matches |= material == expected
            matches &= active
        for excluded in self._line_excluded_materials[line_index]:
            matches &= material != excluded
        return matches

    def _shape_active(self, line_index: int, shape: Any, active: Any) -> Any:
        """Restrict a line dictionary to selected mesh parts, if requested."""
        line = self.config.types[line_index]
        # ``shape`` is optional for callers that only use the tracer as a
        # standalone helper. In the integrator it is always the primary SI's
        # ShapePtr, so this fallback simply preserves the old behavior.
        if not line.include_shapes or shape is None:
            return mi.Bool(active)
        matches = mi.Bool(False)
        for expected in self._line_shapes[line_index]:
            matches |= shape == expected
        return mi.Bool(active) & matches

    def _primary_sampling_radius(
        self,
        material: Any,
        shape: Any,
        active: Any,
        centered_warp: Any | None = None,
        depth: int = 0,
    ) -> Any:
        """Use the tight bounding disk for the directly visible material.

        A floor or mirror may reveal an arbitrary configured material at the
        next edge, so lanes that match no material-filtered dictionary retain
        the global radius. Direct robot lanes avoid wasting their sixteen
        stratified samples in another robot's much larger stencil.
        """
        radius = mi.Float(0.0)
        matched = mi.Bool(False)
        if centered_warp is None:
            centered_warp = mi.Bool(False)
        for line_index, line in enumerate(self.config.types):
            if not line.active_at(depth):
                continue
            line_active = self._material_active(
                line_index, material, active
            )
            line_active &= self._shape_active(line_index, shape, active)
            radius = dr.maximum(
                radius,
                dr.select(
                    line_active,
                    dr.select(
                        centered_warp,
                        float(line.centered_sampling_radius),
                        float(line.sampling_radius),
                    ),
                    0.0,
                ),
            )
            matched |= line_active
        # Unclassified floor/mirror lanes must not inherit the worst-case
        # displacement radius of a warped dictionary. Direct warped hits are
        # centred separately above; the fallback uses only fixed supports.
        return dr.select(matched, radius, float(self.config.max_origin_radius))

    def _primary_inner_radius(
        self,
        material: Any,
        shape: Any,
        active: Any,
        centered_warp: Any | None = None,
        depth: int = 0,
    ) -> Any:
        """Smallest active material stencil for the S1.4 mixture density."""
        radius = mi.Float(float(self.config.max_origin_radius))
        matched = mi.Bool(False)
        if centered_warp is None:
            centered_warp = mi.Bool(False)
        for line_index, line in enumerate(self.config.types):
            if not line.active_at(depth):
                continue
            line_active = self._material_active(line_index, material, active)
            line_active &= self._shape_active(line_index, shape, active)
            radius = dr.minimum(
                radius,
                dr.select(
                    line_active,
                    dr.select(
                        centered_warp,
                        float(line.centered_sampling_radius),
                        float(line.sampling_radius),
                    ),
                    float(self.config.max_origin_radius),
                ),
            )
            matched |= line_active
        # A floor/mirror can reveal any line dictionary after the next edge.
        # Use the global smallest support for the inner half of the mixture
        # and the global largest support for the outer half. Previously both
        # halves used max_radius, starving reflected 1 px part boundaries and
        # turning otherwise recognizable reflections into colour fog.
        return dr.select(matched, radius, float(self.config.min_radius))

    def _primary_anchor_radius(
        self, material: Any, shape: Any, active: Any, depth: int = 0
    ) -> Any:
        """Largest non-warped support kept in the origin-centered half.

        A displaced dictionary can coexist with a conventional depth/shape
        anchor (Fig. 11 magenta). The anchor needs its own fixed offset rather
        than being starved by the tiny global minimum radius.
        """
        radius = mi.Float(0.0)
        matched = mi.Bool(False)
        for line_index, line in enumerate(self.config.types):
            if not line.active_at(depth):
                continue
            if any(float(value) > 0.0 for value in line.stencil_warp_amplitude):
                continue
            line_active = self._material_active(line_index, material, active)
            line_active &= self._shape_active(line_index, shape, active)
            radius = dr.maximum(
                radius,
                dr.select(line_active, float(line.sampling_radius), 0.0),
            )
            matched |= line_active
        return dr.select(matched, radius, self._primary_inner_radius(
            material, shape, active, depth=depth
        ))

    def _primary_has_origin_anchor(
        self, material: Any, shape: Any, active: Any, depth: int = 0
    ) -> Any:
        """Whether a directly visible lane has a non-displaced dictionary."""
        matched = mi.Bool(False)
        for line_index, line in enumerate(self.config.types):
            if not line.active_at(depth):
                continue
            if any(float(value) > 0.0 for value in line.stencil_warp_amplitude):
                continue
            line_active = self._material_active(line_index, material, active)
            line_active &= self._shape_active(line_index, shape, active)
            matched |= line_active
        return matched

    def _primary_warp_center(
        self,
        material: Any,
        shape: Any,
        position: Any,
        active: Any,
        depth: int = 0,
    ) -> tuple[Any, Any, Any]:
        """Center direct-object samples on their shared displaced stencil.

        Fig. 11's magenta dictionaries use one continuous displacement field
        and several fixed offsets. Centering the sixteen primary auxiliary
        samples on that field trades the otherwise huge origin-centered disk
        for the tight stencil union. Reflection lanes hit the floor first and
        intentionally keep the conservative origin-centered bounding disk.
        """
        center_x = mi.Float(0.0)
        center_y = mi.Float(0.0)
        selected = mi.Bool(False)
        position = dr.select(active, position, mi.Point3f(0.0))
        for line_index, line in enumerate(self.config.types):
            if not line.active_at(depth):
                continue
            if not any(
                float(value) > 0.0 for value in line.stencil_warp_amplitude
            ):
                continue
            line_active = self._material_active(line_index, material, active)
            line_active &= self._shape_active(line_index, shape, active)
            take = line_active & (selected == False)
            warp_x, warp_y = self._stencil_warp(line, position)
            center_x = dr.select(take, warp_x, center_x)
            center_y = dr.select(take, warp_y, center_y)
            selected |= line_active
        return center_x, center_y, selected

    def _ray_flags(self, depth: int) -> int:
        """Request only SI fields consumed at this DFS level.

        ``Minimal`` supplies position and geometric normal. Normal/curvature
        metrics use the geometric normal directly; the more expensive shading
        frame is needed only when an auxiliary path is transported again.
        """
        flags = mi.RayFlags.Minimal
        needs_transport = self.config.can_apply_from(depth + 1)
        if needs_transport:
            flags |= mi.RayFlags.ShadingFrame
        return int(flags)

    def _covers_primary_sampling_domain(
        self, line_index: int, line: FeatureLineType
    ) -> bool:
        """Whether every depth-0 auxiliary offset lies in this stencil."""
        return self._primary_domain_covered[line_index]

    def _flush_trace_batch(self, pending: list[list[Any]], force: bool = False) -> None:
        """Materialize compact SI fields before transient BVH graphs pile up."""
        if pending and (force or len(pending) >= self._trace_batch_size):
            dr.eval(*(value for group in pending for value in group))
            pending.clear()

    def spawn_primary(
        self,
        scene: Any,
        sampler: Any,
        ray: Any,
        active: Any,
        base_material: Any,
        base_shape: Any = None,
        base_position: Any = None,
        detection_depth: int = 0,
    ) -> CudaAuxiliaryFrame:
        count = self.config.auxiliary_samples
        if base_position is None:
            base_position = mi.Point3f(0.0)
        center_x, center_y, centered_warp = self._primary_warp_center(
            base_material, base_shape, base_position, mi.Bool(active),
            depth=detection_depth,
        )
        maximum_radius = self._primary_sampling_radius(
            base_material,
            base_shape,
            mi.Bool(active),
            centered_warp,
            depth=detection_depth,
        )
        inner_radius = self._primary_inner_radius(
            base_material,
            base_shape,
            mi.Bool(active),
            centered_warp,
            depth=detection_depth,
        )
        anchor_radius = self._primary_anchor_radius(
            base_material, base_shape, mi.Bool(active), depth=detection_depth
        )
        has_origin_anchor = self._primary_has_origin_anchor(
            base_material, base_shape, mi.Bool(active), depth=detection_depth
        )
        inner_count = count // 2
        golden_ratio = 0.6180339887498949
        will_extend = self.config.can_apply_from(detection_depth + 1)
        keep_positions = will_extend or self.config.needs_measurement(
            detection_depth, "position"
        )
        keep_geometric_normals = will_extend or self.config.needs_measurement(
            detection_depth, "normal", "curvature"
        )
        keep_normals = will_extend
        keep_shapes = will_extend or self.config.needs_measurement(
            detection_depth, "shape_id"
        ) or any(
            line.active_at(detection_depth) and line.include_shapes
            for line in self.config.types
        )
        keep_materials = self.config.needs_measurement(
            detection_depth, "material_id"
        )
        ray_flags = self._ray_flags(detection_depth)
        offsets: list[tuple[Any, Any]] = []
        directions: list[Any] = []
        ray_origins: list[Any] = []
        positions: list[Any] = []
        geometric_normals: list[Any] = []
        shading_normals: list[Any] = []
        depths: list[Any] = []
        shapes: list[Any] = []
        materials: list[Any] = []
        prefix_valid: list[Any] = []
        pending: list[list[Any]] = []

        base_direction = mi.Vector3f(ray.d)
        base_origin = mi.Point3f(ray.o)
        has_differentials = mi.Bool(ray.has_differentials)
        direction_x = mi.Vector3f(ray.d_x - ray.d)
        direction_y = mi.Vector3f(ray.d_y - ray.d)
        origin_x = mi.Vector3f(ray.o_x - ray.o)
        origin_y = mi.Vector3f(ray.o_y - ray.o)
        fallback_first, fallback_second = _view_frame(
            _normalize(base_direction), mi.Vector3f(0.0, 1.0, 0.0)
        )
        fallback_scale = 1e-3

        for index in range(count):
            sample = sampler.next_2d()
            first_group = index < inner_count
            group_index = index if first_group else index - inner_count
            group_count = inner_count if first_group else count - inner_count
            if self.config.sampling_stencil == "disk":
                radial = (group_index + sample[0]) / group_count
                angular = dr.fma(
                    group_index, golden_ratio, sample[1] / group_count
                )
                angular -= dr.floor(angular)
                angle = 2.0 * pi * angular
                disk_radius = dr.safe_sqrt(radial)
                offset_x = disk_radius * dr.cos(angle)
                offset_y = disk_radius * dr.sin(angle)
            else:
                grid = int(ceil(sqrt(group_count)))
                stratum_x = group_index % grid
                stratum_y = group_index // grid
                unit_x = (stratum_x + sample[0]) / grid
                unit_y = (stratum_y + sample[1]) / grid
                offset_x, offset_y = 2.0 * unit_x - 1.0, 2.0 * unit_y - 1.0
            use_inner = (
                inner_radius < maximum_radius
                if first_group
                else mi.Bool(False)
            )
            sample_radius = dr.select(
                use_inner,
                dr.select(centered_warp, anchor_radius, inner_radius),
                maximum_radius,
            )
            # S1.4 mixture specialized for the Fig. 11 displaced style: the
            # inner half remains centered on the primal pixel and preserves
            # thin mechanical anchor lines; the outer half follows the shared
            # continuous displacement field and produces the bent overdraw.
            if first_group:
                use_warp_center = centered_warp & ~has_origin_anchor
                group_center_x = dr.select(use_warp_center, center_x, 0.0)
                group_center_y = dr.select(use_warp_center, center_y, 0.0)
            else:
                group_center_x = center_x
                group_center_y = center_y
            offset_x = dr.fma(offset_x, sample_radius, group_center_x)
            offset_y = dr.fma(offset_y, sample_radius, group_center_y)
            offsets.append((offset_x, offset_y))

            auxiliary = mi.Ray3f(ray)
            differential_origin = (
                base_origin + offset_x * origin_x + offset_y * origin_y
            )
            differential_direction = _normalize(
                base_direction
                + offset_x * direction_x
                + offset_y * direction_y
            )
            fallback_direction = _normalize(
                base_direction
                + fallback_scale
                * (offset_x * fallback_first + offset_y * fallback_second)
            )
            auxiliary.o = dr.select(
                has_differentials, differential_origin, base_origin
            )
            auxiliary.d = dr.select(
                has_differentials, differential_direction, fallback_direction
            )
            interaction = self._ray_intersect(
                scene, auxiliary, ray_flags, active
            )
            if will_extend:
                directions.append(mi.Vector3f(auxiliary.d))
            if keep_geometric_normals:
                geometric_normals.append(mi.Normal3f(interaction.n))
            if keep_positions:
                positions.append(mi.Point3f(interaction.p))
            if keep_normals:
                shading_normals.append(mi.Normal3f(interaction.sh_frame.n))
            depths.append(mi.Float(interaction.t))
            if keep_shapes:
                shapes.append(interaction.shape)
            if keep_materials:
                materials.append(interaction.bsdf(auxiliary))
            prefix_valid.append(mi.Bool(active))
            retained: list[Any] = [
                offset_x, offset_y, depths[-1], prefix_valid[-1]
            ]
            if will_extend:
                retained.append(directions[-1])
            if keep_geometric_normals:
                retained.append(geometric_normals[-1])
            if keep_positions:
                retained.append(positions[-1])
            if keep_normals:
                retained.append(shading_normals[-1])
            if keep_shapes:
                retained.append(shapes[-1])
            if keep_materials:
                retained.append(materials[-1])
            pending.append(retained)
            self._flush_trace_batch(pending)
        self._flush_trace_batch(pending, force=True)
        return CudaAuxiliaryFrame(
            offsets=offsets,
            directions=directions,
            ray_origins=ray_origins,
            positions=positions,
            geometric_normals=geometric_normals,
            shading_normals=shading_normals,
            depths=depths,
            shapes=shapes,
            materials=materials,
            prefix_valid=prefix_valid,
            distributional=mi.Bool(False),
            time=mi.Float(ray.time),
            wavelengths=ray.wavelengths,
        ).eval()

    def local_ray_differential(
        self,
        ray: Any,
        frame: CudaAuxiliaryFrame,
        active: Any,
    ) -> Any:
        """Fit the reflected pixel footprint carried by transported paths.

        The transported auxiliary origins and directions already encode the
        real camera-to-mirror mapping. A two-variable least-squares fit turns
        that footprint into a RayDifferential at the reflected vertex. The
        subsequent material-local stencil therefore retains perspective and
        mirror distortion without inheriting another robot's support radius.
        """
        differential = mi.RayDifferential3f(ray)
        count = min(
            len(frame.offsets),
            len(frame.directions),
            len(frame.ray_origins),
            len(frame.prefix_valid),
        )
        if count < 2:
            first, second = _view_frame(
                _normalize(mi.Vector3f(ray.d)), mi.Vector3f(0.0, 1.0, 0.0)
            )
            differential.o_x = mi.Point3f(ray.o)
            differential.o_y = mi.Point3f(ray.o)
            differential.d_x = _normalize(mi.Vector3f(ray.d) + 1e-3 * first)
            differential.d_y = _normalize(mi.Vector3f(ray.d) + 1e-3 * second)
            differential.has_differentials = True
            return differential

        sxx = mi.Float(0.0)
        sxy = mi.Float(0.0)
        syy = mi.Float(0.0)
        direction_x = mi.Vector3f(0.0)
        direction_y = mi.Vector3f(0.0)
        origin_x = mi.Vector3f(0.0)
        origin_y = mi.Vector3f(0.0)
        for index in range(count):
            offset_x, offset_y = frame.offsets[index]
            weight = dr.select(active & frame.prefix_valid[index], 1.0, 0.0)
            weighted_x = weight * offset_x
            weighted_y = weight * offset_y
            direction_delta = frame.directions[index] - ray.d
            origin_delta = frame.ray_origins[index] - ray.o
            sxx += weighted_x * offset_x
            sxy += weighted_x * offset_y
            syy += weighted_y * offset_y
            direction_x += weighted_x * direction_delta
            direction_y += weighted_y * direction_delta
            origin_x += weighted_x * origin_delta
            origin_y += weighted_y * origin_delta

        determinant = sxx * syy - dr.square(sxy)
        fit_valid = active & dr.isfinite(determinant) & (determinant > 1e-10)
        inverse = dr.rcp(dr.maximum(determinant, 1e-10))
        fitted_direction_x = (
            syy * direction_x - sxy * direction_y
        ) * inverse
        fitted_direction_y = (
            sxx * direction_y - sxy * direction_x
        ) * inverse
        fitted_origin_x = (syy * origin_x - sxy * origin_y) * inverse
        fitted_origin_y = (sxx * origin_y - sxy * origin_x) * inverse

        fallback_x, fallback_y = _view_frame(
            _normalize(mi.Vector3f(ray.d)), mi.Vector3f(0.0, 1.0, 0.0)
        )
        fitted_direction_x = dr.select(
            fit_valid, fitted_direction_x, 1e-3 * fallback_x
        )
        fitted_direction_y = dr.select(
            fit_valid, fitted_direction_y, 1e-3 * fallback_y
        )
        fitted_origin_x = dr.select(fit_valid, fitted_origin_x, 0.0)
        fitted_origin_y = dr.select(fit_valid, fitted_origin_y, 0.0)
        differential.o_x = ray.o + fitted_origin_x
        differential.o_y = ray.o + fitted_origin_y
        differential.d_x = _normalize(ray.d + fitted_direction_x)
        differential.d_y = _normalize(ray.d + fitted_direction_y)
        differential.has_differentials = True
        return differential

    @staticmethod
    def _stencil_warp(
        line: FeatureLineType, position: Any
    ) -> tuple[Any, Any]:
        displacement = []
        for component in range(2):
            axis = line.stencil_warp_axes[component]
            coordinate = (
                float(axis[0]) * position.x
                + float(axis[1]) * position.y
                + float(axis[2]) * position.z
            )
            phase = (
                float(line.stencil_warp_frequency[component]) * coordinate
                + float(line.stencil_warp_phase[component])
            )
            if line.stencil_warp_profile == "ripple":
                waveform = (
                    0.82 * dr.sin(phase)
                    + 0.18 * dr.sin(2.17 * phase + 0.55)
                )
            elif line.stencil_warp_profile == "sketchy":
                waveform = (
                    0.60 * dr.sin(phase)
                    + 0.25 * dr.sin(1.91 * phase + 0.73)
                    + 0.15 * dr.sin(3.17 * phase - 1.11)
                )
            else:
                waveform = dr.sin(phase)
            displacement.append(
                float(line.stencil_warp_amplitude[component]) * waveform
            )
        return displacement[0], displacement[1]

    @staticmethod
    def _stencil_center(
        line: FeatureLineType, position: Any
    ) -> tuple[Any, Any]:
        warp_x, warp_y = CudaFeatureLineTracer._stencil_warp(line, position)
        return (
            float(line.stencil_offset[0]) + warp_x,
            float(line.stencil_offset[1]) + warp_y,
        )

    @staticmethod
    def _line_hatch_weight(line: FeatureLineType, position: Any) -> Any:
        """Object-space hatch coverage applied after edge detection."""
        if line.line_hatch_scale <= 0.0:
            return mi.Float(1.0)
        direction = line.line_hatch_direction
        phase = float(line.line_hatch_scale) * (
            float(direction[0]) * position.x
            + float(direction[1]) * position.y
            + float(direction[2]) * position.z
        ) + float(line.line_hatch_phase)
        phase -= dr.floor(phase)
        distance = dr.minimum(phase, 1.0 - phase)
        softness = float(line.line_hatch_edge_softness)
        if softness <= 0.0:
            return dr.select(distance < float(line.line_hatch_width), 1.0, 0.0)
        weight = dr.clamp(
            (float(line.line_hatch_width) + softness - distance)
            / (2.0 * softness),
            0.0,
            1.0,
        )
        return weight * weight * (3.0 - 2.0 * weight)

    @staticmethod
    def _inside(
        line: FeatureLineType,
        x: Any,
        y: Any,
        center: tuple[Any, Any] | None = None,
    ) -> Any:
        if center is None:
            center = (
                float(line.stencil_offset[0]),
                float(line.stencil_offset[1]),
            )
        x -= center[0]
        y -= center[1]
        if line.stencil == "square":
            return dr.maximum(dr.abs(x), dr.abs(y)) <= line.stencil_radius
        return dr.square(x) + dr.square(y) <= line.stencil_radius**2

    def _pair_available(
        self,
        line: FeatureLineType,
        first_index: int,
        second_index: int,
        offsets_x: list[Any],
        offsets_y: list[Any],
        available: list[Any],
        inside: list[Any] | None = None,
    ) -> Any:
        """Whether a pair belongs to this line type's sampling domain.

        Supplemental S1.4 draws one bounding stencil for all line types, but
        explicitly ignores samples outside each type's own stencil.  Such a
        pair must therefore not consume one of the type's ``n`` searches.
        """
        first_x = offsets_x[first_index]
        first_y = offsets_y[first_index]
        second_x = offsets_x[second_index]
        second_y = offsets_y[second_index]
        distance_squared = (
            dr.square(second_x - first_x)
            + dr.square(second_y - first_y)
        )
        first_inside = (
            inside[first_index]
            if inside is not None
            else self._inside(line, first_x, first_y)
        )
        second_inside = (
            inside[second_index]
            if inside is not None
            else self._inside(line, second_x, second_y)
        )
        return (
            available[first_index]
            & available[second_index]
            & first_inside
            & second_inside
            & (distance_squared > 1e-12)
        )

    def _pair_trigger(
        self,
        line: FeatureLineType,
        first_index: int,
        second_index: int,
        offsets_x: list[Any],
        offsets_y: list[Any],
        available: list[Any],
        hits: list[Any],
        depths: list[Any],
        normals: list[Any],
        positions: list[Any],
        shapes: list[Any],
        materials: list[Any],
        pair_active: Any | None = None,
        inside: list[Any] | None = None,
    ) -> Any:
        first_x = offsets_x[first_index]
        first_y = offsets_y[first_index]
        second_x = offsets_x[second_index]
        second_y = offsets_y[second_index]
        if pair_active is None:
            pair_active = self._pair_available(
                line,
                first_index,
                second_index,
                offsets_x,
                offsets_y,
                available,
                inside,
            )
        first_hit = hits[first_index]
        second_hit = hits[second_index]
        mismatch = first_hit != second_hit
        if not line.include_silhouette:
            mismatch = mi.Bool(False)
        both_hit = first_hit & second_hit

        if line.measurement == "silhouette":
            # Hit/miss mismatch above is the complete silhouette metric. Two
            # surface hits must never turn an internal depth/normal edge into
            # a wide outer contour.
            threshold_crossed = mi.Bool(False)
        elif line.measurement == "depth":
            distance = dr.sqrt(
                dr.square(second_x - first_x)
                + dr.square(second_y - first_y)
            )
            first = depths[first_index]
            second = depths[second_index]
            difference = dr.abs(second - first)
            if line.relative_depth:
                difference /= dr.maximum(
                    dr.minimum(dr.abs(first), dr.abs(second)), 1e-6
                )
            slope = difference / distance
            threshold_crossed = slope >= line.threshold
        elif line.measurement == "normal":
            distance = dr.sqrt(
                dr.square(second_x - first_x)
                + dr.square(second_y - first_y)
            )
            first = _normalize(normals[first_index])
            second = _normalize(normals[second_index])
            difference = dr.norm(second - first)
            if line.normal_orientation_invariant:
                difference = dr.minimum(difference, dr.norm(second + first))
            slope = difference / distance
            threshold_crossed = slope >= line.threshold
            if (
                line.normal_shape_boundary_fallback
                and first_index < len(shapes)
                and second_index < len(shapes)
            ):
                # Decorative coplanar pieces can have identical geometric
                # normals. Close only the explicitly opted-in seam after the
                # normal finite-difference test; this is not enabled for any
                # body/outline dictionary.
                threshold_crossed |= (
                    both_hit & (shapes[first_index] != shapes[second_index])
                )
        elif line.measurement == "curvature":
            distance = dr.sqrt(
                dr.square(second_x - first_x)
                + dr.square(second_y - first_y)
            )
            first = normals[first_index]
            second = normals[second_index]
            angle = dr.acos(dr.clamp(dr.dot(first, second), -1.0, 1.0))
            threshold_crossed = angle / distance >= line.threshold
        elif line.measurement == "position":
            distance = dr.sqrt(
                dr.square(second_x - first_x)
                + dr.square(second_y - first_y)
            )
            first = positions[first_index]
            second = positions[second_index]
            threshold_crossed = dr.norm(second - first) / distance >= line.threshold
        elif line.measurement == "shape_id":
            threshold_crossed = (
                shapes[first_index] != shapes[second_index]
            )
        else:
            threshold_crossed = (
                materials[first_index] != materials[second_index]
            )
        return pair_active & (mismatch | (both_hit & threshold_crossed))

    def detect(
        self,
        base_interaction: Any,
        base_ray: Any,
        frame: CudaAuxiliaryFrame,
        sampler: Any,
        depth: int,
        active: Any,
    ) -> tuple[Any, Any, list[Any]]:
        del sampler
        zero = mi.Float(0.0)
        offsets_x = [zero] + [offset[0] for offset in frame.offsets]
        offsets_y = [zero] + [offset[1] for offset in frame.offsets]
        available = [mi.Bool(active)] + frame.prefix_valid
        hits = [active & base_interaction.is_valid()] + [
            valid & dr.isfinite(depth)
            for valid, depth in zip(frame.prefix_valid, frame.depths)
        ]
        active_line_indices = [
            index for index, line in enumerate(self.config.types)
            if line.active_at(depth)
        ]
        will_extend = self.config.can_apply_from(depth + 1)
        needs_depths = self.config.needs_measurement(depth, "depth")
        needs_normals = will_extend or self.config.needs_measurement(
            depth, "normal", "curvature"
        )
        needs_positions = self.config.needs_measurement(depth, "position")
        needs_shapes = will_extend or self.config.needs_measurement(
            depth, "shape_id"
        ) or any(
            line.active_at(depth) and line.include_shapes
            for line in self.config.types
        )
        needs_materials = self.config.needs_measurement(depth, "material_id")
        depths = [base_interaction.t, *frame.depths] if needs_depths else []
        # Eq. (21) must be evaluated on the *geometric* normal field.  Never
        # fall back to ``sh_frame.n`` here: Blender's interpolated shading
        # normals can vary inside an otherwise planar face and would turn
        # those interpolation gradients into false feature lines.  Shading
        # normals are retained separately only for parallel transport below.
        geometric_available = (
            len(frame.geometric_normals) == len(frame.depths)
        )
        if needs_normals and not geometric_available:
            raise RuntimeError(
                "Feature-line normal finite differences require a complete "
                "geometric-normal frame"
            )
        normals = (
            [base_interaction.n, *frame.geometric_normals]
            if needs_normals else []
        )
        positions = (
            [base_interaction.p, *frame.positions] if needs_positions else []
        )
        shapes = (
            [base_interaction.shape, *frame.shapes] if needs_shapes else []
        )
        base_material = base_interaction.bsdf(base_ray)
        materials = (
            [base_material, *frame.materials] if needs_materials else []
        )

        line_hit = mi.Bool(False)
        line_color = mi.Color3f(0.0)
        # Several Fig. 11 dictionaries share the same stencil. Reuse their
        # per-sample inclusion masks instead of generating duplicate CUDA DAGs.
        stencil_cache: dict[tuple[Any, ...], list[Any]] = {}
        line_inside: dict[int, list[Any]] = {}
        for line_index in active_line_indices:
            line = self.config.types[line_index]
            key = (
                line.stencil,
                line.stencil_warp_profile,
                float(line.stencil_radius),
                float(line.stencil_offset[0]),
                float(line.stencil_offset[1]),
                *tuple(float(value) for value in line.stencil_warp_amplitude),
                *tuple(float(value) for value in line.stencil_warp_frequency),
                *tuple(float(value) for value in line.stencil_warp_phase),
                *tuple(float(value) for value in line.stencil_warp_axes.reshape(-1)),
            )
            inside = stencil_cache.get(key)
            if inside is None:
                center = self._stencil_center(line, base_interaction.p)
                inside = [
                    self._inside(line, offset_x, offset_y, center)
                    for offset_x, offset_y in zip(offsets_x, offsets_y)
                ]
                stencil_cache[key] = inside
            line_inside[line_index] = inside

        material_mask_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], Any] = {}
        line_material_masks: dict[int, Any] = {}
        shape_mask_cache: dict[tuple[str, ...], Any] = {}
        line_shape_masks: dict[int, Any] = {}
        for line_index in active_line_indices:
            line = self.config.types[line_index]
            key = (line.include_materials, line.exclude_materials)
            mask = material_mask_cache.get(key)
            if mask is None:
                mask = self._material_active(
                    line_index, base_material, mi.Bool(True)
                )
                material_mask_cache[key] = mask
            line_material_masks[line_index] = mask
            key = self.config.types[line_index].include_shapes
            shape_mask = shape_mask_cache.get(key)
            if shape_mask is None:
                shape_mask = self._shape_active(
                    line_index, base_interaction.shape, mi.Bool(True)
                )
                shape_mask_cache[key] = shape_mask
            line_shape_masks[line_index] = shape_mask

        for line_position, line_index in enumerate(active_line_indices, start=1):
            line = self.config.types[line_index]
            ordered_pairs = self._pair_orders[line_index]
            if depth == 0 and self._covers_primary_sampling_domain(
                line_index, line
            ):
                # At the camera vertex all prefixes are valid. Once the whole
                # bounding stencil is covered, the first n permuted pairs are
                # exactly the n searches the general scan would select.
                ordered_pairs = ordered_pairs[:line.comparisons]
            inside = line_inside[line_index]
            found = mi.Bool(False)
            searches = mi.UInt32(0)
            line_active = (
                active & (line_hit == False) & line_material_masks[line_index]
                & line_shape_masks[line_index]
            )
            # Material-restricted dictionaries (Fig. 13) are absent from most
            # background, floor, and mirror tiles. Avoid constructing their
            # complete 136-pair CUDA DAG when no lane can possibly select the
            # line. This host query is performed once per dictionary and does
            # not change masks, samples, pair order, or continuation state.
            if line.include_materials and not bool(dr.any(line_active)):
                continue
            # A line type with a smaller stencil ignores out-of-domain pairs;
            # they do not consume one of Eq. (29)'s n metric evaluations.
            # Scan the complete low-discrepancy pair permutation so each lane
            # receives up to ``comparisons`` valid searches from the samples
            # actually available in its stencil (Supplemental S1.4).
            for first, second in ordered_pairs:
                pair_active = self._pair_available(
                    line,
                    first,
                    second,
                    offsets_x,
                    offsets_y,
                    available,
                    inside,
                )
                take = (
                    pair_active
                    & line_active
                    & (found == False)
                    & (searches < line.comparisons)
                )
                triggered = self._pair_trigger(
                    line,
                    first,
                    second,
                    offsets_x,
                    offsets_y,
                    available,
                    hits,
                    depths,
                    normals,
                    positions,
                    shapes,
                    materials,
                    pair_active=take,
                    inside=inside,
                )
                found |= triggered
                searches += dr.select(take, 1, 0)
            hatch_weight = self._line_hatch_weight(line, base_interaction.p)
            selected = (
                active & (line_hit == False) & found & (hatch_weight > 0.0)
            )
            selected_color = dr.select(
                getattr(frame, "distributional", mi.Bool(False)),
                dr.lerp(
                    mi.Color3f(line.color_at(depth)),
                    mi.Color3f(self.config.glossy_line_color),
                    line.glossy_strength(self.config.glossy_line_strength),
                ),
                mi.Color3f(line.color_at(depth)),
            ) * hatch_weight
            line_color = dr.select(
                selected,
                selected_color,
                line_color,
            )
            line_hit |= selected
            # Preserve dictionary priority exactly. Materialize periodically
            # rather than after every dictionary: this substantially reduces
            # CUDA launch/synchronization overhead for Fig. 11's 16 line
            # dictionaries while keeping the temporary DAG bounded.
            if (
                line_position % self.config.cuda_line_eval_interval == 0
                or line_position == len(active_line_indices)
            ):
                dr.eval(line_color, line_hit)

        # At the last configured line depth no child will consume an
        # AuxiliaryPathFrame. Avoid the otherwise redundant N_aux × line-type
        # pruning pass altogether.
        if not self.config.can_apply_from(depth + 1):
            return line_color, line_hit, []

        # Auxiliary paths that already crossed a feature are excluded at
        # deeper vertices, as described in Supplemental S1.5.
        continuation: list[Any] = []
        base_hit = active & (line_hit == False) & base_interaction.is_valid()
        for auxiliary_index, (auxiliary_depth, geometric_normal,
                              auxiliary_shape, prefix) in enumerate(
            zip(
                frame.depths,
                frame.geometric_normals,
                frame.shapes,
                frame.prefix_valid,
            ),
            start=1,
        ):
            aux_hit = dr.isfinite(auxiliary_depth)
            # This is the continuation/pruning part of feature-line
            # detection, so its normal-angle test must use geometric normals
            # as well. The shading-normal cache remains exclusively for
            # transporting the BSDF half-vector in ``_extend``.
            normal_dot = dr.dot(base_interaction.n, geometric_normal)
            valid = (
                prefix
                & base_hit
                & aux_hit
                & (base_interaction.shape == auxiliary_shape)
                & (normal_dot >= self.normal_cosine_limit)
            )
            for line_index in active_line_indices:
                line = self.config.types[line_index]
                inside = line_inside[line_index]
                pair_active = valid & line_material_masks[line_index] & line_shape_masks[line_index] & self._pair_available(
                    line,
                    0,
                    auxiliary_index,
                    offsets_x,
                    offsets_y,
                    available,
                    inside,
                )
                triggered = self._pair_trigger(
                    line,
                    0,
                    auxiliary_index,
                    offsets_x,
                    offsets_y,
                    available,
                    hits,
                    depths,
                    normals,
                    positions,
                    shapes,
                    materials,
                    pair_active=pair_active,
                    inside=inside,
                )
                valid &= ~(
                    triggered
                    & (self._line_hatch_weight(line, base_interaction.p) > 0.0)
                )
            continuation.append(valid)
        # Continuation is the only detection state consumed by the next DFS
        # edge. Do not retain the finite-difference graph behind it.
        if continuation:
            dr.eval(*continuation)
        return line_color, line_hit, continuation

    def extend(
        self,
        scene: Any,
        frame: CudaAuxiliaryFrame,
        continuation: list[Any],
        base_normal: Any,
        base_incoming_direction: Any,
        base_outgoing: Any,
        sampled_type: Any,
        sampled_eta: Any,
        active: Any,
        child_depth: int,
    ) -> CudaAuxiliaryFrame:
        normal = _normalize(base_normal)
        view = _normalize(-base_incoming_direction)
        outgoing = _normalize(base_outgoing)
        reflected_half = _normalize(view + outgoing)
        transmitted_half = _normalize(view + sampled_eta * outgoing)
        half_vector = dr.select(
            mi.has_flag(sampled_type, mi.BSDFFlags.Transmission),
            transmitted_half,
            reflected_half,
        )
        half_vector = dr.select(
            dr.dot(half_vector, normal) < 0.0, -half_vector, half_vector
        )
        is_conditioned = (
            mi.has_flag(sampled_type, mi.BSDFFlags.Glossy)
            | mi.has_flag(sampled_type, mi.BSDFFlags.Delta)
        )
        is_transmission = mi.has_flag(
            sampled_type, mi.BSDFFlags.Transmission
        )
        will_extend = self.config.can_apply_from(child_depth + 1)
        keep_rays_for_local_resampling = (
            self.config.resample_delta_reflections and child_depth > 0
        )
        keep_directions = will_extend or keep_rays_for_local_resampling
        active_types = self.config.active_types(child_depth)
        keep_positions = will_extend or any(
            line.measurement == "position" for line in active_types
        )
        keep_geometric_normals = will_extend or any(
            line.measurement in {"normal", "curvature"}
            for line in active_types
        )
        keep_normals = will_extend
        keep_shapes = will_extend or self.config.needs_measurement(
            child_depth, "shape_id"
        )
        keep_materials = any(
            line.measurement == "material_id" for line in active_types
        )
        child_ray_flags = self._ray_flags(child_depth)

        new_directions: list[Any] = []
        new_ray_origins: list[Any] = []
        new_positions: list[Any] = []
        new_geometric_normals: list[Any] = []
        new_shading_normals: list[Any] = []
        new_depths: list[Any] = []
        new_shapes: list[Any] = []
        new_materials: list[Any] = []
        new_prefix_valid: list[Any] = []
        pending: list[list[Any]] = []
        for direction, position, geometric_normal, auxiliary_normal, prefix in zip(
            frame.directions,
            frame.positions,
            frame.geometric_normals,
            frame.shading_normals,
            continuation,
        ):
            edge_active = active & prefix
            auxiliary_normal = _normalize(auxiliary_normal)
            auxiliary_view = _normalize(-direction)
            auxiliary_half = _transport_half_vector(
                half_vector,
                normal,
                auxiliary_normal,
                view,
                auxiliary_view,
            )
            reflected = (
                2.0 * dr.dot(auxiliary_view, auxiliary_half) * auxiliary_half
                - auxiliary_view
            )
            cosine_incident = dr.clamp(
                dr.dot(auxiliary_view, auxiliary_half), 0.0, 1.0
            )
            eta = dr.maximum(sampled_eta, 1e-8)
            transmission_discriminant = (
                1.0
                - (1.0 - dr.square(cosine_incident)) / dr.square(eta)
            )
            cosine_transmitted = dr.safe_sqrt(transmission_discriminant)
            transmitted = (
                -auxiliary_view / eta
                + (cosine_incident / eta - cosine_transmitted) * auxiliary_half
            )
            conditioned = dr.select(is_transmission, transmitted, reflected)
            fallback = _minimal_rotation(outgoing, normal, auxiliary_normal)
            auxiliary_outgoing = _normalize(
                dr.select(is_conditioned, conditioned, fallback)
            )
            direction_valid = _finite_vector(auxiliary_outgoing) & (
                (is_transmission == False) | (transmission_discriminant >= 0.0)
            )
            interaction = mi.Interaction3f()
            interaction.p = position
            interaction.n = geometric_normal
            interaction.time = frame.time
            interaction.wavelengths = frame.wavelengths
            auxiliary_ray = interaction.spawn_ray(auxiliary_outgoing)
            child_prefix = edge_active & direction_valid
            child_interaction = self._ray_intersect(
                scene, auxiliary_ray, child_ray_flags, child_prefix
            )
            if keep_directions:
                new_directions.append(mi.Vector3f(auxiliary_ray.d))
                new_ray_origins.append(mi.Point3f(auxiliary_ray.o))
            if keep_geometric_normals:
                new_geometric_normals.append(mi.Normal3f(child_interaction.n))
            if keep_positions:
                new_positions.append(mi.Point3f(child_interaction.p))
            if keep_normals:
                new_shading_normals.append(
                    mi.Normal3f(child_interaction.sh_frame.n)
                )
            new_depths.append(mi.Float(child_interaction.t))
            if keep_shapes:
                new_shapes.append(child_interaction.shape)
            if keep_materials:
                new_materials.append(child_interaction.bsdf(auxiliary_ray))
            new_prefix_valid.append(child_prefix)
            retained: list[Any] = [new_depths[-1], new_prefix_valid[-1]]
            if keep_directions:
                retained.extend((new_directions[-1], new_ray_origins[-1]))
            if keep_geometric_normals:
                retained.append(new_geometric_normals[-1])
            if keep_positions:
                retained.append(new_positions[-1])
            if keep_normals:
                retained.append(new_shading_normals[-1])
            if keep_shapes:
                retained.append(new_shapes[-1])
            if keep_materials:
                retained.append(new_materials[-1])
            pending.append(retained)
            self._flush_trace_batch(pending)
        self._flush_trace_batch(pending, force=True)
        distributional = frame.distributional | (
            active
            & mi.has_flag(sampled_type, mi.BSDFFlags.Glossy)
            & ~mi.has_flag(sampled_type, mi.BSDFFlags.Delta)
        )
        return CudaAuxiliaryFrame(
            offsets=frame.offsets,
            directions=new_directions,
            ray_origins=new_ray_origins,
            positions=new_positions,
            geometric_normals=new_geometric_normals,
            shading_normals=new_shading_normals,
            depths=new_depths,
            shapes=new_shapes,
            materials=new_materials,
            prefix_valid=new_prefix_valid,
            distributional=distributional,
            time=frame.time,
            wavelengths=frame.wavelengths,
        ).eval()


__all__ = ["CudaAuxiliaryFrame", "CudaFeatureLineTracer"]

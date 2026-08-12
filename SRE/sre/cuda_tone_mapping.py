"""CUDA implementation of the canonical tone-coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Any

import drjit as dr
import mitsuba as mi

try:
    from .cuda_backend import gather_lanes
    from .tone_mapping import ToneMappingConfig, multiscale_anchor_offsets
except ImportError:
    from cuda_backend import gather_lanes
    from tone_mapping import ToneMappingConfig, multiscale_anchor_offsets


def _normalize(value: Any) -> Any:
    return value * dr.rsqrt(dr.maximum(dr.squared_norm(value), 1e-20))


def _signed_nonzero(value: Any) -> Any:
    return dr.select(
        dr.abs(value) > 1e-20,
        value,
        dr.select(value < 0.0, -1e-20, 1e-20),
    )


def _plane_basis(normal: Any) -> tuple[Any, Any]:
    normal = _normalize(normal)
    axis = dr.select(
        dr.abs(normal[0]) > 0.8,
        mi.Vector3f(0.0, 1.0, 0.0),
        mi.Vector3f(1.0, 0.0, 0.0),
    )
    first = _normalize(dr.cross(normal, axis))
    second = _normalize(dr.cross(normal, first))
    return first, second


@dataclass
class CudaToneFrame:
    # Every anchor coordinate is ``base_coordinate + constant offset``.
    # Storing the expanded pair for all N anchors retained 2*N full-width
    # buffers at every DFS level.  The offsets already live as host constants
    # on CudaToneMapper, so only this single pair belongs in device memory.
    base_coordinate: tuple[Any, Any]
    directions: list[Any]
    positions: list[Any]
    geometric_normals: list[Any]
    shapes: list[Any]
    valid: list[Any]
    time: Any
    wavelengths: Any
    # Supplemental S2.3 prepass cache: the material-independent perfect-
    # mirror projection frame at the next path depth. It contains device
    # arrays but no back-reference, so the chain has bounded depth.
    next_frame: Any = None

    def values(self) -> list[Any]:
        """Return the compact Dr.Jit arrays retained by this frame."""
        values: list[Any] = [
            self.time, self.base_coordinate[0], self.base_coordinate[1]
        ]
        for sequence in (
            self.directions,
            self.positions,
            self.geometric_normals,
            self.shapes,
            self.valid,
        ):
            values.extend(sequence)
        return values

    def eval(self) -> "CudaToneFrame":
        dr.eval(*self.values())
        return self

    def gather(self, indices: Any) -> "CudaToneFrame":
        parent_width = max(
            (dr.width(value) for value in self.values()), default=1
        )
        return CudaToneFrame(
            base_coordinate=(
                gather_lanes(mi.Float, self.base_coordinate[0], indices,
                             parent_width, "tone base coordinate x"),
                gather_lanes(mi.Float, self.base_coordinate[1], indices,
                             parent_width, "tone base coordinate y"),
            ),
            directions=[
                gather_lanes(mi.Vector3f, direction, indices, parent_width,
                             "tone anchor direction")
                for direction in self.directions
            ],
            positions=[
                gather_lanes(mi.Point3f, position, indices, parent_width,
                             "tone anchor position")
                for position in self.positions
            ],
            geometric_normals=[
                gather_lanes(mi.Normal3f, normal, indices, parent_width,
                             "tone anchor normal")
                for normal in self.geometric_normals
            ],
            shapes=[
                gather_lanes(mi.ShapePtr, shape, indices, parent_width,
                             "tone anchor shape") for shape in self.shapes
            ],
            valid=[
                gather_lanes(mi.Bool, valid, indices, parent_width,
                             "tone anchor mask") for valid in self.valid
            ],
            time=gather_lanes(
                mi.Float, self.time, indices, parent_width, "tone time"
            ),
            wavelengths=self.wavelengths,
            next_frame=(
                self.next_frame.gather(indices)
                if self.next_frame is not None else None
            ),
        )


class CudaToneMapper:
    """Streaming mirror anchors and view-oriented linear MLS on the device."""

    def __init__(self, config: ToneMappingConfig) -> None:
        self.config = config
        self.offsets = multiscale_anchor_offsets(config)
        self._trace_batch_size = min(
            config.cuda_anchor_batch_size, config.anchor_samples
        )
        self._sensor: Any = None
        self._to_camera: Any = None
        self._to_world: Any = None
        self._focus_distance = 1.0
        self._film_width = 1.0
        self._film_height = 1.0
        self._pixel_scale = 1.0
        self._angular_cosine = cos(radians(config.angular_limit_degrees))
        self._intersector: Any = None

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
        if self._sensor is not None:
            return
        sensors = scene.sensors()
        if not sensors:
            raise ValueError("tone mapping requires a projective camera")
        self._sensor = sensors[0]
        self._to_world = self._sensor.world_transform()
        self._to_camera = self._to_world.inverse()
        focus_distance = self._sensor.focus_distance()
        self._focus_distance = float(
            dr.slice(focus_distance)
            if dr.width(focus_distance) > 0 else focus_distance
        )
        size = self._sensor.film().size()
        self._film_width = float(size[0])
        self._film_height = float(size[1])
        self._pixel_scale = (
            self._film_width / self.config.reference_width
            if self.config.reference_width > 0.0 else 1.0
        )

    def _local_focus_point(self, origin: Any, direction: Any) -> Any:
        local_origin = mi.Point3f(self._to_camera @ origin)
        local_direction = mi.Vector3f(self._to_camera @ direction)
        distance = (
            self._focus_distance - local_origin[2]
        ) / _signed_nonzero(local_direction[2])
        return local_origin + distance * local_direction

    def primary_coordinate(self, ray: Any) -> Any:
        focus = self._local_focus_point(ray.o, ray.d)
        focus_x = self._local_focus_point(ray.o_x, ray.d_x)
        focus_y = self._local_focus_point(ray.o_y, ray.d_y)
        inverse_focus = 1.0 / max(self._focus_distance, 1e-20)
        quotient_x = focus[0] * inverse_focus
        quotient_y = focus[1] * inverse_focus
        delta_x = (focus_x[0] - focus[0]) * inverse_focus
        delta_y = (focus_y[1] - focus[1]) * inverse_focus
        return mi.Point2f(
            0.5 * self._film_width + quotient_x / _signed_nonzero(delta_x),
            0.5 * self._film_height + quotient_y / _signed_nonzero(delta_y),
        ) / self._pixel_scale

    def _flush(self, pending: list[list[Any]], force: bool = False) -> None:
        if pending and (force or len(pending) >= self._trace_batch_size):
            dr.eval(*(value for group in pending for value in group))
            pending.clear()

    def spawn_primary(self, scene: Any, ray: Any, active: Any) -> CudaToneFrame:
        self.prepare(scene)
        active = mi.Bool(active)
        base_coordinate = self.primary_coordinate(ray)
        local_origin = mi.Point3f(self._to_camera @ ray.o)
        focus = self._local_focus_point(ray.o, ray.d)
        focus_x = self._local_focus_point(ray.o_x, ray.d_x)
        focus_y = self._local_focus_point(ray.o_y, ray.d_y)
        delta_x = focus_x - focus
        delta_y = focus_y - focus

        directions: list[Any] = []
        positions: list[Any] = []
        normals: list[Any] = []
        shapes: list[Any] = []
        valid: list[Any] = []
        pending: list[list[Any]] = []
        ray_flags = int(mi.RayFlags.Minimal)

        for offset in self.offsets:
            offset_x = self._pixel_scale * float(offset[0])
            offset_y = self._pixel_scale * float(offset[1])
            target = focus + offset_x * delta_x + offset_y * delta_y
            local_direction = _normalize(target - local_origin)
            world_direction = _normalize(
                mi.Vector3f(self._to_world @ local_direction)
            )
            auxiliary = mi.Ray3f(ray)
            auxiliary.o = mi.Point3f(ray.o)
            auxiliary.d = world_direction
            interaction = self._ray_intersect(
                scene, auxiliary, ray_flags, active
            )
            directions.append(mi.Vector3f(world_direction))
            positions.append(mi.Point3f(interaction.p))
            normals.append(mi.Normal3f(interaction.n))
            shapes.append(interaction.shape)
            valid.append(active & interaction.is_valid())
            pending.append([
                directions[-1], positions[-1], normals[-1], shapes[-1],
                valid[-1],
            ])
            self._flush(pending)
        self._flush(pending, force=True)
        return CudaToneFrame(
            base_coordinate=(
                mi.Float(base_coordinate[0]), mi.Float(base_coordinate[1])
            ),
            directions=directions,
            positions=positions,
            geometric_normals=normals,
            shapes=shapes,
            valid=valid,
            time=mi.Float(ray.time),
            wavelengths=ray.wavelengths,
        ).eval()

    def precompute_projection_chain(
        self,
        scene: Any,
        frame: CudaToneFrame,
    ) -> CudaToneFrame:
        """Build Supplemental S2.3's mirror-anchor cache once per tile.

        Every anchor follows a material-independent perfect-reflection path.
        The resulting linked frames are shared by all descendant SRE draws;
        query-specific surface-prefix compatibility is applied later using
        masks only, without tracing the same anchor edges again.
        """
        current = frame
        for _ in range(1, self.config.max_depth):
            following = self._trace_projection_edge(scene, current)
            current.next_frame = following
            current = following
        return frame

    def _trace_projection_edge(
        self,
        scene: Any,
        frame: CudaToneFrame,
    ) -> CudaToneFrame:
        """Trace one cached perfect-mirror edge for every anchor."""
        new_directions: list[Any] = []
        new_positions: list[Any] = []
        new_normals: list[Any] = []
        new_shapes: list[Any] = []
        new_valid: list[Any] = []
        pending: list[list[Any]] = []
        ray_flags = int(mi.RayFlags.Minimal)

        for direction, position, normal, valid in zip(
            frame.directions,
            frame.positions,
            frame.geometric_normals,
            frame.valid,
        ):
            normal = _normalize(normal)
            reflected = _normalize(
                direction - 2.0 * dr.dot(direction, normal) * normal
            )
            finite = (
                dr.isfinite(reflected[0])
                & dr.isfinite(reflected[1])
                & dr.isfinite(reflected[2])
            )
            prefix = valid & finite
            source = mi.Interaction3f()
            source.p = position
            source.n = normal
            source.time = frame.time
            source.wavelengths = frame.wavelengths
            child_ray = source.spawn_ray(reflected)
            child = self._ray_intersect(
                scene, child_ray, ray_flags, prefix
            )
            new_directions.append(mi.Vector3f(child_ray.d))
            new_positions.append(mi.Point3f(child.p))
            new_normals.append(mi.Normal3f(child.n))
            new_shapes.append(child.shape)
            new_valid.append(prefix & child.is_valid())
            pending.append([
                new_directions[-1],
                new_positions[-1],
                new_normals[-1],
                new_shapes[-1],
                new_valid[-1],
            ])
            self._flush(pending)
        self._flush(pending, force=True)
        return CudaToneFrame(
            base_coordinate=frame.base_coordinate,
            directions=new_directions,
            positions=new_positions,
            geometric_normals=new_normals,
            shapes=new_shapes,
            valid=new_valid,
            time=frame.time,
            wavelengths=frame.wavelengths,
        ).eval()

    def query(
        self,
        frame: CudaToneFrame,
        interaction: Any,
        ray: Any,
        depth: int,
        active: Any,
        analytic: Any = False,
    ) -> tuple[Any, Any, Any, Any]:
        """Return coordinate, availability, confidence and fallback method."""

        active = mi.Bool(active)
        fallback = mi.Point2f(*frame.base_coordinate)
        if depth == 0:
            return fallback, active, dr.select(active, 1.0, 0.0), mi.UInt32(0)

        analytic_hit = (
            active
            & mi.Bool(analytic)
            & frame.valid[0]
            & (frame.shapes[0] == interaction.shape)
        )

        incoming = _normalize(mi.Vector3f(ray.d))
        first, second = _plane_basis(-incoming)
        query_position = mi.Point3f(interaction.p)
        projected: list[tuple[Any, Any, Any, Any]] = []
        count = mi.Float(0.0)
        distance_sum = mi.Float(0.0)
        nearest_distance = mi.Float(float("inf"))
        nearest_u = mi.Float(fallback[0])
        nearest_v = mi.Float(fallback[1])

        for offset, direction, position, shape, valid in zip(
            self.offsets,
            frame.directions,
            frame.positions,
            frame.shapes,
            frame.valid,
        ):
            coordinate = (
                frame.base_coordinate[0] + float(offset[0]),
                frame.base_coordinate[1] + float(offset[1]),
            )
            candidate = (
                active
                & valid
                & (shape == interaction.shape)
                & (dr.dot(incoming, _normalize(direction)) >= self._angular_cosine)
            )
            safe_position = dr.select(candidate, position, query_position)
            displacement = safe_position - query_position
            x = dr.dot(displacement, first)
            y = dr.dot(displacement, second)
            distance = dr.sqrt(dr.square(x) + dr.square(y))
            projected.append((x, y, distance, candidate))
            count += dr.select(candidate, 1.0, 0.0)
            distance_sum += dr.select(candidate, distance, 0.0)
            closer = candidate & (distance < nearest_distance)
            nearest_distance = dr.select(closer, distance, nearest_distance)
            nearest_u = dr.select(closer, coordinate[0], nearest_u)
            nearest_v = dr.select(closer, coordinate[1], nearest_v)

        sigma = self.config.sigma_scale * dr.maximum(
            distance_sum / dr.maximum(count, 1.0), 1e-8
        )
        weight_sum = mi.Float(0.0)
        mean_x = mi.Float(0.0)
        mean_y = mi.Float(0.0)
        mean_u = mi.Float(0.0)
        mean_v = mi.Float(0.0)
        weights: list[Any] = []
        for offset, (x, y, distance, candidate) in zip(
            self.offsets, projected
        ):
            coordinate = (
                frame.base_coordinate[0] + float(offset[0]),
                frame.base_coordinate[1] + float(offset[1]),
            )
            weight = dr.select(
                candidate,
                dr.exp(-dr.square(distance / dr.maximum(sigma, 1e-8))),
                0.0,
            )
            weights.append(weight)
            weight_sum += weight
            mean_x += weight * x
            mean_y += weight * y
            mean_u += weight * coordinate[0]
            mean_v += weight * coordinate[1]
        inverse_weight = 1.0 / dr.maximum(weight_sum, 1e-20)
        mean_x *= inverse_weight
        mean_y *= inverse_weight
        mean_u *= inverse_weight
        mean_v *= inverse_weight

        cxx = mi.Float(0.0)
        cxy = mi.Float(0.0)
        cyy = mi.Float(0.0)
        cxu = mi.Float(0.0)
        cyu = mi.Float(0.0)
        cxv = mi.Float(0.0)
        cyv = mi.Float(0.0)
        for offset, (x, y, _, _), weight in zip(
            self.offsets, projected, weights
        ):
            coordinate = (
                frame.base_coordinate[0] + float(offset[0]),
                frame.base_coordinate[1] + float(offset[1]),
            )
            dx = x - mean_x
            dy = y - mean_y
            du = coordinate[0] - mean_u
            dv = coordinate[1] - mean_v
            cxx += weight * dx * dx
            cxy += weight * dx * dy
            cyy += weight * dy * dy
            cxu += weight * dx * du
            cyu += weight * dy * du
            cxv += weight * dx * dv
            cyv += weight * dy * dv

        determinant = cxx * cyy - cxy * cxy
        covariance_scale = dr.square(cxx + cyy)
        relative_determinant = determinant / dr.maximum(covariance_scale, 1e-20)
        fit = (
            active
            & (count >= 3.0)
            & (weight_sum > 1e-12)
            & (relative_determinant > self.config.condition_epsilon)
            & dr.isfinite(relative_determinant)
        )
        inverse_determinant = 1.0 / dr.maximum(determinant, 1e-20)
        gradient_ux = (cyy * cxu - cxy * cyu) * inverse_determinant
        gradient_uy = (cxx * cyu - cxy * cxu) * inverse_determinant
        gradient_vx = (cyy * cxv - cxy * cyv) * inverse_determinant
        gradient_vy = (cxx * cyv - cxy * cxv) * inverse_determinant
        fitted_u = mean_u - gradient_ux * mean_x - gradient_uy * mean_y
        fitted_v = mean_v - gradient_vx * mean_x - gradient_vy * mean_y
        has_neighbor = active & (count > 0.0)
        result = mi.Point2f(
            dr.select(fit, fitted_u, dr.select(has_neighbor, nearest_u, fallback[0])),
            dr.select(fit, fitted_v, dr.select(has_neighbor, nearest_v, fallback[1])),
        )
        full_mls = fit & (count >= float(self.config.min_candidates))
        method = dr.select(
            full_mls,
            mi.UInt32(1),
            dr.select(fit, mi.UInt32(2), dr.select(has_neighbor, 3, 4)),
        )
        confidence = dr.select(
            full_mls,
            dr.minimum(
                1.0,
                (count / float(self.config.min_candidates))
                * dr.sqrt(
                    dr.maximum(relative_determinant, 0.0)
                    / self.config.condition_epsilon
                ),
            ),
            dr.select(fit, 0.25, dr.select(has_neighbor, 0.1, 0.0)),
        )
        result = dr.select(analytic_hit, fallback, result)
        confidence = dr.select(analytic_hit, 1.0, confidence)
        method = dr.select(analytic_hit, mi.UInt32(0), method)
        return result, active, confidence, method

    def extend(
        self,
        scene: Any,
        frame: CudaToneFrame,
        base_interaction: Any,
        active: Any,
    ) -> CudaToneFrame:
        """Advance every anchor by one material-independent mirror bounce."""

        active = mi.Bool(active)
        if frame.next_frame is not None:
            cached = frame.next_frame
            # Enforce the same compatible-prefix test as the streaming
            # implementation, but reuse the pretraced next edge. Invalid
            # fields are never consumed; only their masks are path-specific.
            compatible = [
                active
                & valid
                & (shape == base_interaction.shape)
                & cached_valid
                for shape, valid, cached_valid in zip(
                    frame.shapes, frame.valid, cached.valid
                )
            ]
            result = CudaToneFrame(
                base_coordinate=cached.base_coordinate,
                directions=cached.directions,
                positions=cached.positions,
                geometric_normals=cached.geometric_normals,
                shapes=cached.shapes,
                valid=compatible,
                time=cached.time,
                wavelengths=cached.wavelengths,
                next_frame=cached.next_frame,
            )
            dr.eval(*result.valid)
            return result

        new_directions: list[Any] = []
        new_positions: list[Any] = []
        new_normals: list[Any] = []
        new_shapes: list[Any] = []
        new_valid: list[Any] = []
        pending: list[list[Any]] = []
        ray_flags = int(mi.RayFlags.Minimal)

        for direction, position, normal, shape, valid in zip(
            frame.directions,
            frame.positions,
            frame.geometric_normals,
            frame.shapes,
            frame.valid,
        ):
            prefix = active & valid & (shape == base_interaction.shape)
            normal = _normalize(normal)
            reflected = _normalize(direction - 2.0 * dr.dot(direction, normal) * normal)
            finite = (
                dr.isfinite(reflected[0])
                & dr.isfinite(reflected[1])
                & dr.isfinite(reflected[2])
            )
            prefix &= finite
            source = mi.Interaction3f()
            source.p = position
            source.n = normal
            source.time = frame.time
            source.wavelengths = frame.wavelengths
            child_ray = source.spawn_ray(reflected)
            child = self._ray_intersect(
                scene, child_ray, ray_flags, prefix
            )
            new_directions.append(mi.Vector3f(child_ray.d))
            new_positions.append(mi.Point3f(child.p))
            # Keep a structurally uniform frame at every depth. In particular,
            # ShapePtr/Normal arrays must have identical anchor cardinality
            # when a compact frame crosses an asynchronous CUDA evaluation
            # boundary. Empty terminal lists saved little compared with tile
            # streaming and could leave a later pointer kernel with a different
            # aggregate layout.
            new_normals.append(mi.Normal3f(child.n))
            new_shapes.append(child.shape)
            new_valid.append(prefix & child.is_valid())
            retained = [
                new_directions[-1], new_positions[-1], new_normals[-1],
                new_shapes[-1], new_valid[-1],
            ]
            pending.append(retained)
            self._flush(pending)
        self._flush(pending, force=True)
        return CudaToneFrame(
            base_coordinate=frame.base_coordinate,
            directions=new_directions,
            positions=new_positions,
            geometric_normals=new_normals,
            shapes=new_shapes,
            valid=new_valid,
            time=frame.time,
            wavelengths=frame.wavelengths,
        ).eval()


__all__ = ["CudaToneFrame", "CudaToneMapper"]

"""CUDA wavefront implementation of the recursive SRE integrator."""

from __future__ import annotations

from pathlib import Path

import drjit as dr
import mitsuba as mi

try:
    from .config import load_config
    from .cuda_backend import (
        DeviceStats, FrozenRayIntersector, WavefrontEstimator, finite,
        gather_lanes, max_abs,
    )
    from .cuda_feature_lines import CudaFeatureLineTracer
    from .cuda_tone_mapping import CudaToneMapper
    from .estimators import IdentityEstimator
    from .styles import StyleContext, ToneHatch, ToneHalftone
except ImportError:
    from config import load_config
    from cuda_backend import (
        DeviceStats, FrozenRayIntersector, WavefrontEstimator, finite,
        gather_lanes, max_abs,
    )
    from cuda_feature_lines import CudaFeatureLineTracer
    from cuda_tone_mapping import CudaToneMapper
    from estimators import IdentityEstimator
    from styles import StyleContext, ToneHatch, ToneHalftone


_REGISTERED = False


def register_cuda_integrator():
    global _REGISTERED
    if _REGISTERED:
        return

    class CudaSREIntegrator(mi.SamplingIntegrator):
        def __init__(self, props):
            super().__init__(props)
            self.max_depth = int(props.get("max_depth", 6))
            self.rr_depth = int(props.get("rr_depth", 5))
            self.rr_probability = float(props.get("rr_probability", 0.95))
            config_path = str(props.get("style_config", ""))
            self.config = (
                load_config(Path(config_path)) if config_path else load_config(None)
            )
            self.feature_lines = (
                CudaFeatureLineTracer(self.config.feature_lines)
                if self.config.feature_lines.enabled else None
            )
            self.tone_mapper = (
                CudaToneMapper(self.config.tone_mapping)
                if self.config.tone_mapping.enabled else None
            )
            # Do not retain ``scene`` here. A Scene owns its integrator, so a
            # strong Integrator -> Scene reference creates a pybind/C++ cycle.
            # Fresh per-tile processes now provide a hard cleanup boundary,
            # while avoiding this cycle also reduces the live allocation set
            # within an individual tile.
            self._materials_prepared = False
            self._tracked_materials = {}
            self._tracked_bsdfs = {}
            self._tone_materials = {}
            self._tone_bsdfs = {}
            self._only_environment_emitter = False
            self._intersector = None
            if self.max_depth < 1:
                raise ValueError("max_depth must be at least one")
            if not 0.0 < self.rr_probability <= 1.0:
                raise ValueError("rr_probability must be in (0, 1]")

        def aov_names(self):
            return [
                "sre_tree_nodes",
                "sre_style_evaluations",
                "sre_inner_variance",
                "sre_estimated_bias",
            ]

        @staticmethod
        def _shape_pointer(shape):
            return mi.ShapePtr(shape)

        def _ray_intersect(self, scene, ray, active):
            if self._intersector is not None:
                return self._intersector.ray_intersect(
                    ray, active=active
                )
            return scene.ray_intersect(ray, active)

        def _prepare_materials(self, scene):
            # Each integrator instance belongs to exactly one Mitsuba scene.
            # A boolean is therefore sufficient and, unlike storing ``scene``,
            # cannot keep the complete previous tile scene alive.
            if self._materials_prepared:
                return

            materials = {}
            material_bsdfs = {}
            for shape in scene.shapes():
                shape_id = shape.id()
                bsdf = shape.bsdf()
                material_id = (bsdf.id() if bsdf is not None else "") or shape_id
                binding = self.config.shapes.get(
                    shape_id,
                    self.config.materials.get(material_id, self.config.default),
                )
                materials.setdefault(material_id, []).append(
                    (self._shape_pointer(shape), shape_id, binding)
                )
                # A non-empty Mitsuba object ID uniquely names the referenced
                # BSDF. Anonymous BSDFs use shape_id above and are therefore
                # unique as well.
                material_bsdfs.setdefault(material_id, mi.BSDFPtr(bsdf))

            self._tracked_materials = {
                material_id: entries
                for material_id, entries in materials.items()
                if any(
                    not isinstance(binding.estimator, IdentityEstimator)
                    for _, _, binding in entries
                )
            }
            self._tracked_bsdfs = {
                material_id: material_bsdfs[material_id]
                for material_id in self._tracked_materials
            }
            self._tone_materials = {
                material_id: entries
                for material_id, entries in materials.items()
                if any(
                    self._estimator_uses_tone(binding.estimator)
                    for _, _, binding in entries
                )
            }
            self._tone_bsdfs = {
                material_id: material_bsdfs[material_id]
                for material_id in self._tone_materials
            }
            # For a lone environment emitter, glossy BSDF sampling is the
            # low-variance strategy (especially for the near-specular Fig. 10
            # floor). Uniform environment NEE otherwise creates bright GGX
            # fireflies while contributing no additional light source.
            self._only_environment_emitter = (
                len(scene.emitters()) == 1 and scene.environment() is not None
            )
            # Frozen traversal specifically addresses the hundreds of repeated
            # queries in a nested tone expectation. Leave feature-line-only
            # and identity renders on their already validated path.
            self._intersector = (
                FrozenRayIntersector(scene)
                if self.tone_mapper is not None else None
            )
            if self.feature_lines is not None:
                if self._intersector is not None:
                    self.feature_lines.set_intersector(self._intersector)
                self.feature_lines.prepare(scene)
            if self.tone_mapper is not None:
                self.tone_mapper.set_intersector(self._intersector)
                self.tone_mapper.prepare(scene)
            self._materials_prepared = True

        @staticmethod
        def _emission(scene, si, active):
            return si.emitter(scene).eval(si, active)

        @staticmethod
        def _luminance(value):
            return 0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2]

        def _brightness_adjust(self, control, value):
            brightness = (
                (value[0] + value[1] + value[2]) / 3.0
                if control.brightness_mode == "mean"
                else self._luminance(value)
            )
            values = control.levels if control.uses_target_levels else control.gains
            selected = mi.Float(float(values[0]))
            for index, threshold in enumerate(control.thresholds):
                selected = dr.select(
                    brightness >= float(threshold),
                    float(values[index + 1]),
                    selected,
                )
            if control.uses_target_levels:
                scale = dr.select(
                    brightness > 1e-8,
                    selected / dr.maximum(brightness, 1e-8),
                    0.0,
                )
                return value * scale
            return value * selected

        def _lighting_result(self, emission, reflected, depth, distance):
            """Apply Section 6.3 controls before the tone style sees radiance."""
            style = self.config.lighting_style
            # These are image-layer controls, so applying them below the
            # camera vertex would compound the gain along a path.
            if not style.enabled or depth != 0:
                return emission + reflected
            result = (
                self._brightness_adjust(style.emission, emission)
                + (
                    reflected
                    if style.reflected.uses_target_levels
                    else self._brightness_adjust(style.reflected, reflected)
                )
            )
            if depth == 0 and style.far_distance > style.near_distance:
                weight = dr.clamp(
                    (distance - style.near_distance)
                    / (style.far_distance - style.near_distance),
                    0.0,
                    1.0,
                )
                weight = weight * weight * (3.0 - 2.0 * weight)
                result *= dr.lerp(style.near_gain, style.far_gain, weight)
            return result

        @classmethod
        def _estimator_uses_tone(cls, estimator):
            """Whether an estimator consumes the lifted image coordinate.

            Composite estimators may hide the spatial function several levels
            down.  Walking their conventional child fields keeps this test
            independent of a particular estimator composition and lets
            identity mirror vertices bypass an otherwise expensive MLS query.
            """
            if isinstance(
                getattr(estimator, "function", None),
                (ToneHatch, ToneHalftone),
            ):
                return True
            return any(
                cls._estimator_uses_tone(child)
                for name in ("left", "right", "outer", "inner")
                if (child := getattr(estimator, name, None)) is not None
            )

        def _tone_query_active(self, si, bsdf, visits, depth, active):
            """Mask vertices whose selected style actually reads tone data."""
            result = mi.Bool(False)
            for material_id, entries in self._tone_materials.items():
                previous = visits.get(material_id, 0)
                material_active = mi.Bool(active) & (
                    bsdf == self._tone_bsdfs[material_id]
                )
                groups = list(self._binding_groups(entries))
                for binding, shapes in groups:
                    if (
                        not self._estimator_uses_tone(binding.estimator)
                        or not self._can_match_from(
                            binding.predicate, depth, previous
                        )
                    ):
                        continue
                    if len(groups) == 1:
                        result |= material_active
                    else:
                        for shape, _ in shapes:
                            result |= material_active & (si.shape == shape)
            return result

        @staticmethod
        def _any(active):
            return bool(dr.any(active))

        @staticmethod
        def _can_match_from(predicate, depth, previous_occurrences):
            if predicate.max_depth is not None and depth > predicate.max_depth:
                return False
            if predicate.depths is not None and max(predicate.depths) < depth:
                return False
            if predicate.first_hit and previous_occurrences >= 1:
                return False
            if (
                predicate.max_occurrences is not None
                and previous_occurrences >= predicate.max_occurrences
            ):
                return False
            return True

        @staticmethod
        def _binding_groups(entries):
            groups = {}
            for shape, shape_id, binding in entries:
                key = id(binding)
                if key not in groups:
                    groups[key] = [binding, []]
                groups[key][1].append((shape, shape_id))
            return groups.values()

        def _has_future_styles(self, visits, depth):
            for material_id, entries in self._tracked_materials.items():
                previous = visits.get(material_id, 0)
                for _, _, binding in entries:
                    if (
                        not isinstance(binding.estimator, IdentityEstimator)
                        and self._can_match_from(binding.predicate, depth, previous)
                    ):
                        return True
            return False

        @dr.syntax
        def _plain_radiance(self, scene, sampler, ray, depth, stats, active):
            """Trace a linear suffix using one symbolic CUDA loop."""
            result = mi.Color3f(0.0)
            throughput = mi.Color3f(1.0)
            # Camera samples arrive as RayDifferential3f, while spawn_ray()
            # returns Ray3f. Dr.Jit loop state types must remain unchanged.
            current_ray = mi.Ray3f(ray)
            current_active = mi.Bool(active)
            current_depth = mi.UInt32(depth)
            ctx = mi.BSDFContext()

            while dr.hint(
                current_active,
                max_iterations=self.max_depth - depth,
                mode="evaluated",
                label="SRE linear suffix",
            ):
                stats.tree_nodes += dr.select(current_active, 1.0, 0.0)
                si = scene.ray_intersect(current_ray, current_active)
                result += throughput * self._emission(scene, si, current_active)
                continuation = (
                    current_active
                    & si.is_valid()
                    & (current_depth + 1 < self.max_depth)
                )
                use_rr = current_depth >= self.rr_depth
                continuation &= dr.select(
                    use_rr,
                    sampler.next_1d() < self.rr_probability,
                    True,
                )

                bsdf = si.bsdf(current_ray)
                has_smooth = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                can_sample_emitter = has_smooth & (len(scene.emitters()) > 0)
                if self._only_environment_emitter:
                    can_sample_emitter &= ~mi.has_flag(
                        bsdf.flags(), mi.BSDFFlags.Glossy
                    )
                emitter_probability = dr.select(can_sample_emitter, 0.5, 0.0)
                choose_emitter = (
                    continuation
                    & can_sample_emitter
                    & (sampler.next_1d() < emitter_probability)
                )
                choose_bsdf = continuation & (choose_emitter == False)

                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(), True, choose_emitter
                )
                f_cos, bsdf_pdf_emitter = bsdf.eval_pdf(
                    ctx, si, si.to_local(ds.d), choose_emitter
                )
                valid_emitter = (
                    choose_emitter
                    & (ds.pdf > 0.0)
                    & (max_abs(emitter_weight) > 0.0)
                    & finite(emitter_weight)
                )
                # ``emitter_weight`` is L_e / p_e.  Convert it back to the
                # sampled radiance before dividing by the full mixture PDF.
                # This is essential for delta emitters (e.g. Blender point
                # lights): they have no geometry for a continuation ray to
                # intersect, so their contribution must be accumulated here.
                emitter_mixture_pdf = (
                    emitter_probability * ds.pdf
                    + dr.select(
                        ds.delta,
                        0.0,
                        (1.0 - emitter_probability) * bsdf_pdf_emitter,
                    )
                )
                direct = dr.select(
                    valid_emitter & (emitter_mixture_pdf > 0.0),
                    f_cos * emitter_weight * ds.pdf
                    / dr.maximum(emitter_mixture_pdf, 1e-20),
                    0.0,
                )
                direct *= dr.select(use_rr, 1.0 / self.rr_probability, 1.0)
                result += throughput * direct

                bs, bsdf_weight = bsdf.sample(
                    ctx, si, sampler.next_1d(), sampler.next_2d(), choose_bsdf
                )
                valid_bsdf = (
                    choose_bsdf
                    & (bs.pdf > 0.0)
                    & (max_abs(bsdf_weight) > 0.0)
                    & finite(bsdf_weight)
                )
                bsdf_direction = si.to_world(bs.wo)
                next_si = scene.ray_intersect(
                    si.spawn_ray(bsdf_direction), valid_bsdf & can_sample_emitter
                )
                ds_hit = mi.DirectionSample3f(scene, next_si, si)
                emitter_pdf = scene.pdf_emitter_direction(
                    si, ds_hit, valid_bsdf & can_sample_emitter
                )
                bsdf_mixture_pdf = (
                    (1.0 - emitter_probability) * bs.pdf
                    + emitter_probability * emitter_pdf
                )

                numerator = bsdf_weight * bs.pdf
                mixture_pdf = bsdf_mixture_pdf
                current_active = (
                    valid_bsdf
                    & (mixture_pdf > 0.0)
                    & finite(numerator)
                )
                throughput *= dr.select(
                    current_active,
                    numerator / dr.maximum(mixture_pdf, 1e-20),
                    0.0,
                )
                throughput *= dr.select(use_rr, 1.0 / self.rr_probability, 1.0)
                current_ray = si.spawn_ray(bsdf_direction)
                current_depth += 1
            return result

        def _radiance(
            self, scene, sampler, ray, depth, visits, stats, active,
            auxiliary_frame=None, tone_frame=None, tone_analytic=False,
            interaction=None,
        ):
            use_feature_lines = (
                self.feature_lines is not None
                and self.config.feature_lines.can_apply_from(depth)
            )
            if (
                not use_feature_lines
                and (not self.config.lighting_style.enabled or depth > 0)
                and not self._has_future_styles(visits, depth)
            ):
                return self._plain_radiance(
                    scene, sampler, ray, depth, stats, active
                )
            active = mi.Bool(active)
            stats.tree_nodes += dr.select(active, 1.0, 0.0)
            # The primary interaction is already needed for SamplingIntegrator's
            # validity mask. Reuse it instead of tracing the camera ray twice.
            si = (
                interaction
                if interaction is not None
                else self._ray_intersect(scene, ray, active)
            )
            surface_active = active & si.is_valid()
            miss_active = active & (si.is_valid() == False)
            result = dr.select(
                miss_active,
                self._lighting_result(
                    self._emission(scene, si, miss_active),
                    mi.Color3f(0.0),
                    depth,
                    mi.Float(float("inf")),
                ),
                mi.Color3f(0.0),
            )
            if use_feature_lines:
                if (
                    depth > 0
                    and self.config.feature_lines.resample_delta_reflections
                    and auxiliary_frame is not None
                    and (
                        self.config.feature_lines.resample_glossy_reflections
                        or not bool(dr.any(
                            auxiliary_frame.distributional & surface_active
                        ))
                    )
                ):
                    # The parent frame establishes the true mirror mapping.
                    # Refit that mapping at the reflected hit, then allocate
                    # all auxiliary samples to this material's own stencil.
                    parent_distributional = auxiliary_frame.distributional
                    local_ray = self.feature_lines.local_ray_differential(
                        ray, auxiliary_frame, surface_active
                    )
                    auxiliary_frame = self.feature_lines.spawn_primary(
                        scene,
                        sampler,
                        local_ray,
                        surface_active,
                        si.bsdf(ray),
                        si.shape,
                        si.p,
                        detection_depth=depth,
                    )
                    # Local resampling changes the stencil support, not the
                    # scattering event. Retain the glossy marker so line
                    # contrast is integrated across rough-BSDF samples.
                    auxiliary_frame.distributional = parent_distributional
                    auxiliary_frame.eval()
                line_color, line_hit, line_continuation = (
                    self.feature_lines.detect(
                        si,
                        ray,
                        auxiliary_frame,
                        sampler,
                        depth,
                        surface_active,
                    )
                )
                result += dr.select(line_hit, line_color, 0.0)
                surface_active &= line_hit == False
            else:
                line_continuation = None
            emission = self._emission(scene, si, surface_active)
            bsdf = si.bsdf(ray)
            wavefront = WavefrontEstimator(sampler)
            tone_query_active = (
                self._tone_query_active(
                    si, bsdf, visits, depth, surface_active
                )
                if self.tone_mapper is not None else mi.Bool(False)
            )
            if (
                self.tone_mapper is not None
                and self.config.tone_mapping.active_at(depth)
                and self._any(tone_query_active)
            ):
                (
                    tone_coordinate,
                    tone_valid,
                    tone_confidence,
                    tone_method,
                ) = self.tone_mapper.query(
                    tone_frame,
                    si,
                    ray,
                    depth,
                    tone_query_active,
                    tone_analytic,
                )
            else:
                tone_coordinate = mi.Point2f(0.0)
                tone_valid = mi.Bool(False)
                tone_confidence = mi.Float(0.0)
                tone_method = mi.UInt32(5)
            tone_child_frame = None

            def prepare_child(draw_active):
                draw_active = mi.Bool(draw_active) & surface_active
                if depth + 1 >= self.max_depth:
                    return None

                continuation = mi.Bool(draw_active)
                if depth >= self.rr_depth:
                    continuation &= sampler.next_1d() < self.rr_probability

                ctx = mi.BSDFContext()
                has_smooth = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                can_sample_emitter = has_smooth & (len(scene.emitters()) > 0)
                if self._only_environment_emitter:
                    can_sample_emitter &= ~mi.has_flag(
                        bsdf.flags(), mi.BSDFFlags.Glossy
                    )
                emitter_probability = dr.select(can_sample_emitter, 0.5, 0.0)
                choose_emitter = (
                    continuation
                    & can_sample_emitter
                    & (sampler.next_1d() < emitter_probability)
                )
                choose_bsdf = continuation & (choose_emitter == False)

                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(), True, choose_emitter
                )
                f_cos, bsdf_pdf_emitter = bsdf.eval_pdf(
                    ctx, si, si.to_local(ds.d), choose_emitter
                )
                valid_emitter = (
                    choose_emitter
                    & (ds.pdf > 0.0)
                    & (max_abs(emitter_weight) > 0.0)
                    & finite(emitter_weight)
                )
                emitter_mixture_pdf = (
                    emitter_probability * ds.pdf
                    + dr.select(
                        ds.delta,
                        0.0,
                        (1.0 - emitter_probability) * bsdf_pdf_emitter,
                    )
                )
                direct = dr.select(
                    valid_emitter & (emitter_mixture_pdf > 0.0),
                    f_cos * emitter_weight * ds.pdf
                    / dr.maximum(emitter_mixture_pdf, 1e-20),
                    0.0,
                )
                if depth >= self.rr_depth:
                    direct /= self.rr_probability

                bs, bsdf_weight = bsdf.sample(
                    ctx, si, sampler.next_1d(), sampler.next_2d(), choose_bsdf
                )
                valid_bsdf = (
                    choose_bsdf
                    & (bs.pdf > 0.0)
                    & (max_abs(bsdf_weight) > 0.0)
                    & finite(bsdf_weight)
                )
                bsdf_direction = si.to_world(bs.wo)
                bsdf_ray = si.spawn_ray(bsdf_direction)
                next_si = self._ray_intersect(
                    scene,
                    bsdf_ray,
                    valid_bsdf & can_sample_emitter,
                )
                ds_hit = mi.DirectionSample3f(scene, next_si, si)
                emitter_pdf = scene.pdf_emitter_direction(
                    si, ds_hit, valid_bsdf & can_sample_emitter
                )
                bsdf_mixture_pdf = (
                    (1.0 - emitter_probability) * bs.pdf
                    + emitter_probability * emitter_pdf
                )

                numerator = bsdf_weight * bs.pdf
                mixture_pdf = bsdf_mixture_pdf
                child_active = (
                    valid_bsdf
                    & (mixture_pdf > 0.0)
                    & finite(numerator)
                )
                return (
                    bsdf_ray,
                    numerator,
                    mixture_pdf,
                    child_active,
                    direct,
                    draw_active,
                    bsdf_direction,
                    bs.sampled_type,
                    bs.eta,
                )

            def finish_child(child, incoming):
                (
                    _, numerator, mixture_pdf, child_active, direct,
                    draw_active, _, _, _,
                ) = child
                transport = dr.select(
                    child_active,
                    numerator * incoming / dr.maximum(mixture_pdf, 1e-20),
                    0.0,
                )
                if depth >= self.rr_depth:
                    transport /= self.rr_probability
                return dr.select(
                    draw_active,
                    self._lighting_result(
                        emission, direct + transport, depth, si.t
                    ),
                    0.0,
                )

            def sample_integrand(draw_active, next_visits):
                nonlocal tone_child_frame
                draw_active = mi.Bool(draw_active) & surface_active
                if depth + 1 >= self.max_depth:
                    return dr.select(
                        draw_active,
                        self._lighting_result(
                            emission, mi.Color3f(0.0), depth, si.t
                        ),
                        0.0,
                    )
                child = prepare_child(draw_active)
                (
                    child_ray, numerator, mixture_pdf, child_active, _, _,
                    bsdf_direction, sampled_type, sampled_eta,
                ) = child
                dr.eval(
                    child_ray,
                    numerator,
                    mixture_pdf,
                    child_active,
                    bsdf_direction,
                    sampled_type,
                    sampled_eta,
                )

                active_indices = dr.compress(child_active)
                active_count = dr.width(active_indices)
                full_count = dr.width(child_active)
                if active_count == 0:
                    return finish_child(child, mi.Color3f(0.0))

                if (
                    self.tone_mapper is not None
                    and self.config.tone_mapping.can_extend_from(depth)
                    and tone_child_frame is None
                ):
                    # Canonical anchors depend on macro geometry and the path
                    # prefix, never on the sampled BSDF direction. Reuse one
                    # compact frame across every inner estimator draw.
                    tone_child_frame = self.tone_mapper.extend(
                        scene,
                        tone_frame,
                        si,
                        surface_active,
                    )

                if active_count == full_count:
                    child_auxiliary = None
                    if (
                        self.feature_lines is not None
                        and self.config.feature_lines.can_apply_from(depth + 1)
                    ):
                        child_auxiliary = self.feature_lines.extend(
                            scene,
                            auxiliary_frame,
                            line_continuation,
                            si.sh_frame.n,
                            ray.d,
                            bsdf_direction,
                            sampled_type,
                            sampled_eta,
                            child_active,
                            depth + 1,
                        )
                    incoming = self._radiance(
                        scene,
                        sampler,
                        child_ray,
                        depth + 1,
                        next_visits,
                        stats,
                        child_active,
                        child_auxiliary,
                        tone_child_frame,
                        mi.Bool(tone_analytic)
                        & mi.has_flag(
                            sampled_type, mi.BSDFFlags.DeltaReflection
                        ),
                    )
                    return finish_child(child, incoming)

                # A child is evaluated to completion before the next draw.
                # Compact only this child's live transport lanes, recurse,
                # then scatter its contribution back to the camera wavefront.
                # This preserves DFS while releasing dead-lane state at every
                # deeper level.
                seeds = mi.UInt32(sampler.next_1d() * 4294967295.0)
                # Ray3f is ragged in cuda_ad_rgb: origins/directions have the
                # parent width, ``maxt`` and ``time`` are commonly width-one
                # constants, and the RGB wavelength packet has width zero.
                # Aggregate ``dr.gather(mi.Ray3f, ...)`` recursively indexes
                # every field and therefore reads past maxt/time. Compact the
                # lane-varying/scalar fields independently and share the
                # compile-time empty wavelength packet.
                compact_ray = mi.Ray3f()
                compact_ray.o = gather_lanes(
                    mi.Point3f, child_ray.o, active_indices, full_count,
                    "child ray origin",
                )
                compact_ray.d = gather_lanes(
                    mi.Vector3f, child_ray.d, active_indices, full_count,
                    "child ray direction",
                )
                compact_ray.maxt = gather_lanes(
                    mi.Float, child_ray.maxt, active_indices, full_count,
                    "child ray maximum distance",
                )
                compact_ray.time = gather_lanes(
                    mi.Float, child_ray.time, active_indices, full_count,
                    "child ray time",
                )
                compact_ray.wavelengths = child_ray.wavelengths
                compact_numerator = gather_lanes(
                    mi.Color3f, numerator, active_indices, full_count,
                    "child numerator",
                )
                compact_pdf = gather_lanes(
                    mi.Float, mixture_pdf, active_indices, full_count,
                    "child mixture PDF",
                )
                compact_seeds = gather_lanes(
                    mi.UInt32, seeds, active_indices, full_count,
                    "child sampler seeds",
                )
                compact_sampler = sampler.fork()
                compact_sampler.seed(compact_seeds, active_count)
                compact_sampler.schedule_state()
                compact_auxiliary = None
                if (
                    self.feature_lines is not None
                    and self.config.feature_lines.can_apply_from(depth + 1)
                ):
                    compact_parent_frame = auxiliary_frame.gather(active_indices)
                    compact_continuation = [
                        gather_lanes(
                            mi.Bool, valid, active_indices, full_count,
                            "feature-line continuation",
                        )
                        for valid in line_continuation
                    ]
                    compact_auxiliary = self.feature_lines.extend(
                        scene,
                        compact_parent_frame,
                        compact_continuation,
                        gather_lanes(
                            mi.Normal3f, si.sh_frame.n, active_indices,
                            full_count, "surface shading normal",
                        ),
                        gather_lanes(
                            mi.Vector3f, ray.d, active_indices, full_count,
                            "incoming direction",
                        ),
                        gather_lanes(
                            mi.Vector3f, bsdf_direction, active_indices,
                            full_count, "sampled BSDF direction",
                        ),
                        gather_lanes(
                            mi.UInt32, sampled_type, active_indices,
                            full_count, "sampled BSDF type",
                        ),
                        gather_lanes(
                            mi.Float, sampled_eta, active_indices, full_count,
                            "sampled BSDF eta",
                        ),
                        dr.full(mi.Bool, True, active_count),
                        depth + 1,
                    )
                compact_tone = None
                if tone_child_frame is not None:
                    # Materialize pointer-bearing tone gathers before the
                    # compact child starts building another recursive CUDA
                    # graph. This explicit lifetime boundary prevents the
                    # source frame from being recycled while ShapePtr gathers
                    # are still queued on another stream/kernel.
                    compact_tone = tone_child_frame.gather(
                        active_indices
                    ).eval()
                compact_tone_analytic = gather_lanes(
                    mi.Bool,
                    mi.Bool(tone_analytic)
                    & mi.has_flag(
                        sampled_type, mi.BSDFFlags.DeltaReflection
                    ),
                    active_indices,
                    full_count,
                    "analytic tone mask",
                )
                dr.eval(
                    compact_ray,
                    compact_numerator,
                    compact_pdf,
                    compact_seeds,
                )

                child_stats = DeviceStats()
                compact_active = dr.full(mi.Bool, True, active_count)
                incoming = self._radiance(
                    scene,
                    compact_sampler,
                    compact_ray,
                    depth + 1,
                    next_visits,
                    child_stats,
                    compact_active,
                    compact_auxiliary,
                    compact_tone,
                    compact_tone_analytic,
                )
                compact_transport = (
                    compact_numerator * incoming
                    / dr.maximum(compact_pdf, 1e-20)
                )
                if depth >= self.rr_depth:
                    compact_transport /= self.rr_probability
                transport = dr.zeros(mi.Color3f, full_count)
                dr.scatter(transport, compact_transport, active_indices)

                # ``dr.scatter()`` is an asynchronous side effect. Its target
                # remains marked dirty until an evaluation boundary executes
                # the write. Reading such a target while Dr.Jit assembles a
                # later kernel eventually aborts with
                # ``jit_assemble(): dirty variable ... encountered``. This is
                # especially easy to trigger in bottom/background crops where
                # compaction is frequent.
                #
                # Collect every full-wavefront scatter target first and
                # materialize them together below. The boundary is deliberately
                # placed after the compact child DFS has completed and before
                # its contribution is consumed by the parent. Consequently it
                # neither changes recursion order nor revives discarded lanes,
                # and costs one synchronization per compact child rather than
                # one synchronization per AOV.
                expanded_stats = []
                for name in (
                    "tree_nodes",
                    "draws",
                    "style_evaluations",
                    "inner_variance",
                    "estimated_bias",
                ):
                    source = getattr(child_stats, name)
                    if dr.width(source) == 1:
                        source = dr.repeat(source, active_count)
                    expanded = dr.zeros(mi.Float, full_count)
                    dr.scatter(expanded, source, active_indices)
                    expanded_stats.append((name, expanded))

                dr.eval(
                    transport,
                    *(expanded for _, expanded in expanded_stats),
                )

                for name, expanded in expanded_stats:
                    setattr(stats, name, getattr(stats, name) + expanded)
                return dr.select(
                    draw_active, emission + child[4] + transport, 0.0
                )

            def make_draw(next_visits):
                def draw(draw_active):
                    return sample_integrand(draw_active, next_visits)

                return draw

            def sample_integrand_packed(
                draw_active, next_visits, sample_count, recursive=True
            ):
                """Trace a small group of direct-estimator draws as lanes.

                Recursive styles pack two siblings to preserve the DFS memory
                bound. A first-hit-only direct estimator can pack all siblings
                because every one immediately enters the same linear physical
                radiance suffix and cannot create an exponential style tree.
                """
                nonlocal tone_child_frame
                draw_active = mi.Bool(draw_active) & surface_active
                parent_width = dr.width(surface_active)
                packed_width = parent_width * sample_count
                parent_indices = (
                    dr.arange(mi.UInt32, packed_width) // sample_count
                )

                def pack(dtype, value, label):
                    return gather_lanes(
                        dtype,
                        value,
                        parent_indices,
                        parent_width,
                        label,
                    )

                packed_draw_active = pack(
                    mi.Bool, draw_active, "packed direct active mask"
                )
                if depth + 1 >= self.max_depth:
                    return dr.select(
                        packed_draw_active,
                        pack(mi.Color3f, emission, "packed emission"),
                        0.0,
                    )

                packed_ray = mi.Ray3f()
                packed_ray.o = pack(
                    mi.Point3f, ray.o, "packed parent ray origin"
                )
                packed_ray.d = pack(
                    mi.Vector3f, ray.d, "packed parent ray direction"
                )
                packed_ray.maxt = pack(
                    mi.Float, ray.maxt, "packed parent ray maximum distance"
                )
                packed_ray.time = pack(
                    mi.Float, ray.time, "packed parent ray time"
                )
                packed_ray.wavelengths = ray.wavelengths
                # Packed direct estimators may coexist with different sample
                # counts (e.g. 8 default tone samples and 16 van samples in
                # Fig. 1). A frozen OptiX recording is keyed by RayFlags, but
                # SurfaceInteraction's virtual-call fields are also tied to
                # the packed allocation width. Replaying an 8-sibling query
                # after a 16-sibling query can therefore pair an N-wide SI
                # with a 2N-wide emitter pointer array. Keep these two
                # variable-width intersections explicit; fixed-width camera,
                # feature-line and MLS anchor queries remain frozen.
                packed_si = scene.ray_intersect(
                    packed_ray, packed_draw_active
                )
                packed_surface = packed_draw_active & packed_si.is_valid()
                packed_emission = self._emission(
                    scene, packed_si, packed_surface
                )
                packed_bsdf = packed_si.bsdf(packed_ray)

                # Seed the packed sibling lanes independently. The hash is a
                # bijective integer mixer; it does not reduce the estimator's
                # sample count or introduce shared random streams.
                base_seed = mi.UInt32(
                    sampler.next_1d() * 4294967295.0
                )
                base_seed = pack(
                    mi.UInt32, base_seed, "packed direct base seed"
                )
                sibling = (
                    dr.arange(mi.UInt32, packed_width)
                    - parent_indices * sample_count
                )
                seeds = base_seed ^ (
                    (sibling + 1) * mi.UInt32(0x9E3779B9)
                )
                seeds ^= seeds >> 16
                seeds *= mi.UInt32(0x7FEB352D)
                seeds ^= seeds >> 15
                seeds *= mi.UInt32(0x846CA68B)
                seeds ^= seeds >> 16
                packed_sampler = sampler.fork()
                packed_sampler.seed(seeds, packed_width)
                packed_sampler.schedule_state()

                continuation = mi.Bool(packed_surface)
                if depth >= self.rr_depth:
                    continuation &= (
                        packed_sampler.next_1d() < self.rr_probability
                    )

                ctx = mi.BSDFContext()
                has_smooth = mi.has_flag(
                    packed_bsdf.flags(), mi.BSDFFlags.Smooth
                )
                can_sample_emitter = has_smooth & (
                    len(scene.emitters()) > 0
                )
                if self._only_environment_emitter:
                    can_sample_emitter &= ~mi.has_flag(
                        packed_bsdf.flags(), mi.BSDFFlags.Glossy
                    )
                emitter_probability = dr.select(
                    can_sample_emitter, 0.5, 0.0
                )
                choose_emitter = (
                    continuation
                    & can_sample_emitter
                    & (packed_sampler.next_1d() < emitter_probability)
                )
                choose_bsdf = continuation & (choose_emitter == False)

                ds, emitter_weight = scene.sample_emitter_direction(
                    packed_si,
                    packed_sampler.next_2d(),
                    True,
                    choose_emitter,
                )
                f_cos, bsdf_pdf_emitter = packed_bsdf.eval_pdf(
                    ctx,
                    packed_si,
                    packed_si.to_local(ds.d),
                    choose_emitter,
                )
                valid_emitter = (
                    choose_emitter
                    & (ds.pdf > 0.0)
                    & (max_abs(emitter_weight) > 0.0)
                    & finite(emitter_weight)
                )
                emitter_mixture_pdf = (
                    emitter_probability * ds.pdf
                    + dr.select(
                        ds.delta,
                        0.0,
                        (1.0 - emitter_probability) * bsdf_pdf_emitter,
                    )
                )
                direct = dr.select(
                    valid_emitter & (emitter_mixture_pdf > 0.0),
                    f_cos * emitter_weight * ds.pdf
                    / dr.maximum(emitter_mixture_pdf, 1e-20),
                    0.0,
                )
                if depth >= self.rr_depth:
                    direct /= self.rr_probability

                bs, bsdf_weight = packed_bsdf.sample(
                    ctx,
                    packed_si,
                    packed_sampler.next_1d(),
                    packed_sampler.next_2d(),
                    choose_bsdf,
                )
                valid_bsdf = (
                    choose_bsdf
                    & (bs.pdf > 0.0)
                    & (max_abs(bsdf_weight) > 0.0)
                    & finite(bsdf_weight)
                )
                bsdf_direction = packed_si.to_world(bs.wo)
                child_ray = packed_si.spawn_ray(bsdf_direction)
                next_si = scene.ray_intersect(
                    child_ray,
                    valid_bsdf & can_sample_emitter,
                )
                ds_hit = mi.DirectionSample3f(scene, next_si, packed_si)
                emitter_pdf = scene.pdf_emitter_direction(
                    packed_si,
                    ds_hit,
                    valid_bsdf & can_sample_emitter,
                )
                mixture_pdf = (
                    (1.0 - emitter_probability) * bs.pdf
                    + emitter_probability * emitter_pdf
                )
                numerator = bsdf_weight * bs.pdf
                child_active = (
                    valid_bsdf
                    & (mixture_pdf > 0.0)
                    & finite(numerator)
                )

                child_stats = DeviceStats()
                if not recursive:
                    # Paper-aligned first-hit-only estimator: evaluate the
                    # full physical suffix, but do not recursively stylize it.
                    incoming = self._plain_radiance(
                        scene,
                        packed_sampler,
                        child_ray,
                        depth + 1,
                        child_stats,
                        child_active,
                    )
                else:
                    packed_auxiliary = None
                    if (
                        self.feature_lines is not None
                        and self.config.feature_lines.can_apply_from(depth + 1)
                    ):
                        packed_auxiliary = self.feature_lines.extend(
                            scene,
                            auxiliary_frame.gather(parent_indices),
                            [
                                pack(
                                    mi.Bool,
                                    valid,
                                    "packed feature-line continuation",
                                )
                                for valid in line_continuation
                            ],
                            packed_si.sh_frame.n,
                            packed_ray.d,
                            bsdf_direction,
                            bs.sampled_type,
                            bs.eta,
                            child_active,
                            depth + 1,
                        )

                    packed_tone = None
                    if (
                        self.tone_mapper is not None
                        and self.config.tone_mapping.can_extend_from(depth)
                    ):
                        if tone_child_frame is None:
                            tone_child_frame = self.tone_mapper.extend(
                                scene,
                                tone_frame,
                                si,
                                surface_active,
                            )
                        packed_tone = tone_child_frame.gather(
                            parent_indices
                        ).eval()
                    packed_tone_analytic = (
                        pack(
                            mi.Bool,
                            tone_analytic,
                            "packed analytic tone mask",
                        )
                        & mi.has_flag(
                            bs.sampled_type,
                            mi.BSDFFlags.DeltaReflection,
                        )
                    )
                    incoming = self._radiance(
                        scene,
                        packed_sampler,
                        child_ray,
                        depth + 1,
                        next_visits,
                        child_stats,
                        child_active,
                        packed_auxiliary,
                        packed_tone,
                        packed_tone_analytic,
                    )
                transport = dr.select(
                    child_active,
                    numerator * incoming
                    / dr.maximum(mixture_pdf, 1e-20),
                    0.0,
                )
                if depth >= self.rr_depth:
                    transport /= self.rr_probability

                for name in (
                    "tree_nodes",
                    "draws",
                    "style_evaluations",
                    "inner_variance",
                    "estimated_bias",
                ):
                    value = getattr(child_stats, name)
                    if dr.width(value) == 1:
                        value = dr.repeat(value, packed_width)
                    setattr(
                        stats,
                        name,
                        getattr(stats, name)
                        + dr.block_sum(value, sample_count),
                    )
                result = self._lighting_result(
                    packed_emission,
                    direct + transport,
                    depth,
                    packed_si.t,
                )
                return dr.select(packed_draw_active, result, 0.0)

            def make_packed_draw(next_visits, recursive=True):
                def draw(draw_active, sample_count):
                    return sample_integrand_packed(
                        draw_active,
                        next_visits,
                        sample_count,
                        recursive=recursive,
                    )

                return draw

            processed = mi.Bool(False)
            for material_id, entries in self._tracked_materials.items():
                previous = visits.get(material_id, 0)
                relevant = [
                    binding
                    for _, _, binding in entries
                    if not isinstance(binding.estimator, IdentityEstimator)
                    and self._can_match_from(binding.predicate, depth, previous)
                ]
                if not relevant:
                    continue

                # Most exported shapes share a material object. Resolve the
                # material with a handful of BSDF pointer comparisons instead
                # of testing every shape (20 BSDFs versus 933 shapes in F10).
                material_active = surface_active & (
                    bsdf == self._tracked_bsdfs[material_id]
                )
                if not self._any(material_active):
                    continue

                next_visits = visits.copy()
                occurrence = previous + 1
                next_visits[material_id] = occurrence
                material_processed = mi.Bool(False)
                has_plain_binding = False
                groups = list(self._binding_groups(entries))
                for binding, shapes in groups:
                    if len(groups) == 1:
                        binding_active = material_active
                    else:
                        binding_active = mi.Bool(False)
                        for shape, _ in shapes:
                            binding_active |= material_active & (si.shape == shape)
                    context = StyleContext(
                        depth=depth,
                        position=si.p,
                        normal=si.sh_frame.n,
                        uv=si.uv,
                        material_id=material_id,
                        shape_id=shapes[0][1],
                        occurrence=occurrence,
                        tone_coordinate=tone_coordinate,
                        tone_valid=tone_valid,
                        tone_confidence=tone_confidence,
                        tone_inversion_method=tone_method,
                    )
                    if (
                        not isinstance(binding.estimator, IdentityEstimator)
                        and binding.predicate.matches(context)
                    ):
                        if len(groups) > 1 and not self._any(binding_active):
                            continue
                        result += wavefront.estimate(
                            binding.estimator,
                            make_draw(next_visits),
                            context,
                            stats,
                            binding_active,
                            packed_draw=make_packed_draw(
                                next_visits,
                                recursive=getattr(
                                    binding.estimator, "recursive", True
                                ),
                            ),
                        )
                        material_processed |= binding_active
                        # A crop may contain several styled materials. Keep
                        # their complete nested expectation graphs strictly
                        # sequential: without this boundary, ``result +=``
                        # retains every preceding material's 8^depth DFS graph
                        # until the entire wavefront returns.
                        wavefront._materialize(
                            stats, result, material_processed
                        )
                    else:
                        has_plain_binding = True

                if has_plain_binding:
                    plain_material = material_active & (material_processed == False)
                    result += wavefront.estimate(
                        IdentityEstimator(),
                        make_draw(next_visits),
                        StyleContext(
                            depth=depth,
                            position=si.p,
                            normal=si.sh_frame.n,
                            uv=si.uv,
                            material_id=material_id,
                            occurrence=occurrence,
                            tone_coordinate=tone_coordinate,
                            tone_valid=tone_valid,
                            tone_confidence=tone_confidence,
                            tone_inversion_method=tone_method,
                        ),
                        stats,
                        plain_material,
                        packed_draw=make_packed_draw(next_visits),
                    )
                    wavefront._materialize(stats, result)
                processed |= material_active
                wavefront._materialize(stats, result, processed)

            fallback_active = surface_active & (processed == False)
            result += wavefront.estimate(
                IdentityEstimator(),
                make_draw(visits),
                StyleContext(
                    depth=depth,
                    position=si.p,
                    normal=si.sh_frame.n,
                    uv=si.uv,
                    tone_coordinate=tone_coordinate,
                    tone_valid=tone_valid,
                    tone_confidence=tone_confidence,
                    tone_inversion_method=tone_method,
                ),
                stats,
                fallback_active,
                packed_draw=make_packed_draw(visits),
            )
            return result

        def sample(self, scene, sampler, ray, medium=None, active=True):
            del medium
            self._prepare_materials(scene)
            active = mi.Bool(active)
            primary = self._ray_intersect(scene, ray, active)
            valid = active & primary.is_valid()
            stats = DeviceStats()
            auxiliary_frame = (
                # Camera misses cannot contain a line or continue to a deeper
                # path vertex. Mask them before all N_aux BVH traversals.
                self.feature_lines.spawn_primary(
                    scene,
                    sampler,
                    ray,
                    valid,
                    primary.bsdf(ray),
                    primary.shape,
                    primary.p,
                )
                if self.feature_lines is not None else None
            )
            tone_frame = None
            if self.tone_mapper is not None:
                # Supplemental S2.3 constructs the material-independent
                # perfect-mirror projection paths once, before evaluating the
                # stylized transport tree.  Keeping this prepass outside
                # ``_radiance()`` avoids retracing all tone anchors for every
                # direct-estimator draw at every recursive depth.
                tone_frame = self.tone_mapper.spawn_primary(scene, ray, valid)
                tone_frame = self.tone_mapper.precompute_projection_chain(
                    scene, tone_frame
                )
            radiance = self._radiance(
                scene,
                sampler,
                ray,
                0,
                {},
                stats,
                active,
                auxiliary_frame,
                tone_frame,
                mi.Bool(True),
                interaction=primary,
            )
            return radiance, valid, [
                stats.tree_nodes,
                stats.style_evaluations,
                stats.inner_variance,
                stats.estimated_bias,
            ]

        def to_string(self):
            return (
                f"CudaSREIntegrator[max_depth={self.max_depth}, "
                f"rr_depth={self.rr_depth}, "
                f"rr_probability={self.rr_probability}]"
            )

    mi.register_integrator("sre_cuda", lambda props: CudaSREIntegrator(props))
    _REGISTERED = True


__all__ = ["register_cuda_integrator"]

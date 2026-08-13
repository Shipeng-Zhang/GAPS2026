"""Device-side style functions and expectation estimators for CUDA SRE."""

from __future__ import annotations

from math import comb, cos, sin
import os

import drjit as dr
import mitsuba as mi


_TRACED_BROADCAST_GATHERS: set[str] = set()

try:
    from .estimators import (
        AdditionEstimator,
        CompositionEstimator,
        ConstantEstimator,
        DirectApplicationEstimator,
        GammaPowerSeriesEstimator,
        IdentityEstimator,
        MultiplicationEstimator,
        PolynomialEstimator,
        TelescopingEstimator,
    )
    from .styles import (
        Cel, ColorMap, ColorMap_Nonlinear, CrossHatch, Gamma, Gooch,
        Halftone, Identity, Saturation, TieDye, ToneHatch, ToneHalftone,
    )
except ImportError:
    from estimators import (
        AdditionEstimator,
        CompositionEstimator,
        ConstantEstimator,
        DirectApplicationEstimator,
        GammaPowerSeriesEstimator,
        IdentityEstimator,
        MultiplicationEstimator,
        PolynomialEstimator,
        TelescopingEstimator,
    )
    from styles import (
        Cel, ColorMap, ColorMap_Nonlinear, CrossHatch, Gamma, Gooch,
        Halftone, Identity, Saturation, TieDye, ToneHatch, ToneHalftone,
    )


def color(value):
    if isinstance(value, mi.Color3f):
        return value
    try:
        if len(value) == 3:
            return mi.Color3f(float(value[0]), float(value[1]), float(value[2]))
    except TypeError:
        pass
    return mi.Color3f(value)


def luminance(value):
    value = color(value)
    return 0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2]


def max_abs(value):
    value = dr.abs(color(value))
    return dr.maximum(value[0], dr.maximum(value[1], value[2]))


def finite(value):
    value = color(value)
    return dr.isfinite(value[0]) & dr.isfinite(value[1]) & dr.isfinite(value[2])


def gather_lanes(dtype, source, indices, expected_width=None, label="value"):
    """Gather a compact wavefront without indexing broadcast constants.

    Dr.Jit represents a lane-invariant value with width one. CUDA ``gather``
    does not implicitly broadcast such an allocation: indexing it with the
    original wavefront's compressed indices is an out-of-bounds read. This is
    intermittent because constant folding depends on the tile's material and
    path distribution. Explicitly repeat width-one values and reject every
    other source/parent width mismatch before submitting a CUDA kernel.
    """
    source_width = dr.width(source)
    index_count = dr.width(indices)
    if expected_width is not None and source_width not in (1, expected_width):
        raise RuntimeError(
            f"{label} has lane width {source_width}; expected a broadcast "
            f"constant or parent width {expected_width}"
        )
    if source_width == 1:
        if (
            os.environ.get("SRE_TRACE_BROADCAST_GATHERS") == "1"
            and label not in _TRACED_BROADCAST_GATHERS
        ):
            _TRACED_BROADCAST_GATHERS.add(label)
            print(
                f"[SRE gather] broadcast {label}: 1 -> {index_count} lanes",
                flush=True,
            )
        return dr.repeat(source, index_count)
    return dr.gather(dtype, source, indices)


class FrozenRayIntersector:
    """Replay stable OptiX traversal graphs across a recursive SRE tree.

    DFS control flow itself cannot be frozen because it reads compressed lane
    counts on the host. A single ray query has fixed topology, however. Keeping
    one recording per RayFlags/coherency pair prevents every estimator draw and
    every auxiliary anchor from compiling another equivalent OptiX program.
    """

    def __init__(self, scene):
        self.scene = scene
        self._recordings = {}
        # Set by the first camera-width query in a fresh tile process. Only
        # this stable width is safe to replay: DFS-compacted widths depend on
        # the random seed and can change on every pass.
        self._root_width = None

    def ray_intersect(
        self,
        ray,
        ray_flags=None,
        coherent=False,
        active=True,
    ):
        flags = int(mi.RayFlags.All if ray_flags is None else ray_flags)
        coherent = bool(coherent)
        if mi.variant() != "cuda_ad_rgb" or dr.flag(dr.JitFlag.Debug):
            return self.scene.ray_intersect(ray, flags, coherent, active)

        ray = mi.Ray3f(ray)
        width = max(dr.width(ray.o), dr.width(ray.d), dr.width(active))
        if self._root_width is None:
            self._root_width = width
        elif width != self._root_width:
            # SurfaceInteraction contains polymorphic shape/emitter arrays.
            # dr.freeze can replay their old allocation width even when the
            # numeric ray inputs retrace correctly, which pairs an N-wide SI
            # with a previous M-wide virtual-call pointer array. Compact DFS
            # widths are also unlikely to repeat, so recording them provides
            # no useful cache hit. Keep only root-width camera/line/MLS
            # traversals frozen and execute compact queries explicitly.
            return self.scene.ray_intersect(ray, flags, coherent, active)

        key = (flags, coherent)
        frozen = self._recordings.get(key)
        if frozen is None:
            scene = self.scene

            @dr.freeze(
                state_fn=lambda origin, direction, maxt, time, mask: (scene,),
                backend=dr.JitBackend.CUDA,
                limit=4,
                # Compacted DFS levels legitimately produce several input
                # allocation layouts. They still replay the same small set of
                # compiled kernels (verified through KernelHistory), so avoid
                # flooding worker logs with benign retracing diagnostics.
                warn_after=1000,
            )
            def frozen(origin, direction, maxt, time, mask):
                frozen_ray = mi.Ray3f()
                frozen_ray.o = origin
                frozen_ray.d = direction
                frozen_ray.maxt = maxt
                frozen_ray.time = time
                return scene.ray_intersect(
                    frozen_ray, flags, coherent, mask
                )

            self._recordings[key] = frozen
        # Ray3f is ragged in cuda_ad_rgb: o/d normally have N lanes while
        # maxt/time and masks may be width-one broadcasts. Frozen-function
        # inputs are structural, so normalize these fields to a single width
        # and pass only what OptiX consumes. This avoids a new recording for
        # every scalar/full-width combination encountered after compaction.
        def normalize(dtype, value):
            value_width = dr.width(value)
            if value_width == width:
                return dtype(value)
            if value_width == 1:
                return dr.repeat(dtype(value), width)
            raise RuntimeError(
                "frozen ray input has incompatible lane width "
                f"{value_width}; expected 1 or {width}"
            )

        return frozen(
            normalize(mi.Point3f, ray.o),
            normalize(mi.Vector3f, ray.d),
            normalize(mi.Float, ray.maxt),
            normalize(mi.Float, ray.time),
            normalize(mi.Bool, active),
        )


def evaluate_style(function, value, context):
    value = color(value)
    if isinstance(function, Identity):
        return value
    if isinstance(function, Gamma):
        return dr.power(dr.maximum(value, 0.0), function.exponent)
    if isinstance(function, Saturation):
        gray = luminance(value)
        return gray + function.amount * (value - gray)
    if isinstance(function, ColorMap_Nonlinear):
        t = dr.clamp(
            (luminance(value) - function.low)
            / (function.high - function.low),
            0.0,
            1.0,
        )
        result = color(function.colors[0])
        for index in range(len(function.positions) - 1):
            low = float(function.positions[index])
            high = float(function.positions[index + 1])
            local = dr.clamp((t - low) / (high - low), 0.0, 1.0)
            weight = local * local * (3.0 - 2.0 * local)
            segment = dr.lerp(
                color(function.colors[index]),
                color(function.colors[index + 1]),
                weight,
            )
            result = dr.select(t >= low, segment, result)
        return result
    if isinstance(function, ColorMap):
        t = dr.clamp(
            (luminance(value) - function.low) / (function.high - function.low),
            0.0, 1.0,
        )
        result = color(function.colors[0])
        for index in range(len(function.positions) - 1):
            low = float(function.positions[index])
            high = float(function.positions[index + 1])
            weight = dr.clamp((t - low) / (high - low), 0.0, 1.0)
            segment = dr.lerp(
                color(function.colors[index]),
                color(function.colors[index + 1]),
                weight,
            )
            result = dr.select(t >= low, segment, result)
        return result
    if isinstance(function, Cel):
        weights = function.brightness_weights
        current = (
            float(weights[0]) * value[0]
            + float(weights[1]) * value[1]
            + float(weights[2]) * value[2]
        )
        normalized = dr.clamp(current / function.max_value, 0.0, 1.0)
        index = mi.UInt32(0)
        for threshold in function.thresholds:
            index += mi.UInt32(normalized >= float(threshold))
        if function.palette is not None:
            result = color(function.palette[0])
            for level in range(1, function.levels):
                result = dr.select(index == level, color(function.palette[level]), result)
            if function.chroma_strength > 0.0:
                target = (
                    float(weights[0]) * result[0]
                    + float(weights[1]) * result[1]
                    + float(weights[2]) * result[2]
                )
                chroma = value - current
                chroma_weight = mi.Float(float(function.chroma_weights[0]))
                for level in range(1, function.levels):
                    chroma_weight = dr.select(
                        index == level,
                        float(function.chroma_weights[level]),
                        chroma_weight,
                    )
                result += function.chroma_strength * chroma_weight * chroma * (
                    target / dr.maximum(current, 1e-8)
                )
                result = dr.maximum(result, 0.0)
            return result
        target = mi.Float(0.0)
        for level in range(function.levels):
            target = dr.select(
                index == level,
                float(function.band_values[level]) * function.max_value,
                target,
            )
        if not function.preserve_chroma:
            return mi.Color3f(target)
        return value * (target / dr.maximum(current, 1e-8))
    if isinstance(function, CrossHatch):
        normalized = dr.clamp(
            luminance(value) / function.max_value, 0.0, 1.0
        )
        darkness = dr.power(1.0 - normalized, function.darkness_gamma)
        coverage = mi.Float(0.0)
        point = context.position
        for index, direction in enumerate(function.directions):
            active = darkness >= float(function.activation_thresholds[index])
            phase = function.scale * float(function.scale_factors[index]) * (
                float(direction[0]) * point[0]
                + float(direction[1]) * point[1]
                + float(direction[2]) * point[2]
            ) + float(function.phase_offsets[index])
            phase -= dr.floor(phase)
            distance = dr.minimum(phase, 1.0 - phase)
            half_width = (
                function.width
                * float(function.family_widths[index])
                * (1.0 + function.width_growth * darkness)
            )
            if function.edge_softness > 0.0:
                weight = dr.clamp(
                    (half_width + function.edge_softness - distance)
                    / (2.0 * function.edge_softness),
                    0.0,
                    1.0,
                )
                weight = weight * weight * (3.0 - 2.0 * weight)
            else:
                weight = dr.select(distance < half_width, 1.0, 0.0)
            coverage = dr.maximum(coverage, dr.select(active, weight, 0.0))
        return dr.lerp(color(function.paper), color(function.ink), coverage)
    if isinstance(function, Halftone):
        normalized = dr.clamp(
            luminance(value) / function.max_value, 0.0, 1.0
        )
        darkness = 1.0 - normalized
        tone = dr.clamp(
            (darkness - function.dot_threshold)
            / (1.0 - function.dot_threshold),
            0.0,
            1.0,
        )
        radius_weight = dr.power(tone, function.radius_gamma)
        radius = function.min_radius + (
            function.max_radius - function.min_radius
        ) * radius_weight
        point = context.position
        lattice = []
        for row in function.orientation:
            lattice.append(
                function.scale * (
                    float(row[0]) * point[0]
                    + float(row[1]) * point[1]
                    + float(row[2]) * point[2]
                )
            )
        cell = [
            lattice[index]
            + float(function.phase[index])
            - dr.floor(lattice[index] + float(function.phase[index]))
            - 0.5
            for index in range(3)
        ]
        distance = dr.sqrt(
            dr.square(cell[0]) + dr.square(cell[1]) + dr.square(cell[2])
        )
        if function.edge_softness > 0.0:
            coverage = dr.clamp(
                (radius + function.edge_softness - distance)
                / (2.0 * function.edge_softness),
                0.0,
                1.0,
            )
            coverage = coverage * coverage * (3.0 - 2.0 * coverage)
        else:
            coverage = dr.select(distance < radius, 1.0, 0.0)
        coverage = dr.select(
            darkness > function.dot_threshold, coverage, 0.0
        )
        ink_strength = function.min_ink_strength + (
            1.0 - function.min_ink_strength
        ) * tone
        coverage *= ink_strength
        return dr.lerp(color(function.paper), color(function.ink), coverage)
    if isinstance(function, ToneHatch):
        if function.brightness_levels.size:
            current = (
                (value[0] + value[1] + value[2]) / 3.0
                if function.brightness_mode == "mean"
                else luminance(value)
            )
            target = mi.Float(float(function.brightness_levels[0]))
            for index, threshold in enumerate(function.brightness_thresholds):
                target = dr.select(
                    current >= float(threshold),
                    float(function.brightness_levels[index + 1]),
                    target,
                )
            value *= dr.select(
                current > 1e-8,
                target / dr.maximum(current, 1e-8),
                0.0,
            )
        normalized = dr.clamp(
            luminance(value) / function.max_value, 0.0, 1.0
        )
        darkness = dr.power(1.0 - normalized, function.darkness_gamma)
        coordinate = context.tone_coordinate
        coverage = mi.Float(0.0)
        for index, angle_degrees in enumerate(function.angles_degrees):
            angle = float(angle_degrees) * 0.017453292519943295
            normal_x = -sin(angle)
            normal_y = cos(angle)
            phase = (
                (normal_x * coordinate[0] + normal_y * coordinate[1])
                / function.spacing
                + float(function.phase_offsets[index])
            )
            phase -= dr.floor(phase)
            distance = dr.minimum(phase, 1.0 - phase) * function.spacing
            threshold = float(function.activation_thresholds[index])
            tone = dr.clamp(
                (darkness - threshold) / max(1.0 - threshold, 1e-8),
                0.0,
                1.0,
            )
            fraction = function.min_coverage + (
                function.max_coverage - function.min_coverage
            ) * tone
            half_width = (
                0.5 * function.spacing * fraction
                * float(function.family_widths[index])
            )
            if function.edge_softness > 0.0:
                weight = dr.clamp(
                    (half_width + function.edge_softness - distance)
                    / (2.0 * function.edge_softness),
                    0.0,
                    1.0,
                )
                weight = weight * weight * (3.0 - 2.0 * weight)
            else:
                weight = dr.select(distance < half_width, 1.0, 0.0)
            active_family = darkness >= threshold
            coverage = dr.maximum(
                coverage, dr.select(active_family, weight, 0.0)
            )
        region_weight = mi.Float(1.0)
        if function.region_center is not None:
            region_x = (
                coordinate[0] - float(function.region_center[0])
            ) / float(function.region_radius[0])
            region_y = (
                coordinate[1] - float(function.region_center[1])
            ) / float(function.region_radius[1])
            region_distance = dr.sqrt(
                dr.square(region_x) + dr.square(region_y)
            )
            if function.region_feather > 0.0:
                region_weight = dr.clamp(
                    (1.0 + function.region_feather - region_distance)
                    / function.region_feather,
                    0.0,
                    1.0,
                )
                region_weight = region_weight * region_weight * (
                    3.0 - 2.0 * region_weight
                )
            else:
                region_weight = dr.select(region_distance <= 1.0, 1.0, 0.0)
            coverage *= region_weight
        if function.shadow_strength > 0.0:
            shadow_threshold = float(function.activation_thresholds[0])
            shadow_tone = dr.clamp(
                (darkness - shadow_threshold)
                / max(1.0 - shadow_threshold, 1e-8),
                0.0,
                1.0,
            )
            shadow = function.shadow_strength * shadow_tone * region_weight
            base = dr.lerp(color(function.paper), color(function.ink), shadow)
        else:
            base = color(function.paper)
        return dr.lerp(base, color(function.ink), coverage)
    if isinstance(function, ToneHalftone):
        coordinate = context.tone_coordinate
        angle = function.angle_degrees * 0.017453292519943295
        cosine = cos(angle)
        sine = sin(angle)
        rotated_x = cosine * coordinate[0] - sine * coordinate[1]
        rotated_y = sine * coordinate[0] + cosine * coordinate[1]
        cell_x = (
            rotated_x / function.spacing + float(function.phase[0])
        )
        cell_y = (
            rotated_y / function.spacing + float(function.phase[1])
        )
        cell_x = (cell_x - dr.floor(cell_x) - 0.5) * function.spacing
        cell_y = (cell_y - dr.floor(cell_y) - 0.5) * function.spacing
        distance = dr.sqrt(dr.square(cell_x) + dr.square(cell_y))
        darkness = dr.clamp(
            1.0 - luminance(value) / function.max_value, 0.0, 1.0
        )
        tone = dr.clamp(
            (darkness - function.dot_threshold)
            / (1.0 - function.dot_threshold),
            0.0,
            1.0,
        )
        radius = function.min_radius + (
            function.max_radius - function.min_radius
        ) * dr.power(tone, function.radius_gamma)
        if function.edge_softness > 0.0:
            coverage = dr.clamp(
                (radius + function.edge_softness - distance)
                / (2.0 * function.edge_softness),
                0.0,
                1.0,
            )
            coverage = coverage * coverage * (3.0 - 2.0 * coverage)
        else:
            coverage = dr.select(distance < radius, 1.0, 0.0)
        coverage = dr.select(
            darkness > function.dot_threshold, coverage, 0.0
        )
        return dr.lerp(color(function.paper), color(function.ink), coverage)
    if isinstance(function, Gooch):
        t = dr.clamp(luminance(value) / function.max_value, 0.0, 1.0)
        return dr.lerp(color(function.cool), color(function.warm), t)
    if isinstance(function, TieDye):
        return function.offset + function.cosine_sign * function.amplitude * dr.cos(
            color(function.frequencies) * value + color(function.phases)
        )
    raise TypeError(f"No CUDA style implementation for {type(function).__name__}")


class DeviceStats:
    def __init__(self):
        self.tree_nodes = mi.Float(0.0)
        self.draws = mi.Float(0.0)
        self.style_evaluations = mi.Float(0.0)
        self.inner_variance = mi.Float(0.0)
        self.estimated_bias = mi.Float(0.0)


class WavefrontEstimator:
    # Keep the estimator mathematically and numerically sequential, but let
    # Dr.Jit submit a small group of consecutive draws together.  Evaluating
    # every draw separately turns direct(8) at three styled vertices into 512
    # host/device synchronization points per camera wavefront.  Eight retains
    # the original sample order and bounded-memory DFS behavior while matching
    # the inner sample count used by the paper/Fig. 13 configuration.
    _DIRECT_EVAL_BATCH_SIZE = 8
    _DIRECT_LANE_PACK_SIZE = 2

    def __init__(self, sampler):
        self.sampler = sampler

    @staticmethod
    def _counter(active):
        return dr.select(active, 1.0, 0.0)

    def _draw(self, draw, stats, active):
        stats.draws += self._counter(active)
        return dr.select(active, color(draw(active)), 0.0)

    @staticmethod
    def _materialize(stats, *values):
        dr.eval(
            *values,
            stats.tree_nodes,
            stats.draws,
            stats.style_evaluations,
            stats.inner_variance,
            stats.estimated_bias,
        )

    def _stream_mean(self, draw, count, stats, active):
        running_sum = mi.Color3f(0.0)
        for _ in range(count):
            sample = self._draw(draw, stats, active)
            running_sum += sample
            self._materialize(stats, running_sum)
            del sample
        return running_sum / count

    def _direct_jackknife(self, estimator, draw, context, stats, active):
        # Jackknife needs every leave-one-out sample. It remains an explicit
        # storage fallback; all default project configurations use none/delta.
        samples = []
        for _ in range(estimator.samples):
            sample = self._draw(draw, stats, active)
            dr.eval(sample)
            samples.append(sample)
        mean = mi.Color3f(0.0)
        for sample in samples:
            mean += sample
        mean /= estimator.samples
        result = evaluate_style(estimator.function, mean, context)
        total = mean * estimator.samples
        leave_one_out = mi.Color3f(0.0)
        for sample in samples:
            leave_one_out += evaluate_style(
                estimator.function,
                (total - sample) / (estimator.samples - 1),
                context,
            )
        stats.style_evaluations += self._counter(active)
        result = (
            estimator.samples * result
            - (estimator.samples - 1)
            * leave_one_out / estimator.samples
        )
        return dr.select(active, result, 0.0)

    def _direct(
        self, estimator, draw, context, stats, active, packed_draw=None
    ):
        if (
            estimator.bias_correction == "jackknife"
            and estimator.samples > 1
        ):
            return self._direct_jackknife(
                estimator, draw, context, stats, active
            )

        mean = mi.Color3f(0.0)
        covariance_sum = [
            [mi.Float(0.0) for _ in range(3)] for _ in range(3)
        ]
        seen = 0
        while seen < estimator.samples:
            pack_size = (
                estimator.samples
                if not getattr(estimator, "recursive", True)
                else self._DIRECT_LANE_PACK_SIZE
            )
            batch_size = min(
                pack_size,
                estimator.samples - seen,
            ) if packed_draw is not None else 1
            if batch_size > 1:
                packed = packed_draw(active, batch_size)
                parent_width = dr.width(active)
                packed_indices = dr.arange(
                    mi.UInt32, parent_width
                ) * batch_size
                samples = [
                    dr.gather(
                        mi.Color3f, packed, packed_indices + offset
                    )
                    for offset in range(batch_size)
                ]
                stats.draws += self._counter(active) * batch_size
            else:
                samples = [self._draw(draw, stats, active)]

            for sample in samples:
                seen += 1
                delta = sample - mean
                mean += delta / seen
                delta_after = sample - mean
                for a in range(3):
                    for b in range(3):
                        covariance_sum[a][b] += delta[a] * delta_after[b]
                del sample, delta, delta_after
            self._materialize(
                stats,
                mean,
                *[
                    covariance_sum[a][b]
                    for a in range(3) for b in range(3)
                ],
            )
            del samples

        result = evaluate_style(estimator.function, mean, context)
        stats.style_evaluations += self._counter(active)
        if estimator.samples == 1:
            return dr.select(active, result, 0.0)

        variance = mi.Color3f(*[
            covariance_sum[channel][channel] / (estimator.samples - 1)
            for channel in range(3)
        ])
        stats.inner_variance += self._counter(active) * (
            (variance[0] + variance[1] + variance[2])
            / (3.0 * estimator.samples)
        )

        epsilon = estimator.bias_epsilon
        covariance = [[mi.Float(0.0) for _ in range(3)] for _ in range(3)]
        for a in range(3):
            for b in range(3):
                covariance[a][b] = covariance_sum[a][b] / (
                    (estimator.samples - 1) * estimator.samples
                )

        f0 = evaluate_style(estimator.function, mean, context)
        hessian = [[None for _ in range(3)] for _ in range(3)]
        for a in range(3):
            ea = [0.0, 0.0, 0.0]
            ea[a] = epsilon
            ea = color(ea)
            hessian[a][a] = (
                evaluate_style(estimator.function, mean + ea, context)
                - 2.0 * f0
                + evaluate_style(estimator.function, mean - ea, context)
            ) / epsilon**2
            for b in range(a + 1, 3):
                eb = [0.0, 0.0, 0.0]
                eb[b] = epsilon
                eb = color(eb)
                mixed = (
                    evaluate_style(estimator.function, mean + ea + eb, context)
                    - evaluate_style(estimator.function, mean + ea - eb, context)
                    - evaluate_style(estimator.function, mean - ea + eb, context)
                    + evaluate_style(estimator.function, mean - ea - eb, context)
                ) / (4.0 * epsilon**2)
                hessian[a][b] = mixed
                hessian[b][a] = mixed
        bias = mi.Color3f(0.0)
        for a in range(3):
            for b in range(3):
                bias += 0.5 * hessian[a][b] * covariance[a][b]
        stats.estimated_bias += self._counter(active) * (
            (dr.abs(bias[0]) + dr.abs(bias[1]) + dr.abs(bias[2])) / 3.0
        )
        if estimator.bias_correction == "delta":
            result -= bias
        return dr.select(active, result, 0.0)

    def _polynomial(self, estimator, draw, context, stats, active):
        if estimator.evaluation_precision == "float64":
            return self._polynomial_float64(
                estimator, draw, context, stats, active
            )
        degree = estimator.degree
        coefficients = estimator.coefficients
        if estimator.projection == "luminance":
            elementary = [mi.Float(0.0) for _ in range(degree + 1)]
            elementary[0] = mi.Float(1.0)
            for seen in range(1, estimator.sample_count + 1):
                sample = self._draw(draw, stats, active)
                if estimator.clamp_samples:
                    sample = dr.clamp(sample, *estimator.fit_interval)
                if estimator.normalized_domain:
                    low, high = estimator.fit_interval
                    sample = 2.0 * (sample - low) / (high - low) - 1.0
                scalar = luminance(sample)
                for order in range(min(seen, degree), 0, -1):
                    elementary[order] += scalar * elementary[order - 1]
                self._materialize(stats, *elementary)
                del sample, scalar
            result = mi.Color3f(0.0)
            for order in range(degree + 1):
                result += color(coefficients[order]) * (
                    elementary[order] / comb(estimator.sample_count, order)
                )
        else:
            elementary = [mi.Color3f(0.0) for _ in range(degree + 1)]
            elementary[0] = mi.Color3f(1.0)
            for seen in range(1, estimator.sample_count + 1):
                sample = self._draw(draw, stats, active)
                if estimator.clamp_samples:
                    sample = dr.clamp(sample, *estimator.fit_interval)
                if estimator.normalized_domain:
                    low, high = estimator.fit_interval
                    sample = 2.0 * (sample - low) / (high - low) - 1.0
                for order in range(min(seen, degree), 0, -1):
                    elementary[order] += sample * elementary[order - 1]
                self._materialize(stats, *elementary)
                del sample
            result = mi.Color3f(0.0)
            for order in range(degree + 1):
                result += color(coefficients[order]) * (
                    elementary[order] / comb(estimator.sample_count, order)
                )
        stats.style_evaluations += self._counter(active)
        return dr.select(active, result, 0.0)

    def _polynomial_float64(self, estimator, draw, context, stats, active):
        """Evaluate high-order power-basis cancellation in CUDA Float64."""
        degree = estimator.degree
        coefficients = estimator.coefficients
        elementary = [
            [mi.Float64(0.0) for _ in range(3)]
            for _ in range(degree + 1)
        ]
        elementary[0] = [mi.Float64(1.0) for _ in range(3)]
        for seen in range(1, estimator.sample_count + 1):
            sample = self._draw(draw, stats, active)
            if estimator.clamp_samples:
                sample = dr.clamp(sample, *estimator.fit_interval)
            sample64 = [mi.Float64(sample[channel]) for channel in range(3)]
            if estimator.normalized_domain:
                low, high = estimator.fit_interval
                sample64 = [
                    2.0 * (channel - low) / (high - low) - 1.0
                    for channel in sample64
                ]
            if estimator.projection == "luminance":
                scalar = (
                    0.2126 * sample64[0]
                    + 0.7152 * sample64[1]
                    + 0.0722 * sample64[2]
                )
                sample64 = [scalar, scalar, scalar]
            for order in range(min(seen, degree), 0, -1):
                for channel in range(3):
                    elementary[order][channel] += (
                        sample64[channel] * elementary[order - 1][channel]
                    )
            self._materialize(
                stats,
                *[
                    elementary[order][channel]
                    for order in range(degree + 1)
                    for channel in range(3)
                ],
            )
            del sample, sample64

        result64 = [mi.Float64(0.0) for _ in range(3)]
        for order in range(degree + 1):
            coefficient = coefficients[order]
            if coefficients.ndim == 1:
                coefficient = [coefficient] * 3
            denominator = comb(estimator.sample_count, order)
            for channel in range(3):
                result64[channel] += (
                    float(coefficient[channel])
                    * elementary[order][channel]
                    / denominator
                )
        result = mi.Color3f(*[mi.Float(channel) for channel in result64])
        stats.style_evaluations += self._counter(active)
        return dr.select(active, result, 0.0)

    def _random_degree(self, active, probability):
        degree = mi.UInt32(0)
        continuing = mi.Bool(active)
        while True:
            continuing &= self.sampler.next_1d() < probability
            if not bool(dr.any(continuing)):
                break
            degree += dr.select(continuing, 1, 0)
        return degree

    @staticmethod
    def _maximum(value):
        return int(dr.slice(dr.max(value)))

    def _gamma(self, estimator, draw, context, stats, active):
        if estimator.pilot_samples:
            expansion = dr.maximum(
                self._stream_mean(
                    draw, estimator.pilot_samples, stats, active
                ),
                estimator.min_expansion_point,
            )
        else:
            expansion = mi.Color3f(
                max(estimator.expansion_point, estimator.min_expansion_point)
            )
        degree = self._random_degree(active, estimator.continuation_probability)
        max_degree = self._maximum(degree)
        count = dr.maximum(mi.UInt32(1), degree * estimator.oversampling)
        max_count = max(1, max_degree * estimator.oversampling)
        elementary = [mi.Color3f(0.0) for _ in range(max_degree + 1)]
        elementary[0] = mi.Color3f(1.0)
        for seen in range(1, max_count + 1):
            sample_active = active & (count >= seen)
            sample = self._draw(draw, stats, sample_active) - expansion
            for order in range(min(seen, max_degree), 0, -1):
                updated = elementary[order] + sample * elementary[order - 1]
                elementary[order] = dr.select(sample_active, updated, elementary[order])
            self._materialize(stats, *elementary)
            del sample
        result = mi.Color3f(0.0)
        falling = 1.0
        factorial = 1.0
        combinations = mi.Float(1.0)
        count_float = mi.Float(count)
        for order in range(max_degree + 1):
            if order:
                falling *= estimator.exponent - (order - 1)
                factorial *= order
                combinations *= (count_float - (order - 1)) / order
            coefficient = (
                dr.power(expansion, estimator.exponent - order)
                * falling / factorial
            )
            result += dr.select(
                active & (degree >= order),
                coefficient * elementary[order]
                / dr.maximum(combinations, 1.0)
                / estimator.continuation_probability**order,
                0.0,
            )
        stats.style_evaluations += self._counter(active)
        return dr.select(active, result, 0.0)

    def _telescoping(self, estimator, draw, context, stats, active):
        running_sum = mi.Color3f(0.0)
        for _ in range(estimator.base_samples):
            sample = self._draw(draw, stats, active)
            running_sum += sample
            self._materialize(stats, running_sum)
            del sample
        previous = evaluate_style(
            estimator.function, running_sum / estimator.base_samples, context
        )
        stats.style_evaluations += self._counter(active)
        result = previous
        corrections = self._random_degree(active, estimator.continuation_probability)
        for correction in range(1, self._maximum(corrections) + 1):
            correction_active = active & (corrections >= correction)
            running_sum += self._draw(draw, stats, correction_active)
            current = evaluate_style(
                estimator.function,
                running_sum / (estimator.base_samples + correction),
                context,
            )
            stats.style_evaluations += self._counter(correction_active)
            result += dr.select(
                correction_active,
                (current - previous) / estimator.continuation_probability**correction,
                0.0,
            )
            previous = dr.select(correction_active, current, previous)
            self._materialize(stats, running_sum, previous, result)
        return dr.select(active, result, 0.0)

    def estimate(
        self, estimator, draw, context, stats, active, packed_draw=None
    ):
        if isinstance(estimator, IdentityEstimator):
            return self._draw(draw, stats, active)
        if isinstance(estimator, ConstantEstimator):
            stats.style_evaluations += self._counter(active)
            return dr.select(active, color(estimator.value), 0.0)
        if isinstance(estimator, DirectApplicationEstimator):
            return self._direct(
                estimator,
                draw,
                context,
                stats,
                active,
                packed_draw=packed_draw,
            )
        if isinstance(estimator, PolynomialEstimator):
            return self._polynomial(estimator, draw, context, stats, active)
        if isinstance(estimator, GammaPowerSeriesEstimator):
            return self._gamma(estimator, draw, context, stats, active)
        if isinstance(estimator, TelescopingEstimator):
            return self._telescoping(estimator, draw, context, stats, active)
        if isinstance(estimator, AdditionEstimator):
            # Addition does not require independent or shared samples for
            # unbiasedness. Independent sequential evaluation avoids replay
            # caches whose size grows with the larger child estimator.
            left = self.estimate(estimator.left, draw, context, stats, active)
            self._materialize(stats, left)
            right = self.estimate(estimator.right, draw, context, stats, active)
            return left + right
        if isinstance(estimator, MultiplicationEstimator):
            left = self.estimate(
                estimator.left, draw, context, stats, active
            )
            self._materialize(stats, left)
            right = self.estimate(
                estimator.right, draw, context, stats, active
            )
            return left * right
        if isinstance(estimator, CompositionEstimator):
            return self.estimate(
                estimator.outer,
                lambda nested_active: self.estimate(
                    estimator.inner, draw, context, stats, nested_active
                ),
                context, stats, active,
            )
        raise TypeError(f"No CUDA estimator for {type(estimator).__name__}")


__all__ = [
    "DeviceStats", "WavefrontEstimator", "finite", "gather_lanes", "max_abs",
]

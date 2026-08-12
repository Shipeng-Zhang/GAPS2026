from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config import load_config
    from .estimators import EstimateStats
    from .feature_lines import (
        FeatureLineType,
        minimal_rotation,
        normalize,
        normal_finite_difference_slope,
        parallel_transport_half_vector,
        view_oriented_frame,
    )
    from .styles import StyleContext, as_rgb
    from .tone_mapping import ScalarToneMapper
except ImportError:
    from config import load_config
    from estimators import EstimateStats
    from feature_lines import (
        FeatureLineType,
        minimal_rotation,
        normalize,
        normal_finite_difference_slope,
        parallel_transport_half_vector,
        view_oriented_frame,
    )
    from styles import StyleContext, as_rgb
    from tone_mapping import ScalarToneMapper


_REGISTERED = False


@dataclass
class TraceStats:
    tree_nodes: int = 0 # 树节点访问数
    style_evaluations: int = 0 # 风格函数评估次数
    inner_variance: float = 0.0 # 内部方差
    estimated_bias: float = 0.0 # 估计偏差

# 记录光线传播碰到了哪些物质
@dataclass
class PathState:
    visits: dict[str, int] = field(default_factory=dict) # 材质访问计数器

    def enter(self, material_id: str) -> tuple["PathState", int]:
        visits = self.visits.copy()
        occurrence = visits.get(material_id, 0) + 1
        visits[material_id] = occurrence
        return PathState(visits), occurrence


@dataclass
class ScalarAuxiliarySample:
    offset: np.ndarray
    ray: Any
    interaction: Any
    prefix_valid: bool = True


@dataclass
class ScalarAuxiliaryFrame:
    samples: list[ScalarAuxiliarySample] = field(default_factory=list)


class SamplerRandom:
    def __init__(self, sampler):
        self.sampler = sampler

    def random(self):
        return float(self.sampler.next_1d())

# 注册SRE采样积分器
def register_sre_integrator():
    global _REGISTERED
    if _REGISTERED:
        return
    import mitsuba as mi

    class SREIntegrator(mi.SamplingIntegrator):
        def __init__(self, props):
            super().__init__(props)
            self.max_depth = int(props.get("max_depth", 6)) # 最大路径深度
            self.rr_depth = int(props.get("rr_depth", 5)) # 俄式轮盘赌触发深度
            self.rr_probability = float(props.get("rr_probability", 0.95)) # 俄式轮盘赌存活概率
            config_path = str(props.get("style_config", "")) # 风格配置文件路径
            self.config = load_config(Path(config_path)) if config_path else load_config(None) # 加载风格配置
            self.feature_line_config = self.config.feature_lines
            self.tone_mapper = (
                ScalarToneMapper(self.config.tone_mapping)
                if self.config.tone_mapping.enabled else None
            )
            if self.max_depth < 1:
                raise ValueError("max_depth must be at least one")
            if not 0.0 < self.rr_probability <= 1.0:
                raise ValueError("rr_probability must be in (0, 1]")

        # 任意输出变量 
        def aov_names(self):
            return [
                "sre_tree_nodes", # 树节点耗时图层
                "sre_style_evaluations", # 风格函数评估图层
                "sre_inner_variance", # 内部方差图层
                "sre_estimated_bias", # 估计残差图层
            ]

        # 颜色数据标准化
        @staticmethod
        def _rgb(value):
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if len(array) == 1:
                array = np.repeat(array, 3)
            return as_rgb(array[:3])

        # 将RGB数组转换为mi颜色类型对象
        @staticmethod
        def _spectrum(value):
            value = as_rgb(value)
            return mi.Color3f(float(value[0]), float(value[1]), float(value[2]))

        # 评估光线相交点是否自发光
        @staticmethod
        def _emission(scene, si):
            emitter = si.emitter(scene) # 相交点表面是否挂载了发光器
            if emitter is None:
                return np.zeros(3)
            return SREIntegrator._rgb(emitter.eval(si))

        @staticmethod
        def _concentric_disk(first: float, second: float) -> np.ndarray:
            x = 2.0 * first - 1.0
            y = 2.0 * second - 1.0
            if x == 0.0 and y == 0.0:
                return np.zeros(2)
            if abs(x) > abs(y):
                radius = x
                angle = np.pi * 0.25 * y / x
            else:
                radius = y
                angle = np.pi * 0.5 - np.pi * 0.25 * x / y
            return radius * np.array([np.cos(angle), np.sin(angle)])

        def _spawn_auxiliary_paths(
            self,
            scene,
            sampler,
            ray,
            material_id: str | None = None,
            shape_id: str | None = None,
            base_position: np.ndarray | None = None,
        ) -> ScalarAuxiliaryFrame:
            count = self.feature_line_config.auxiliary_samples
            active_lines = [
                line
                for line in self.feature_line_config.types
                if material_id is not None
                and line.applies_to_material(material_id)
                and (shape_id is None or line.applies_to_shape(shape_id))
            ]
            warped = next(
                (
                    line for line in active_lines
                    if np.any(line.stencil_warp_amplitude > 0.0)
                ),
                None,
            )
            sampling_center = (
                warped.stencil_warp(base_position)
                if warped is not None and base_position is not None
                else np.zeros(2, dtype=np.float64)
            )
            material_radii = [
                (
                    line.centered_sampling_radius
                    if warped is not None else line.sampling_radius
                )
                for line in active_lines
            ]
            anchor_radii = [
                line.sampling_radius
                for line in active_lines
                if not np.any(line.stencil_warp_amplitude > 0.0)
            ]
            radius = (
                max(material_radii)
                if material_radii else self.feature_line_config.max_origin_radius
            )
            # Supplemental S1.4 explicitly permits stencil-aware mixture
            # densities. Split direct object samples between its smallest and
            # largest active dictionaries so a 1 px internal-line stencil is
            # not starved by a much wider silhouette stencil. For an
            # unclassified reflector, split between the global smallest and
            # largest supports: its next vertex is unknown, but reflected
            # fine-line dictionaries must still receive useful samples.
            inner_radius = (
                min(material_radii)
                if material_radii else self.feature_line_config.min_radius
            )
            anchor_radius = (
                max(anchor_radii)
                if anchor_radii else inner_radius
            )
            inner_count = count // 2
            golden_ratio = 0.6180339887498949
            base_origin = np.asarray(ray.o, dtype=np.float64)
            base_direction = normalize(np.asarray(ray.d, dtype=np.float64))
            has_differentials = bool(ray.has_differentials)
            if has_differentials:
                origin_x = np.asarray(ray.o_x - ray.o, dtype=np.float64)
                origin_y = np.asarray(ray.o_y - ray.o, dtype=np.float64)
                direction_x = np.asarray(ray.d_x - ray.d, dtype=np.float64)
                direction_y = np.asarray(ray.d_y - ray.d, dtype=np.float64)
            else:
                first, second = view_oriented_frame(base_direction, [0.0, 1.0, 0.0])
            samples = []
            for index in range(count):
                random = np.asarray(sampler.next_2d(), dtype=np.float64)
                use_inner = inner_radius < radius and index < inner_count
                group_index = index if index < inner_count else index - inner_count
                group_count = inner_count if index < inner_count else count - inner_count
                if self.feature_line_config.sampling_stencil == "disk":
                    radial = (group_index + float(random[0])) / group_count
                    angular = np.mod(
                        group_index * golden_ratio
                        + float(random[1]) / group_count,
                        1.0,
                    )
                    angle = 2.0 * np.pi * angular
                    offset = np.sqrt(radial) * np.array(
                        [np.cos(angle), np.sin(angle)], dtype=np.float64
                    )
                else:
                    grid = int(np.ceil(np.sqrt(group_count)))
                    unit = np.array(
                        [group_index % grid, group_index // grid],
                        dtype=np.float64,
                    )
                    unit = (unit + random) / grid
                    offset = 2.0 * unit - 1.0
                group_center = (
                    np.zeros(2, dtype=np.float64)
                    if index < inner_count and anchor_radii
                    else sampling_center
                )
                offset = group_center + offset * (
                    (
                        anchor_radius
                        if warped is not None and use_inner
                        else inner_radius
                    )
                    if use_inner else radius
                )
                auxiliary = mi.Ray3f(ray)
                if has_differentials:
                    auxiliary.o = mi.Point3f(
                        base_origin + offset[0] * origin_x + offset[1] * origin_y
                    )
                    auxiliary.d = mi.Vector3f(
                        normalize(
                            base_direction
                            + offset[0] * direction_x
                            + offset[1] * direction_y
                        )
                    )
                else:
                    auxiliary.d = mi.Vector3f(
                        normalize(
                            base_direction
                            + 1e-3 * (offset[0] * first + offset[1] * second)
                        )
                    )
                samples.append(
                    ScalarAuxiliarySample(
                        offset=offset,
                        ray=auxiliary,
                        interaction=scene.ray_intersect(auxiliary),
                    )
                )
            return ScalarAuxiliaryFrame(samples)

        @staticmethod
        def _inside_stencil(
            line: FeatureLineType,
            offset: np.ndarray,
            position: np.ndarray,
        ) -> bool:
            offset = offset - line.stencil_center(position)
            if line.stencil == "square":
                return float(np.max(np.abs(offset))) <= line.stencil_radius
            return float(np.dot(offset, offset)) <= line.stencil_radius**2

        def _line_pair_trigger(
            self,
            line: FeatureLineType,
            first: ScalarAuxiliarySample,
            second: ScalarAuxiliarySample,
            position: np.ndarray,
        ) -> bool:
            if not self._line_pair_available(line, first, second, position):
                return False
            distance = float(np.linalg.norm(second.offset - first.offset))
            first_hit = bool(first.interaction.is_valid())
            second_hit = bool(second.interaction.is_valid())
            if first_hit != second_hit:
                return bool(line.include_silhouette)
            if not first_hit:
                return False
            if line.measurement == "silhouette":
                return False
            if line.measurement == "depth":
                first_depth = float(first.interaction.t)
                second_depth = float(second.interaction.t)
                difference = abs(second_depth - first_depth)
                if line.relative_depth:
                    difference /= max(
                        min(abs(first_depth), abs(second_depth)), 1e-8
                    )
                slope = difference / distance
            elif line.measurement == "normal":
                slope = normal_finite_difference_slope(
                    self._rgb(first.interaction.n),
                    self._rgb(second.interaction.n),
                    distance,
                    line.normal_orientation_invariant,
                )
                if line.normal_shape_boundary_fallback:
                    slope_triggered = (
                        first.interaction.shape.id()
                        != second.interaction.shape.id()
                    )
                    if slope_triggered:
                        return True
            elif line.measurement == "curvature":
                normal_dot = float(
                    np.clip(
                        np.dot(
                            self._rgb(first.interaction.n),
                            self._rgb(second.interaction.n),
                        ),
                        -1.0,
                        1.0,
                    )
                )
                slope = float(np.arccos(normal_dot) / distance)
            elif line.measurement == "position":
                slope = float(
                    np.linalg.norm(
                        self._rgb(second.interaction.p)
                        - self._rgb(first.interaction.p)
                    ) / distance
                )
            elif line.measurement == "shape_id":
                return first.interaction.shape.id() != second.interaction.shape.id()
            else:
                return (
                    first.interaction.bsdf(first.ray).id()
                    != second.interaction.bsdf(second.ray).id()
                )
            return slope >= line.threshold

        def _line_pair_available(
            self,
            line: FeatureLineType,
            first: ScalarAuxiliarySample,
            second: ScalarAuxiliarySample,
            position: np.ndarray,
        ) -> bool:
            if not first.prefix_valid or not second.prefix_valid:
                return False
            if not self._inside_stencil(line, first.offset, position):
                return False
            if not self._inside_stencil(line, second.offset, position):
                return False
            return float(np.linalg.norm(second.offset - first.offset)) > 1e-8

        def _detect_feature_line(
            self,
            base_interaction,
            base_ray,
            auxiliary_frame: ScalarAuxiliaryFrame,
            sampler,
            depth: int,
        ) -> tuple[np.ndarray | None, list[bool]]:
            base = ScalarAuxiliarySample(
                offset=np.zeros(2),
                ray=base_ray,
                interaction=base_interaction,
            )
            samples = [base, *auxiliary_frame.samples]
            base_position = self._rgb(base_interaction.p)
            line_color = None
            for line in self.feature_line_config.types:
                if not line.active_at(depth):
                    continue
                base_bsdf = base_interaction.bsdf(base_ray)
                material_id = base_bsdf.id() or base_interaction.shape.id()
                shape_id = base_interaction.shape.id()
                if not line.applies_to_material(material_id) or not line.applies_to_shape(shape_id):
                    continue
                valid_pairs = [
                    (first, second)
                    for first_index, first in enumerate(samples)
                    for second in samples[first_index + 1:]
                    if self._line_pair_available(
                        line, first, second, base_position
                    )
                ]
                if not valid_pairs:
                    continue
                for _ in range(line.comparisons):
                    pair_index = min(
                        int(float(sampler.next_1d()) * len(valid_pairs)),
                        len(valid_pairs) - 1,
                    )
                    if self._line_pair_trigger(
                        line, *valid_pairs[pair_index], base_position
                    ):
                        hatch_weight = line.line_hatch_weight(base_position)
                        if hatch_weight > 0.0:
                            line_color = (
                                line.color_at(depth).copy() * hatch_weight
                            )
                            break
                if line_color is not None:
                    break

            # A scalar DFS path can honor the paper's early exit literally:
            # a detected line terminates this sample, and the last configured
            # line depth has no child frame to prune.
            if (
                line_color is not None
                or not self.feature_line_config.can_apply_from(depth + 1)
            ):
                return line_color, []

            normal_limit = np.cos(
                np.radians(self.feature_line_config.max_normal_angle_degrees)
            )
            continuation = []
            for auxiliary in auxiliary_frame.samples:
                interaction = auxiliary.interaction
                valid = (
                    auxiliary.prefix_valid
                    and bool(interaction.is_valid())
                    and base_interaction.shape.id() == interaction.shape.id()
                    # Geometric normals are used for the feature-line
                    # continuation/pruning test. Shading normals are kept
                    # only for BSDF/parallel-transport evaluation.
                    and float(
                        np.dot(
                            self._rgb(base_interaction.n),
                            self._rgb(interaction.n),
                        )
                    ) >= normal_limit
                )
                if valid:
                    for line in self.feature_line_config.types:
                        base_bsdf = base_interaction.bsdf(base_ray)
                        material_id = base_bsdf.id() or base_interaction.shape.id()
                        shape_id = base_interaction.shape.id()
                        if (
                            line.active_at(depth)
                            and line.applies_to_material(material_id)
                            and line.applies_to_shape(shape_id)
                            and self._line_pair_trigger(
                            line, base, auxiliary, base_position
                            )
                            and line.line_hatch_weight(base_position) > 0.0
                        ):
                            valid = False
                            break
                continuation.append(valid)
            return line_color, continuation

        def _extend_auxiliary_paths(
            self,
            scene,
            frame: ScalarAuxiliaryFrame,
            continuation: list[bool],
            base_interaction,
            base_ray,
            base_outgoing,
            bsdf_sample,
        ) -> ScalarAuxiliaryFrame:
            normal = normalize(self._rgb(base_interaction.sh_frame.n))
            view = normalize(-self._rgb(base_ray.d))
            outgoing = normalize(self._rgb(base_outgoing))
            sampled_type = int(bsdf_sample.sampled_type)
            transmission = bool(
                sampled_type & int(mi.BSDFFlags.Transmission)
            )
            conditioned = bool(
                sampled_type
                & int(mi.BSDFFlags.Glossy | mi.BSDFFlags.Delta)
            )
            eta = max(float(bsdf_sample.eta), 1e-8)
            half_vector = normalize(
                view + eta * outgoing if transmission else view + outgoing
            )
            if float(np.dot(half_vector, normal)) < 0.0:
                half_vector = -half_vector

            children = []
            for auxiliary, valid in zip(frame.samples, continuation):
                if not valid:
                    children.append(
                        ScalarAuxiliarySample(
                            auxiliary.offset,
                            auxiliary.ray,
                            auxiliary.interaction,
                            False,
                        )
                    )
                    continue
                auxiliary_normal = normalize(
                    self._rgb(auxiliary.interaction.sh_frame.n)
                )
                auxiliary_view = normalize(-self._rgb(auxiliary.ray.d))
                if conditioned:
                    transported_half = parallel_transport_half_vector(
                        half_vector,
                        normal,
                        auxiliary_normal,
                        view,
                        auxiliary_view,
                    )
                    if transmission:
                        cosine_incident = float(
                            np.clip(
                                np.dot(auxiliary_view, transported_half),
                                0.0,
                                1.0,
                            )
                        )
                        cosine_transmitted = np.sqrt(
                            max(
                                0.0,
                                1.0
                                - (1.0 - cosine_incident**2) / eta**2,
                            )
                        )
                        auxiliary_outgoing = (
                            -auxiliary_view / eta
                            + (
                                cosine_incident / eta - cosine_transmitted
                            )
                            * transported_half
                        )
                    else:
                        auxiliary_outgoing = (
                            2.0
                            * np.dot(auxiliary_view, transported_half)
                            * transported_half
                            - auxiliary_view
                        )
                else:
                    auxiliary_outgoing = minimal_rotation(
                        outgoing, normal, auxiliary_normal
                    )
                auxiliary_outgoing = normalize(auxiliary_outgoing)
                child_ray = auxiliary.interaction.spawn_ray(
                    mi.Vector3f(auxiliary_outgoing)
                )
                children.append(
                    ScalarAuxiliarySample(
                        offset=auxiliary.offset,
                        ray=child_ray,
                        interaction=scene.ray_intersect(child_ray),
                    )
                )
            return ScalarAuxiliaryFrame(children)

        def _radiance(
            self, scene, sampler, ray, depth, path_state, trace_stats,
            auxiliary_frame=None, tone_frame=None, tone_analytic=False,
        ):
            trace_stats.tree_nodes += 1
            si = scene.ray_intersect(ray)
            if not bool(si.is_valid()): # 没有相交物体
                return self._emission(scene, si)

            use_feature_lines = self.feature_line_config.can_apply_from(depth)
            if use_feature_lines:
                line_color, line_continuation = self._detect_feature_line(
                    si, ray, auxiliary_frame, sampler, depth
                )
                if line_color is not None:
                    return line_color
            else:
                line_continuation = []

            # 存在碰撞物体
            bsdf = si.bsdf(ray) 
            material_id = bsdf.id() or si.shape.id()
            shape_id = si.shape.id()
            next_state, occurrence = path_state.enter(material_id)
            tone_result = (
                self.tone_mapper.query(
                    tone_frame, si, ray, depth, bool(tone_analytic)
                )
                if self.tone_mapper is not None
                and self.config.tone_mapping.active_at(depth)
                else None
            )
            context = StyleContext(
                depth=depth,
                position=self._rgb(si.p),
                normal=self._rgb(si.sh_frame.n),
                uv=np.asarray(si.uv, dtype=np.float64).reshape(-1)[:2],
                material_id=material_id,
                shape_id=shape_id,
                occurrence=occurrence,
                tone_coordinate=(
                    tone_result.coordinate
                    if tone_result is not None else np.zeros(2, dtype=np.float64)
                ),
                tone_valid=(tone_result.valid if tone_result is not None else False),
                tone_confidence=(
                    tone_result.confidence if tone_result is not None else 0.0
                ),
                tone_inversion_method=(
                    tone_result.method if tone_result is not None else "disabled"
                ),
            )
            estimator = self.config.resolve(material_id, shape_id, context) # 构建估计器
            emission = self._emission(scene, si) 
            rng = SamplerRandom(sampler)
            local_stats = EstimateStats()
            tone_child_frame = None

            # 单次光线采样过程
            def sample_integrand():
                nonlocal tone_child_frame
                if depth + 1 >= self.max_depth:
                    return emission.copy()
                if depth >= self.rr_depth and rng.random() >= self.rr_probability:
                    return emission.copy()
                ctx = mi.BSDFContext()
                has_smooth = bool(int(bsdf.flags()) & int(mi.BSDFFlags.Smooth))
                use_emitter_mixture = has_smooth and len(scene.emitters()) > 0
                emitter_probability = 0.5 if use_emitter_mixture else 0.0
                choose_emitter = use_emitter_mixture and rng.random() < emitter_probability

                # 采样光源
                if choose_emitter:
                    ds, emitter_weight = scene.sample_emitter_direction(
                        si, sampler.next_2d(), True
                    )
                    emitter_weight_rgb = self._rgb(emitter_weight)
                    if float(ds.pdf) <= 0.0 or np.max(np.abs(emitter_weight_rgb)) == 0.0:
                        return emission.copy()
                    direction = ds.d
                    f_cos, bsdf_pdf = bsdf.eval_pdf(ctx, si, si.to_local(direction))
                    mixture_pdf = (
                        emitter_probability * float(ds.pdf)
                        + (
                            0.0 if bool(ds.delta)
                            else (1.0 - emitter_probability) * float(bsdf_pdf)
                        )
                    )
                    if mixture_pdf <= 0.0:
                        return emission.copy()
                    # A point/directional emitter has no surface that a ray
                    # can intersect. ``emitter_weight`` already contains its
                    # radiance divided by the emitter PDF, so finish this
                    # branch directly instead of recursively tracing it.
                    transport = (
                        self._rgb(f_cos)
                        * emitter_weight_rgb
                        * float(ds.pdf)
                        / mixture_pdf
                    )
                    if depth >= self.rr_depth:
                        transport /= self.rr_probability
                    return emission + transport
                # 采样BSDF反射方向
                else:
                    bs, weight = bsdf.sample(
                        ctx, si, sampler.next_1d(), sampler.next_2d()
                    )
                    bsdf_pdf = float(bs.pdf)
                    weight_rgb = self._rgb(weight)
                    if bsdf_pdf <= 0.0 or not np.all(np.isfinite(weight_rgb)) \
                            or np.max(np.abs(weight_rgb)) == 0.0:
                        return emission.copy()
                    direction = si.to_world(bs.wo)
                    emitter_pdf = 0.0
                    if use_emitter_mixture:
                        next_si = scene.ray_intersect(si.spawn_ray(direction))
                        ds_hit = mi.DirectionSample3f(scene, next_si, si)
                        if ds_hit.emitter is not None:
                            emitter_pdf = float(scene.pdf_emitter_direction(si, ds_hit))
                    mixture_pdf = (
                        (1.0 - emitter_probability) * bsdf_pdf
                        + emitter_probability * emitter_pdf
                    )
                    numerator = weight_rgb * bsdf_pdf

                if mixture_pdf <= 0.0:
                    return emission.copy()

                # 递归追踪与通量计算
                next_ray = si.spawn_ray(direction)
                next_auxiliary = (
                    self._extend_auxiliary_paths(
                        scene,
                        auxiliary_frame,
                        line_continuation,
                        si,
                        ray,
                        direction,
                        bs,
                    )
                    if self.feature_line_config.can_apply_from(depth + 1)
                    else None
                )
                if (
                    self.tone_mapper is not None
                    and self.config.tone_mapping.can_extend_from(depth)
                    and tone_child_frame is None
                ):
                    # The canonical anchor is material-independent and does not
                    # depend on this estimator draw. Build it once at the DFS
                    # vertex and reuse it for all inner samples.
                    tone_child_frame = self.tone_mapper.extend(
                        scene, tone_frame, si, True
                    )
                incoming = self._radiance(
                    scene, sampler, next_ray, depth + 1, next_state,
                    trace_stats, next_auxiliary, tone_child_frame,
                    bool(tone_analytic)
                    and bool(
                        int(bs.sampled_type)
                        & int(mi.BSDFFlags.DeltaReflection)
                    ),
                )
                transport = numerator * incoming / mixture_pdf
                if depth >= self.rr_depth:
                    transport /= self.rr_probability
                return emission + transport

            result = estimator.estimate(sample_integrand, rng, context, local_stats) # 估计器求值与统计数据回写
            trace_stats.style_evaluations += local_stats.style_evaluations
            trace_stats.inner_variance += local_stats.inner_variance
            trace_stats.estimated_bias += local_stats.estimated_bias
            return result

        def sample(self, scene, sampler, ray, medium=None, active=True):
            if not bool(active):
                return mi.Color3f(0.0), False, [0.0] * 4
            stats = TraceStats()
            # 第一条光线求交校验
            primary = scene.ray_intersect(ray)
            valid = bool(primary.is_valid())
            primary_material_id = None
            primary_shape_id = None
            primary_position = None
            if valid:
                primary_bsdf = primary.bsdf(ray)
                primary_material_id = primary_bsdf.id() or primary.shape.id()
                primary_shape_id = primary.shape.id()
                primary_position = self._rgb(primary.p)
            auxiliary_frame = (
                self._spawn_auxiliary_paths(
                    scene,
                    sampler,
                    ray,
                    primary_material_id,
                    primary_shape_id,
                    primary_position,
                )
                if self.feature_line_config.enabled else None
            )
            tone_frame = (
                self.tone_mapper.spawn_primary(scene, ray, valid)
                if self.tone_mapper is not None else None
            )
            # 调用核心递归 Path Tracer & 估计器求值
            radiance = self._radiance(
                scene, sampler, ray, 0, PathState(), stats,
                auxiliary_frame, tone_frame, True,
            )
            aovs = [
                float(stats.tree_nodes),
                float(stats.style_evaluations),
                float(stats.inner_variance),
                float(stats.estimated_bias),
            ]
            return self._spectrum(radiance), valid, aovs

        def to_string(self):
            return (
                f"SREIntegrator[max_depth={self.max_depth}, rr_depth={self.rr_depth}, "
                f"rr_probability={self.rr_probability}]"
            )

    mi.register_integrator("sre", lambda props: SREIntegrator(props))
    _REGISTERED = True

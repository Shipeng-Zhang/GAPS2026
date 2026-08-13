"""Style functions g_theta used by the SRE estimators.

The functions in this module are renderer-independent NumPy callables. This is
intentional: their expectation estimators can be statistically tested without
mixing those tests with ray tracing or Mitsuba scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


RGB = np.ndarray
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64) # 相对颜色权重系数


@dataclass(frozen=True) # 数据类不可变，属性不可被修改
class StyleContext:
    depth: int = 0 # 光线弹射深度
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64)) # 交点空间三维坐标
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)) # 表面法线向量
    uv: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64)) # 纹理坐标
    material_id: str = "" # 材质标识符
    shape_id: str = "" # 物体标识符
    occurrence: int = 1 # 采样次数
    # Canonical image-space coordinate reconstructed by Section 5's mirror
    # anchors. ``Any`` keeps this renderer-independent dataclass compatible
    # with both NumPy arrays and Dr.Jit Point2f wavefronts.
    tone_coordinate: Any = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    tone_valid: Any = False
    tone_confidence: Any = 0.0
    tone_inversion_method: Any = "disabled"


class StyleFunction(Protocol):
    def __call__(self, value, context): ...

# 转换为(3,)格式RGB数据
def as_rgb(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.repeat(result, 3)
    if result.shape != (3,):
        raise ValueError(f"Expected an RGB vector, got shape {result.shape}")
    return result

# 计算标量相对亮度
def luminance(value):
    return float(np.dot(LUMA, as_rgb(value)))

# 恒等映射,相当于PRB
class Identity:
    def __call__(self, value, context):
        return as_rgb(value)

# Gamma 校正风格函数: V_out = V_in^{1/2.2}
class Gamma:
    def __init__(self, gamma = 2.2):
        self.exponent = 1.0 / float(gamma)

    def __call__(self, value, context):
        return np.power(np.maximum(as_rgb(value), 0.0), self.exponent)

# 色彩饱和度调节风格函数: C_out = gray + s(C_in - gray)
class Saturation:
    def __init__(self, amount = 1.5):
        self.amount = float(amount)

    def __call__(self, value, context):
        value = as_rgb(value)
        gray = luminance(value)
        return gray + self.amount * (value - gray)

# 渐变色彩映射风格函数
class ColorMap:
    def __init__(self, colors, positions = None, input_range= (0.0, 1.0)):
        # colors:渐变色列表[[0,0,0],[0,0,1]]
        # positions: 渐变节点位置,长度与colors长度一致
        # input_range: 输入的映射亮度区间 
        if len(colors) < 2:
            raise ValueError("A color map needs at least two colors")
        self.colors = np.asarray(colors, dtype=np.float64) # 转为(n,3)列表
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must have shape (n, 3)")
        # 不存在渐变位置则线性插值
        self.positions = np.asarray(
            positions if positions is not None else np.linspace(0.0, 1.0, len(colors)),
            dtype=np.float64,
        ) 
        if self.positions.shape != (len(colors),) or np.any(np.diff(self.positions) <= 0):
            raise ValueError("positions must be strictly increasing and match colors")
        self.low, self.high = map(float, input_range)
        if self.high <= self.low:
            raise ValueError("input_range must have positive width")

    def __call__(self, value, context):
        # 计算像素亮度并映射
        t = np.clip((luminance(value) - self.low) / (self.high - self.low), 0.0, 1.0)
        # 对R G B 三个通道进行一维线性插值
        return np.array([
            np.interp(t, self.positions, self.colors[:, channel]) for channel in range(3)
        ])


class ColorMap_Nonlinear:
    """Color gradient with a nonlinear smoothstep transfer per segment."""

    def __init__(
        self,
        colors,
        positions = None,
        input_range = (0.0, 1.0),
    ):
        if len(colors) < 2:
            raise ValueError(
                "A nonlinear color map needs at least two colors"
            )
        self.colors = np.asarray(colors, dtype=np.float64)
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must have shape (n, 3)")
        self.positions = np.asarray(
            positions
            if positions is not None
            else np.linspace(0.0, 1.0, len(colors)),
            dtype=np.float64,
        )
        if (
            self.positions.shape != (len(colors),)
            or np.any(np.diff(self.positions) <= 0)
        ):
            raise ValueError(
                "positions must be strictly increasing and match colors"
            )
        self.low, self.high = map(float, input_range)
        if self.high <= self.low:
            raise ValueError("input_range must have positive width")

    def __call__(self, value, context):
        del context
        t = np.clip(
            (luminance(value) - self.low) / (self.high - self.low),
            0.0,
            1.0,
        )
        result = self.colors[0].copy()
        for index in range(len(self.positions) - 1):
            low = self.positions[index]
            high = self.positions[index + 1]
            local = np.clip((t - low) / (high - low), 0.0, 1.0)
            weight = local * local * (3.0 - 2.0 * local)
            segment = (
                (1.0 - weight) * self.colors[index]
                + weight * self.colors[index + 1]
            )
            if t >= low:
                result = segment
        return result

# 卡通/赛璐珞着色风格函数
class Cel:
    """Discrete brightness bands that preserve the estimated RGB chroma.

    ``brightness_mode="mean"`` implements the GI cel style from supplemental
    Section S5.2: compute ``u = (R + G + B) / 3``, select a target brightness
    ``u'``, then return ``RGB * u' / u``.  The older luminance and palette
    modes remain available for existing configurations.
    """

    def __init__(
        self,
        levels: int = 4,
        max_value: float = 1.0,
        palette: list[list[float]] | None = None,
        preserve_chroma: bool = True,
        thresholds: list[float] | tuple[float, ...] | None = None,
        band_values: list[float] | tuple[float, ...] | None = None,
        chroma_strength: float = 0.0,
        chroma_weights: list[float] | tuple[float, ...] | None = None,
        brightness_mode: str = "luminance",
    ) -> None:
        if levels < 2:
            raise ValueError("Cel shading needs at least two levels")
        self.levels = int(levels)
        self.max_value = float(max_value)
        self.palette = None if palette is None else np.asarray(palette, dtype=np.float64)
        self.preserve_chroma = bool(preserve_chroma)
        self.thresholds = np.asarray(
            thresholds
            if thresholds is not None
            else np.linspace(0.0, 1.0, self.levels + 1)[1:-1],
            dtype=np.float64,
        )
        self.band_values = np.asarray(
            band_values
            if band_values is not None
            else (np.arange(self.levels, dtype=np.float64) + 0.5) / self.levels,
            dtype=np.float64,
        )
        self.chroma_strength = float(chroma_strength)
        self.brightness_mode = str(brightness_mode).lower()
        if self.brightness_mode not in {"luminance", "mean"}:
            raise ValueError("brightness_mode must be 'luminance' or 'mean'")
        self.brightness_weights = (
            LUMA.copy()
            if self.brightness_mode == "luminance"
            else np.full(3, 1.0 / 3.0, dtype=np.float64)
        )
        self.chroma_weights = np.asarray(
            chroma_weights
            if chroma_weights is not None
            else np.ones(self.levels, dtype=np.float64),
            dtype=np.float64,
        )
        if self.max_value <= 0.0 or not np.isfinite(self.max_value):
            raise ValueError("Cel max_value must be finite and positive")
        if self.palette is not None and self.palette.shape != (self.levels, 3):
            raise ValueError("palette must contain one RGB color per level")
        if self.palette is not None and np.any(~np.isfinite(self.palette)):
            raise ValueError("palette colors must be finite")
        if (
            self.thresholds.shape != (self.levels - 1,)
            or np.any(~np.isfinite(self.thresholds))
            or np.any(np.diff(self.thresholds) <= 0.0)
            or np.any((self.thresholds <= 0.0) | (self.thresholds >= 1.0))
        ):
            raise ValueError(
                "thresholds must contain levels-1 increasing values in (0, 1)"
            )
        if (
            self.band_values.shape != (self.levels,)
            or np.any(~np.isfinite(self.band_values))
            or np.any(self.band_values < 0.0)
        ):
            raise ValueError("band_values must contain one non-negative value per level")
        if not 0.0 <= self.chroma_strength <= 1.0:
            raise ValueError("chroma_strength must lie in [0, 1]")
        if (
            self.chroma_weights.shape != (self.levels,)
            or np.any(~np.isfinite(self.chroma_weights))
            or np.any((self.chroma_weights < 0.0) | (self.chroma_weights > 1.0))
        ):
            raise ValueError(
                "chroma_weights must contain one value in [0, 1] per level"
            )

    def __call__(self, value: RGB, context: StyleContext) -> RGB:
        del context
        value = as_rgb(value)
        current = float(np.dot(self.brightness_weights, value))
        normalized = np.clip(current / self.max_value, 0.0, 1.0)
        index = int(np.searchsorted(self.thresholds, normalized, side="right"))
        if self.palette is not None:
            base = self.palette[index].copy()
            if self.chroma_strength == 0.0:
                return base
            target = float(np.dot(self.brightness_weights, base))
            chroma = value - current
            result = base + (
                self.chroma_strength
                * self.chroma_weights[index]
                * chroma
                * (target / max(current, 1e-8))
            )
            return np.maximum(result, 0.0)
        target = self.band_values[index] * self.max_value
        if not self.preserve_chroma:
            return np.repeat(target, 3)
        return value * (target / max(current, 1e-8))

# 交叉阴影风格函数
class CrossHatch:
    """Nested object-space hatching planes controlled by expected brightness.

    Each direction is the normal of one family of parallel slice planes. The
    visible strokes are their intersections with the surface, so they remain
    attached to the geometry instead of swimming in screen space. Successively
    darker tones activate additional families, following the construction used
    for the Deussen et al. result in Fig. 12.
    """

    DEFAULT_DIRECTIONS = (
        (-0.848907, 0.286233, -0.444327),
        (-0.321988, 0.817848, -0.476915),
        (0.608575, 0.785522, -0.112213),
        (0.931034, 0.286233, 0.226375),
    )

    def __init__(
        self,
        scale: float = 34.0,
        width: float = 0.06,
        max_value: float = 1.0,
        ink: tuple[float, float, float] = (0.035, 0.032, 0.03),
        paper: tuple[float, float, float] = (0.94, 0.935, 0.91),
        directions: tuple[tuple[float, float, float], ...] | None = None,
        activation_thresholds: tuple[float, ...] = (0.12, 0.44, 0.64, 0.82),
        phase_offsets: tuple[float, ...] = (0.13, 0.0, 0.37, 0.61),
        scale_factors: tuple[float, ...] = (1.04, 1.0, 0.97, 0.94),
        family_widths: tuple[float, ...] = (0.85, 1.0, 0.9, 0.8),
        width_growth: float = 0.3,
        edge_softness: float = 0.012,
        darkness_gamma: float = 1.0,
    ) -> None:
        self.scale = float(scale)
        self.width = float(width)
        self.max_value = float(max_value)
        self.ink = as_rgb(ink)
        self.paper = as_rgb(paper)
        raw_directions = np.asarray(
            directions if directions is not None else self.DEFAULT_DIRECTIONS,
            dtype=np.float64,
        )
        if raw_directions.ndim != 2 or raw_directions.shape[1] != 3:
            raise ValueError("cross-hatch directions must have shape (n, 3)")
        lengths = np.linalg.norm(raw_directions, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-8):
            raise ValueError("cross-hatch directions must be finite and non-zero")
        self.directions = raw_directions / lengths[:, None]
        family_count = len(self.directions)
        self.activation_thresholds = np.asarray(
            activation_thresholds, dtype=np.float64
        )
        self.phase_offsets = np.mod(
            np.asarray(phase_offsets, dtype=np.float64), 1.0
        )
        self.scale_factors = np.asarray(scale_factors, dtype=np.float64)
        self.family_widths = np.asarray(family_widths, dtype=np.float64)
        for name, values in (
            ("activation_thresholds", self.activation_thresholds),
            ("phase_offsets", self.phase_offsets),
            ("scale_factors", self.scale_factors),
            ("family_widths", self.family_widths),
        ):
            if values.shape != (family_count,) or np.any(~np.isfinite(values)):
                raise ValueError(
                    f"cross-hatch {name} must contain one finite value per direction"
                )
        self.width_growth = float(width_growth)
        self.edge_softness = float(edge_softness)
        self.darkness_gamma = float(darkness_gamma)
        if self.scale <= 0.0 or self.max_value <= 0.0:
            raise ValueError("cross-hatch scale and max_value must be positive")
        if not 0.0 < self.width < 0.5:
            raise ValueError("cross-hatch width must lie in (0, 0.5)")
        if np.any(np.diff(self.activation_thresholds) <= 0.0):
            raise ValueError("cross-hatch activation_thresholds must increase")
        if np.any((self.activation_thresholds < 0.0) | (self.activation_thresholds > 1.0)):
            raise ValueError("cross-hatch activation_thresholds must lie in [0, 1]")
        if np.any(self.scale_factors <= 0.0) or np.any(self.family_widths <= 0.0):
            raise ValueError("cross-hatch scale_factors and family_widths must be positive")
        if self.width_growth < 0.0 or self.edge_softness < 0.0:
            raise ValueError("cross-hatch width growth and edge softness cannot be negative")
        if self.darkness_gamma <= 0.0:
            raise ValueError("cross-hatch darkness_gamma must be positive")

    def __call__(self, value: RGB, context: StyleContext) -> RGB:
        normalized = np.clip(luminance(value) / self.max_value, 0.0, 1.0)
        darkness = (1.0 - normalized) ** self.darkness_gamma
        point = as_rgb(context.position)
        coverage = 0.0
        for index, direction in enumerate(self.directions):
            if darkness < self.activation_thresholds[index]:
                continue
            phase = np.mod(
                self.scale * self.scale_factors[index] * np.dot(direction, point)
                + self.phase_offsets[index],
                1.0,
            )
            distance = min(phase, 1.0 - phase)
            half_width = (
                self.width
                * self.family_widths[index]
                * (1.0 + self.width_growth * darkness)
            )
            if self.edge_softness > 0.0:
                weight = np.clip(
                    (half_width + self.edge_softness - distance)
                    / (2.0 * self.edge_softness),
                    0.0,
                    1.0,
                )
                weight = weight * weight * (3.0 - 2.0 * weight)
            else:
                weight = float(distance < half_width)
            coverage = max(coverage, float(weight))
        return (1.0 - coverage) * self.paper + coverage * self.ink

# 半色调风格函数
class Halftone:
    """Fig. 12-style world-space lattice of tone-controlled spheres.

    The surface pattern is the intersection of the shaded geometry with a
    regular 3D grid of spheres. This matches the paper's Q-map-like
    construction and remains stable on meshes such as the Stanford Dragon that
    do not provide useful UV coordinates.
    """

    def __init__(
        self,
        scale: float = 42.0,
        max_value: float = 1.0,
        ink: tuple[float, float, float] = (0.025, 0.027, 0.03),
        paper: tuple[float, float, float] = (0.96, 0.955, 0.93),
        min_radius: float = 0.055,
        max_radius: float = 0.78,
        radius_gamma: float = 0.82,
        dot_threshold: float = 0.03,
        edge_softness: float = 0.02,
        min_ink_strength: float = 0.9,
        phase: tuple[float, float, float] = (0.17, 0.37, 0.11),
        orientation: tuple[tuple[float, float, float], ...] = (
            (0.895, -0.259, 0.362),
            (0.240, 0.966, 0.097),
            (-0.375, 0.0, 0.927),
        ),
    ) -> None:
        self.scale = float(scale)
        self.max_value = float(max_value)
        self.ink = as_rgb(ink)
        self.paper = as_rgb(paper)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.radius_gamma = float(radius_gamma)
        self.dot_threshold = float(dot_threshold)
        self.edge_softness = float(edge_softness)
        self.min_ink_strength = float(min_ink_strength)
        self.phase = as_rgb(phase)
        self.orientation = np.asarray(orientation, dtype=np.float64)
        if self.scale <= 0.0 or self.max_value <= 0.0:
            raise ValueError("halftone scale and max_value must be positive")
        if not 0.0 <= self.min_radius < self.max_radius <= np.sqrt(0.75):
            raise ValueError(
                "halftone radii must satisfy 0 <= min < max <= sqrt(3)/2"
            )
        if self.radius_gamma <= 0.0:
            raise ValueError("halftone radius_gamma must be positive")
        if not 0.0 <= self.dot_threshold < 1.0:
            raise ValueError("halftone dot_threshold must lie in [0, 1)")
        if self.edge_softness < 0.0:
            raise ValueError("halftone edge_softness cannot be negative")
        if not 0.0 <= self.min_ink_strength <= 1.0:
            raise ValueError("halftone min_ink_strength must lie in [0, 1]")
        if self.orientation.shape != (3, 3) or np.any(~np.isfinite(self.orientation)):
            raise ValueError("halftone orientation must be a finite 3x3 matrix")
        if not np.allclose(
            self.orientation @ self.orientation.T,
            np.eye(3),
            atol=2e-3,
            rtol=0.0,
        ):
            raise ValueError("halftone orientation must be orthonormal")

    def __call__(self, value: RGB, context: StyleContext) -> RGB:
        normalized = np.clip(luminance(value) / self.max_value, 0.0, 1.0)
        darkness = 1.0 - normalized
        if darkness <= self.dot_threshold:
            return self.paper.copy()
        tone = np.clip(
            (darkness - self.dot_threshold) / (1.0 - self.dot_threshold),
            0.0,
            1.0,
        )
        radius_weight = tone ** self.radius_gamma
        radius = self.min_radius + (
            self.max_radius - self.min_radius
        ) * radius_weight
        lattice = (
            self.scale * self.orientation.dot(as_rgb(context.position))
            + self.phase
        )
        cell = np.mod(lattice, 1.0) - 0.5
        distance = float(np.linalg.norm(cell))
        if self.edge_softness > 0.0:
            coverage = np.clip(
                (radius + self.edge_softness - distance)
                / (2.0 * self.edge_softness),
                0.0,
                1.0,
            )
            coverage = coverage * coverage * (3.0 - 2.0 * coverage)
        else:
            coverage = float(distance < radius)
        ink_strength = self.min_ink_strength + (
            1.0 - self.min_ink_strength
        ) * tone
        coverage *= ink_strength
        return (1.0 - coverage) * self.paper + coverage * self.ink


class ToneHatch:
    """Image-space hatching evaluated through the canonical tone inverse.

    Angles specify the visible stroke direction in degrees. Darker expected
    radiance widens the active line families while the reconstructed film
    coordinate fixes their orientation, scale, and phase through direct,
    reflected, glossy, and multi-bounce paths.
    """

    def __init__(
        self,
        spacing: float = 7.0,
        angles_degrees: tuple[float, ...] = (78.0,),
        activation_thresholds: tuple[float, ...] = (0.0,),
        phase_offsets: tuple[float, ...] = (0.17,),
        family_widths: tuple[float, ...] = (1.0,),
        min_coverage: float = 0.045,
        max_coverage: float = 0.72,
        darkness_gamma: float = 0.85,
        edge_softness: float = 0.65,
        max_value: float = 1.0,
        brightness_thresholds: tuple[float, ...] = (),
        brightness_levels: tuple[float, ...] = (),
        brightness_mode: str = "mean",
        shadow_strength: float = 0.0,
        ink: tuple[float, float, float] = (0.018, 0.019, 0.021),
        paper: tuple[float, float, float] = (0.965, 0.958, 0.925),
        region_center: tuple[float, float] | None = None,
        region_radius: tuple[float, float] | None = None,
        region_feather: float = 0.2,
    ) -> None:
        self.spacing = float(spacing)
        self.angles_degrees = np.asarray(angles_degrees, dtype=np.float64)
        self.activation_thresholds = np.asarray(
            activation_thresholds, dtype=np.float64
        )
        self.phase_offsets = np.asarray(phase_offsets, dtype=np.float64)
        self.family_widths = np.asarray(family_widths, dtype=np.float64)
        self.min_coverage = float(min_coverage)
        self.max_coverage = float(max_coverage)
        self.darkness_gamma = float(darkness_gamma)
        self.edge_softness = float(edge_softness)
        self.max_value = float(max_value)
        self.brightness_thresholds = np.asarray(
            brightness_thresholds, dtype=np.float64
        )
        self.brightness_levels = np.asarray(
            brightness_levels, dtype=np.float64
        )
        self.brightness_mode = str(brightness_mode).lower()
        # Optional soft shadow bed used by glossy floors. It is deliberately
        # gated by the first line threshold, so ordinary bright paper remains
        # white while the low-frequency reflected/shadow contribution can be
        # retained underneath the hatch strokes.
        self.shadow_strength = float(shadow_strength)
        self.ink = as_rgb(ink)
        self.paper = as_rgb(paper)
        self.region_center = (
            None if region_center is None
            else np.asarray(region_center, dtype=np.float64)
        )
        self.region_radius = (
            None if region_radius is None
            else np.asarray(region_radius, dtype=np.float64)
        )
        self.region_feather = float(region_feather)
        family_count = len(self.angles_degrees)
        if family_count < 1:
            raise ValueError("tone hatching needs at least one line family")
        for name, values in (
            ("activation_thresholds", self.activation_thresholds),
            ("phase_offsets", self.phase_offsets),
            ("family_widths", self.family_widths),
        ):
            if values.shape != (family_count,) or np.any(~np.isfinite(values)):
                raise ValueError(
                    f"tone hatch {name} must contain one finite value per angle"
                )
        if self.spacing <= 0.0 or self.max_value <= 0.0:
            raise ValueError("tone hatch spacing and max_value must be positive")
        if not 0.0 <= self.min_coverage < self.max_coverage < 1.0:
            raise ValueError(
                "tone hatch coverage must satisfy 0 <= min < max < 1"
            )
        if self.darkness_gamma <= 0.0 or self.edge_softness < 0.0:
            raise ValueError(
                "tone hatch darkness_gamma must be positive and softness non-negative"
            )
        if self.brightness_levels.size:
            if self.brightness_levels.shape != (
                len(self.brightness_thresholds) + 1,
            ):
                raise ValueError(
                    "tone hatch brightness_levels needs one more value than thresholds"
                )
            if (
                np.any(~np.isfinite(self.brightness_thresholds))
                or np.any(self.brightness_thresholds < 0.0)
                or np.any(np.diff(self.brightness_thresholds) <= 0.0)
            ):
                raise ValueError(
                    "tone hatch brightness thresholds must be finite, non-negative, "
                    "and strictly increasing"
                )
            if np.any(~np.isfinite(self.brightness_levels)) or np.any(
                self.brightness_levels < 0.0
            ):
                raise ValueError(
                    "tone hatch brightness levels must be finite and non-negative"
                )
        if self.brightness_mode not in {"mean", "luminance"}:
            raise ValueError("tone hatch brightness_mode must be 'mean' or 'luminance'")
        if not 0.0 <= self.shadow_strength <= 1.0:
            raise ValueError("tone hatch shadow_strength must lie in [0, 1]")
        if np.any((self.activation_thresholds < 0.0) | (self.activation_thresholds > 1.0)):
            raise ValueError("tone hatch activation thresholds must lie in [0, 1]")
        if np.any(self.family_widths <= 0.0):
            raise ValueError("tone hatch family widths must be positive")
        if (self.region_center is None) != (self.region_radius is None):
            raise ValueError(
                "tone hatch region_center and region_radius must be specified together"
            )
        if self.region_center is not None:
            if self.region_center.shape != (2,) or self.region_radius.shape != (2,):
                raise ValueError("tone hatch region parameters must contain two values")
            if np.any(~np.isfinite(self.region_center)) or np.any(
                ~np.isfinite(self.region_radius)
            ) or np.any(self.region_radius <= 0.0):
                raise ValueError("tone hatch region must be finite with positive radii")
        if not 0.0 <= self.region_feather <= 1.0:
            raise ValueError("tone hatch region_feather must lie in [0, 1]")

    def __call__(self, value: RGB, context: StyleContext) -> RGB:
        coordinate = np.asarray(context.tone_coordinate, dtype=np.float64).reshape(2)
        value = as_rgb(value)
        if self.brightness_levels.size:
            current = (
                float(np.mean(value))
                if self.brightness_mode == "mean"
                else luminance(value)
            )
            index = int(
                np.searchsorted(self.brightness_thresholds, current, side="right")
            )
            target = float(self.brightness_levels[index])
            value = value * (target / max(current, 1e-8))
        darkness = np.clip(1.0 - luminance(value) / self.max_value, 0.0, 1.0)
        darkness = darkness ** self.darkness_gamma
        coverage = 0.0
        region_mask = 1.0
        for index, angle in enumerate(np.radians(self.angles_degrees)):
            if darkness < self.activation_thresholds[index]:
                continue
            # The normal, rather than tangent, measures distance to a stroke.
            normal = np.array([-np.sin(angle), np.cos(angle)])
            phase = np.mod(
                np.dot(normal, coordinate) / self.spacing
                + self.phase_offsets[index],
                1.0,
            )
            distance = min(phase, 1.0 - phase) * self.spacing
            tone = np.clip(
                (darkness - self.activation_thresholds[index])
                / max(1.0 - self.activation_thresholds[index], 1e-8),
                0.0,
                1.0,
            )
            fraction = self.min_coverage + (
                self.max_coverage - self.min_coverage
            ) * tone
            half_width = 0.5 * self.spacing * fraction * self.family_widths[index]
            if self.edge_softness > 0.0:
                weight = np.clip(
                    (half_width + self.edge_softness - distance)
                    / (2.0 * self.edge_softness),
                    0.0,
                    1.0,
                )
                weight = weight * weight * (3.0 - 2.0 * weight)
            else:
                weight = float(distance < half_width)
            coverage = max(coverage, float(weight))
        if self.region_center is not None:
            normalized = (coordinate - self.region_center) / self.region_radius
            radius = float(np.linalg.norm(normalized))
            if self.region_feather > 0.0:
                mask = np.clip(
                    (1.0 + self.region_feather - radius) / self.region_feather,
                    0.0,
                    1.0,
                )
                mask = mask * mask * (3.0 - 2.0 * mask)
            else:
                mask = float(radius <= 1.0)
            region_mask = mask
            coverage *= region_mask
        if self.shadow_strength > 0.0:
            threshold = float(self.activation_thresholds[0])
            shadow_tone = np.clip(
                (darkness - threshold) / max(1.0 - threshold, 1e-8),
                0.0,
                1.0,
            )
            shadow = self.shadow_strength * shadow_tone * region_mask
            base = (1.0 - shadow) * self.paper + shadow * self.ink
        else:
            base = self.paper
        return (1.0 - coverage) * base + coverage * self.ink


class ToneHalftone:
    """A two-dimensional, image-space-consistent halftone dot field."""

    def __init__(
        self,
        spacing: float = 8.0,
        angle_degrees: float = 45.0,
        min_radius: float = 0.35,
        max_radius: float = 3.55,
        radius_gamma: float = 0.78,
        dot_threshold: float = 0.025,
        edge_softness: float = 0.6,
        max_value: float = 1.0,
        phase: tuple[float, float] = (0.19, 0.37),
        ink: tuple[float, float, float] = (0.018, 0.019, 0.021),
        paper: tuple[float, float, float] = (0.965, 0.958, 0.925),
    ) -> None:
        self.spacing = float(spacing)
        self.angle_degrees = float(angle_degrees)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.radius_gamma = float(radius_gamma)
        self.dot_threshold = float(dot_threshold)
        self.edge_softness = float(edge_softness)
        self.max_value = float(max_value)
        self.phase = np.asarray(phase, dtype=np.float64)
        self.ink = as_rgb(ink)
        self.paper = as_rgb(paper)
        if self.spacing <= 0.0 or self.max_value <= 0.0:
            raise ValueError("tone halftone spacing and max_value must be positive")
        if not 0.0 <= self.min_radius < self.max_radius <= 0.5 * self.spacing:
            raise ValueError(
                "tone halftone radii must satisfy 0 <= min < max <= spacing/2"
            )
        if self.radius_gamma <= 0.0 or self.edge_softness < 0.0:
            raise ValueError(
                "tone halftone radius_gamma must be positive and softness non-negative"
            )
        if not 0.0 <= self.dot_threshold < 1.0:
            raise ValueError("tone halftone dot_threshold must lie in [0, 1)")
        if self.phase.shape != (2,) or np.any(~np.isfinite(self.phase)):
            raise ValueError("tone halftone phase must contain two finite values")

    def __call__(self, value: RGB, context: StyleContext) -> RGB:
        coordinate = np.asarray(context.tone_coordinate, dtype=np.float64).reshape(2)
        angle = np.radians(self.angle_degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = np.array([
            cosine * coordinate[0] - sine * coordinate[1],
            sine * coordinate[0] + cosine * coordinate[1],
        ])
        cell = np.mod(rotated / self.spacing + self.phase, 1.0) - 0.5
        distance = float(np.linalg.norm(cell * self.spacing))
        darkness = np.clip(1.0 - luminance(value) / self.max_value, 0.0, 1.0)
        tone = np.clip(
            (darkness - self.dot_threshold) / (1.0 - self.dot_threshold),
            0.0,
            1.0,
        )
        radius = self.min_radius + (
            self.max_radius - self.min_radius
        ) * tone ** self.radius_gamma
        if self.edge_softness > 0.0:
            coverage = np.clip(
                (radius + self.edge_softness - distance)
                / (2.0 * self.edge_softness),
                0.0,
                1.0,
            )
            coverage = coverage * coverage * (3.0 - 2.0 * coverage)
        else:
            coverage = float(distance < radius)
        coverage = float(coverage) if darkness > self.dot_threshold else 0.0
        return (1.0 - coverage) * self.paper + coverage * self.ink

# Gooch着色风格函数
class Gooch:
    def __init__(self, cool=(0.1, 0.25, 0.75), warm=(0.95, 0.75, 0.15), max_value = 1.0):
        # cool: 冷色调
        # warm: 暖色调
        self.cool = as_rgb(cool)
        self.warm = as_rgb(warm)
        self.max_value = float(max_value)

    def __call__(self, value, context):
        # 根据亮度映射到冷暖色调
        t = np.clip(luminance(value) / self.max_value, 0.0, 1.0)
        return (1.0 - t) * self.cool + t * self.warm

# 扎染风格函数
class TieDye:
    """Component-wise cosine waves used by the Fig. 11 tie-dye style.

    Supplemental Section S5.5 defines each channel as
    ``(1-cos(pi*m*(x+s)))/2`` with channel-specific multipliers ``m`` and
    shifts ``s``. Keeping the channels component-separable lets the polynomial
    estimator apply Eq. (17) independently to R, G, and B. The older angular
    ``frequencies``/``phases`` form is accepted for existing configurations.
    """

    def __init__(
        self,
        frequency_multipliers: tuple[float, float, float] = (2.0, 2.15, 2.3),
        input_shifts: tuple[float, float, float] = (1.0, 1.13, 1.29),
        frequencies: tuple[float, float, float] | None = None,
        phases: tuple[float, float, float] | None = None,
        amplitude: float = 0.5,
        offset: float = 0.5,
        inverted: bool | None = None,
    ) -> None:
        self.frequency_multipliers = as_rgb(frequency_multipliers)
        self.input_shifts = as_rgb(input_shifts)
        legacy_parameters = frequencies is not None or phases is not None
        if legacy_parameters:
            self.frequencies = as_rgb(
                frequencies if frequencies is not None else np.pi * self.frequency_multipliers
            )
            self.phases = as_rgb(phases if phases is not None else (0.0, 0.0, 0.0))
        else:
            self.frequencies = np.pi * self.frequency_multipliers
            self.phases = self.frequencies * self.input_shifts
        self.amplitude = float(amplitude)
        self.offset = float(offset)
        self.inverted = bool(not legacy_parameters if inverted is None else inverted)
        self.cosine_sign = -1.0 if self.inverted else 1.0
        if not np.all(np.isfinite(self.frequency_multipliers)):
            raise ValueError("tie-dye frequency_multipliers must be finite")
        if not np.all(np.isfinite(self.input_shifts)):
            raise ValueError("tie-dye input_shifts must be finite")
        if not np.all(np.isfinite(self.frequencies)):
            raise ValueError("tie-dye frequencies must be finite")
        if not np.all(np.isfinite(self.phases)):
            raise ValueError("tie-dye phases must be finite")
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("tie-dye amplitude must be finite and non-negative")
        if not np.isfinite(self.offset):
            raise ValueError("tie-dye offset must be finite")

    def __call__(self, value: RGB, context: StyleContext) -> RGB:
        del context
        return self.offset + self.cosine_sign * self.amplitude * np.cos(
            self.frequencies * as_rgb(value) + self.phases
        )

# 工厂函数
def build_function(spec):
    spec = dict(spec or {"type": "identity"})
    function_type = spec.pop("type", "identity").lower()
    constructors = {
        "identity": Identity,
        "gamma": Gamma,
        "saturation": Saturation,
        "color_map": ColorMap,
        "colormap": ColorMap,
        "color_map_nonlinear": ColorMap_Nonlinear,
        "colormap_nonlinear": ColorMap_Nonlinear,
        "nonlinear_colormap": ColorMap_Nonlinear,
        "cel": Cel,
        "crosshatch": CrossHatch,
        "cross_hatch": CrossHatch,
        "halftone": Halftone,
        "tone_hatch": ToneHatch,
        "image_hatch": ToneHatch,
        "tone_halftone": ToneHalftone,
        "image_halftone": ToneHalftone,
        "gooch": Gooch,
        "tie_dye": TieDye,
    }
    if function_type not in constructors:
        raise ValueError(f"Unknown style function '{function_type}'")
    if "input_range" in spec:
        spec["input_range"] = tuple(spec["input_range"])
    return constructors[function_type](**spec)

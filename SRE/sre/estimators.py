"""Estimators for functions of expectation from Sections 4.1--4.4.

Every estimator consumes independent samples through a zero-argument callback.
This mirrors Algorithm 1: in the renderer, one callback invocation expands one
independent child branch of the radiance tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

try:
    from .styles import LUMA, StyleContext, StyleFunction, as_rgb, build_function
except ImportError:
    from styles import LUMA, StyleContext, StyleFunction, as_rgb, build_function


Draw = Callable[[], np.ndarray] # 代表没有入参,且返回对象为np.ndarray的函数或对象


@dataclass
class EstimateStats:
    draws: int = 0 # 随机采样次数
    style_evaluations: int = 0 # 风格化采样次数
    inner_variance: float = 0.0 # 内部方差
    estimated_bias: float = 0.0 # 估计偏差

    # 合并统计数据
    def merge(self, other: "EstimateStats") -> None:
        self.draws += other.draws
        self.style_evaluations += other.style_evaluations
        self.inner_variance += other.inner_variance
        self.estimated_bias += other.estimated_bias


class Estimator(Protocol):
    def estimate(self, draw, rng, context, stats) -> np.ndarray: ...
    # draw: 采样数据
    # rng: 随机数源
    # context: 渲染上下文
    # stats: 统计收集器


class RandomSource(Protocol):
    def random(self) -> float: ...


def _draw(draw, stats):
    stats.draws += 1
    return as_rgb(draw())

# 批量采样矩阵声测会给你函数
def _sample_matrix(draw, count, stats):
    # draw: 函数对象
    # count: 物理采样次数
    # stats: 统计收集器
    if count < 1:
        raise ValueError("At least one inner sample is required")
    return np.stack([_draw(draw, stats) for _ in range(count)])


def _function(function, value, context, stats):
    # function: 风格化调用函数
    stats.style_evaluations += 1
    return as_rgb(function(as_rgb(value), context))

# 直接估计法的偏差估计——Eq.(21)
def delta_method_bias(function, samples, context, epsilon = 1e-3):
    # samples: 物理采样样本
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < 2:
        return np.zeros(3)
    mean = samples.mean(axis=0) # 计算平均向量
    covariance_of_mean = np.cov(samples, rowvar=False, ddof=1) / len(samples) # 计算样本均值的协方差矩阵,(3,3)
    hessian = np.empty((3, 3, 3), dtype=np.float64) # 第一维代表输出的RGB通道;2 3 维代表偏导组合
    f0 = as_rgb(function(mean, context)) # 函数在中心点mean处的基准输出
    for a in range(3): # 主对角线二次偏导
        ea = np.zeros(3)
        ea[a] = epsilon
        hessian[:, a, a] = (
            as_rgb(function(mean + ea, context)) - 2.0 * f0
            + as_rgb(function(mean - ea, context))
        ) / epsilon**2
        for b in range(a + 1, 3): # 非对角线混合偏导
            eb = np.zeros(3)
            eb[b] = epsilon
            mixed = (
                as_rgb(function(mean + ea + eb, context))
                - as_rgb(function(mean + ea - eb, context))
                - as_rgb(function(mean - ea + eb, context))
                + as_rgb(function(mean - ea - eb, context))
            ) / (4.0 * epsilon**2)
            hessian[:, a, b] = mixed
            hessian[:, b, a] = mixed
    return 0.5 * np.einsum("jab,ab->j", hessian, covariance_of_mean) # 最终RGB偏差

# 幂级数无偏估计
def unbiased_powers(samples, degree):
    # degree: 最高估计阶数
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    n, channels = samples.shape
    if degree > n:
        raise ValueError(f"Degree {degree} requires at least {degree} samples")
    elementary = np.zeros((degree + 1, channels), dtype=np.float64)
    elementary[0] = 1.0
    seen = 0
    for sample in samples:
        seen += 1
        for k in range(min(seen, degree), 0, -1):
            elementary[k] += sample * elementary[k - 1]
    for k in range(degree + 1):
        elementary[k] /= comb(n, k)
    return elementary # 输出估计值

# 恒等估计类
class IdentityEstimator:
    def estimate(self, draw, rng, context, stats):
        return _draw(draw, stats)


class ConstantEstimator:
    """Deterministic local style used for paper/ink feature-line renders.

    This is the constant SRE function ``g(L)=c``.  It intentionally performs
    no child radiance draw: line detection is evaluated before the material
    style, while identity-bound mirrors can still recurse to a later vertex
    whose constant material supplies the paper color.
    """

    def __init__(self, value=(1.0, 1.0, 1.0)):
        self.value = as_rgb(value)

    def estimate(self, draw, rng, context, stats):
        del draw, rng, context
        stats.style_evaluations += 1
        return self.value.copy()

# 直接应用估计器
class DirectApplicationEstimator:
    def __init__(
        self,
        function,
        samples=1,
        bias_correction="none",
        bias_epsilon=1e-3,
        recursive=True,
    ):
        # bias_correction: 偏差修正策略
        # bias_epsilon: Delta方法有限差分步长
        self.function = function
        self.samples = int(samples)
        self.bias_correction = bias_correction.lower()
        self.bias_epsilon = float(bias_epsilon)
        # ``False`` implements the paper's first-hit-only direct estimator:
        # the current style still averages every requested physical sample,
        # while those samples trace an ordinary radiance suffix instead of
        # recursively spawning another style expectation tree.
        self.recursive = bool(recursive)
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if self.bias_correction not in {"none", "delta", "jackknife"}:
            raise ValueError("bias_correction must be none, delta, or jackknife")

    def estimate(self, draw, rng, context, stats):
        samples = _sample_matrix(draw, self.samples, stats)
        mean = samples.mean(axis=0)
        result = _function(self.function, mean, context, stats) # 直接带入风格函数计算基础着色结果
        if self.samples > 1:
            stats.inner_variance += float(np.mean(np.var(samples, axis=0, ddof=1)) / self.samples) # 计算方差
            bias = delta_method_bias(self.function, samples, context, self.bias_epsilon) # 计算偏差
            stats.estimated_bias += float(np.mean(np.abs(bias)))
            if self.bias_correction == "delta":
                result -= bias
            elif self.bias_correction == "jackknife":
                leave_one_out = np.stack([
                    as_rgb(self.function((samples.sum(axis=0) - samples[i]) / (self.samples - 1), context))
                    for i in range(self.samples)
                ])
                result = self.samples * result - (self.samples - 1) * leave_one_out.mean(axis=0)
        return result

# 无偏有限多项式估计器 Eq.(17)
class PolynomialEstimator:
    def __init__(self, coefficients, sample_count = None, projection = "component", fit_interval=(0.0, 1.0), clamp_samples = False, normalized_domain = False, evaluation_precision = "float32"):
        # coefficients: 多项式系数矩阵
        # projection: 颜色通道投影模式
        # fit_interval: 色彩取值截断空间
        # clamp_samples: 样本是否截断
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim not in (1, 2):
            raise ValueError("coefficients must have shape (degree+1[, 3])")
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1
        self.sample_count = int(sample_count or max(1, self.degree))
        self.projection = projection
        self.fit_interval = tuple(map(float, fit_interval))
        self.clamp_samples = bool(clamp_samples)
        self.normalized_domain = bool(normalized_domain)
        self.evaluation_precision = str(evaluation_precision).lower()
        if self.sample_count < self.degree:
            raise ValueError("sample_count must be at least the polynomial degree")
        if projection not in {"component", "luminance"}:
            raise ValueError("projection must be component or luminance")
        if self.fit_interval[1] <= self.fit_interval[0]:
            raise ValueError("fit_interval must have positive width")
        if self.evaluation_precision not in {"float32", "float64"}:
            raise ValueError("evaluation_precision must be float32 or float64")

    def estimate(self, draw, rng, context, stats):
        samples = _sample_matrix(draw, self.sample_count, stats)
        if self.clamp_samples:
            samples = np.clip(samples, *self.fit_interval)
        if self.normalized_domain:
            low, high = self.fit_interval
            samples = 2.0 * (samples - low) / (high - low) - 1.0
        if self.projection == "luminance": # 基于亮度的标量估计
            scalar_samples = samples.dot(LUMA)
            powers = unbiased_powers(scalar_samples, self.degree)[:, 0]
            result = np.sum(self.coefficients * powers[:, None], axis=0)
        else: # R G B三通道独立估计
            powers = unbiased_powers(samples, self.degree)
            coefficients = self.coefficients
            if coefficients.ndim == 1:
                coefficients = coefficients[:, None]
            result = np.sum(coefficients * powers, axis=0)
        stats.style_evaluations += 1
        return as_rgb(result)

# 多项式拟合函数
def fit_polynomial(function, degree, interval=(0.0, 1.0),
                   projection = "luminance", basis = "chebyshev",
                   fit_samples = 1024, normalized_domain = False):
    low, high = map(float, interval)
    if high <= low:
        raise ValueError("fit interval must have positive width")
    x = np.linspace(low, high, max(int(fit_samples), degree + 1))
    fit_x = 2.0 * (x - low) / (high - low) - 1.0 if normalized_domain else x
    fit_domain = [-1.0, 1.0] if normalized_domain else [low, high]
    context = StyleContext()
    values = np.stack([as_rgb(function(np.repeat(v, 3), context)) for v in x])
    coefficients = np.zeros((degree + 1, 3), dtype=np.float64)
    for channel in range(3):
        if basis == "chebyshev": # 切比可夫拟合与转换
            polynomial = Chebyshev.fit(fit_x, values[:, channel], degree, domain=fit_domain).convert(
                kind=Polynomial
            )
        elif basis in {"least_squares", "power"}:
            polynomial = Polynomial.fit(
                fit_x, values[:, channel], degree, domain=fit_domain
            ).convert()
        else:
            raise ValueError("basis must be chebyshev or least_squares")
        coefficients[:, channel] = np.pad(
            polynomial.coef, (0, degree + 1 - len(polynomial.coef))
        )[:degree + 1]
    if projection == "component":
        return coefficients
    return coefficients # 返回系数矩阵

# 针对Gamma校正设计的无偏估计器 Eq(24)
class GammaPowerSeriesEstimator:
    def __init__(self, gamma = 2.2, expansion_point = 0.5,
                 continuation_probability = 0.55, pilot_samples = 0,
                 min_expansion_point = 0.1, oversampling = 1):
        # gamma: 校正值
        # expansion_point: 静态泰勒展开点
        # continuation_probability: 俄式轮盘赌的存活率
        # pilot_samples: 预探索采样数
        # min_expansion_point: 展开点下限
        # oversampling: 超采样乘子
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if not 0.0 < continuation_probability < 1.0:
            raise ValueError("continuation_probability must be in (0, 1)")
        self.exponent = 1.0 / float(gamma)
        self.expansion_point = float(expansion_point)
        self.continuation_probability = float(continuation_probability)
        self.pilot_samples = int(pilot_samples)
        self.min_expansion_point = float(min_expansion_point)
        self.oversampling = max(1, int(oversampling))

    def estimate(self, draw, rng, context, stats):
        if self.pilot_samples:
            pilot = _sample_matrix(draw, self.pilot_samples, stats).mean(axis=0)
            b = np.maximum(pilot, self.min_expansion_point)
        else:
            b = np.repeat(max(self.expansion_point, self.min_expansion_point), 3)
        degree = 0
        while rng.random() < self.continuation_probability:
            degree += 1
        count = max(1, degree * self.oversampling)
        samples = _sample_matrix(draw, count, stats) - b
        powers = unbiased_powers(samples, degree)
        result = np.zeros(3)
        falling = 1.0
        factorial = 1.0
        for k in range(degree + 1):
            if k:
                falling *= self.exponent - (k - 1)
                factorial *= k
            coefficient = np.power(b, self.exponent - k) * falling / factorial
            survival = self.continuation_probability**k
            result += coefficient * powers[k] / survival
        stats.style_evaluations += 1
        return result

# 伸缩级数估计器 Eq(15)
class TelescopingEstimator:
    def __init__(self, function, base_samples = 1, continuation_probability = 0.5):
        if base_samples < 1:
            raise ValueError("base_samples must be positive")
        if not 0.0 < continuation_probability < 1.0:
            raise ValueError("continuation_probability must be in (0, 1)")
        self.function = function
        self.base_samples = int(base_samples)
        self.continuation_probability = float(continuation_probability)

    def estimate(self, draw, rng, context, stats):
        samples = list(_sample_matrix(draw, self.base_samples, stats))
        running_sum = np.sum(samples, axis=0)
        previous = _function(self.function, running_sum / len(samples), context, stats)
        result = previous.copy()
        correction = 0
        while rng.random() < self.continuation_probability:
            correction += 1
            running_sum += _draw(draw, stats)
            current = _function(self.function, running_sum / (self.base_samples + correction), context, stats)
            result += (current - previous) / self.continuation_probability**correction
            previous = current
        return result

# 光线采样缓存与重放
class _ReplaySource:
    def __init__(self, draw):
        self.draw = draw
        self.cache = []
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def __call__(self):
        # 缓存未命中
        if self.cursor == len(self.cache):
            self.cache.append(as_rgb(self.draw()))
        value = self.cache[self.cursor]
        self.cursor += 1
        return value.copy()

# 加法复合估计器
class AdditionEstimator:
    def __init__(self, left, right):
        # left, right: 估计器
        self.left, self.right = left, right

    def estimate(self, draw, rng, context, stats) -> np.ndarray:
        source = _ReplaySource(draw)
        left_stats, right_stats = EstimateStats(), EstimateStats()
        left = self.left.estimate(source, rng, context, left_stats) # 评估左侧估计器
        source.reset()
        right = self.right.estimate(source, rng, context, right_stats) # 评估右侧估计器
        stats.draws += len(source.cache)
        stats.style_evaluations += left_stats.style_evaluations + right_stats.style_evaluations
        stats.inner_variance += left_stats.inner_variance + right_stats.inner_variance
        stats.estimated_bias += left_stats.estimated_bias + right_stats.estimated_bias
        return left + right

# 乘法复合估计器
class MultiplicationEstimator:
    def __init__(self, left, right):
        self.left, self.right = left, right

    def estimate(self, draw, rng, context, stats) -> np.ndarray:
        return self.left.estimate(draw, rng, context, stats) * self.right.estimate(
            draw, rng, context, stats
        )

# 函数符合估计器
class CompositionEstimator:
    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner

    def estimate(self, draw, rng, context, stats):
        return self.outer.estimate(
            lambda: self.inner.estimate(draw, rng, context, stats), rng, context, stats
        )

# 工厂函数
def build_estimator(spec):
    spec = dict(spec or {"estimator": "identity"})
    estimator_type = str(spec.pop("estimator", spec.pop("type", "identity"))).lower()
    if estimator_type == "identity": # 恒等估计器
        return IdentityEstimator()
    if estimator_type in {"constant", "flat"}:
        value = spec.pop("value", spec.pop("color", (1.0, 1.0, 1.0)))
        if spec:
            raise ValueError(
                f"Unknown constant estimator fields: {sorted(spec)}"
            )
        return ConstantEstimator(value)
    if estimator_type == "direct": # 直接应用估计器
        function = build_function(spec.pop("function", None))
        return DirectApplicationEstimator(function=function, **spec)
    if estimator_type == "polynomial": # 多项式估计器
        interval = tuple(spec.pop("fit_interval", (0.0, 1.0)))
        projection = spec.pop("projection", None)
        normalized_domain = bool(spec.pop("normalized_domain", False))
        raw_coefficients = spec.pop("coefficients", None)
        # 用户未传入多项式系数,在线拟合
        if raw_coefficients is None:
            projection = projection or "luminance"
            function = build_function(spec.pop("function", None))
            degree = int(spec.pop("degree", 7))
            basis = spec.pop("basis", "chebyshev")
            fit_samples = int(spec.pop("fit_samples", 1024))
            coefficients = fit_polynomial(
                function, degree, interval, projection, basis, fit_samples,
                normalized_domain,
            )
        # 用户传入多项式系数 
        else:
            coefficients = np.asarray(raw_coefficients, dtype=np.float64)
            projection = projection or "component"
        return PolynomialEstimator(
            coefficients, projection=projection, fit_interval=interval,
            normalized_domain=normalized_domain, **spec
        )
    if estimator_type in {"power_series_gamma", "gamma_power_series"}: # Gamma幂级数估计器
        return GammaPowerSeriesEstimator(**spec)
    if estimator_type == "telescoping": # 伸缩计数估计器
        function = build_function(spec.pop("function", None))
        return TelescopingEstimator(function=function, **spec)
    if estimator_type in {"add", "addition"}: 
        return AdditionEstimator(build_estimator(spec["left"]), build_estimator(spec["right"]))
    if estimator_type in {"multiply", "multiplication"}:
        return MultiplicationEstimator(
            build_estimator(spec["left"]), build_estimator(spec["right"])
        )
    if estimator_type in {"compose", "composition"}:
        return CompositionEstimator(
            outer=build_estimator(spec["outer"]), inner=build_estimator(spec["inner"])
        )
    raise ValueError(f"Unknown estimator '{estimator_type}'")

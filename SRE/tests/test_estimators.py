from __future__ import annotations

import numpy as np

from sre.estimators import (
    ConstantEstimator,
    CompositionEstimator,
    DirectApplicationEstimator,
    EstimateStats,
    GammaPowerSeriesEstimator,
    MultiplicationEstimator,
    PolynomialEstimator,
    TelescopingEstimator,
    delta_method_bias,
    unbiased_powers,
    build_estimator,
)
from sre.styles import StyleContext, build_function


class RNG:
    def __init__(self, seed=3):
        self.generator = np.random.default_rng(seed)

    def random(self):
        return float(self.generator.random())


class Square:
    def __call__(self, value, context):
        return np.asarray(value) ** 2


class Affine:
    def __init__(self, scale, offset):
        self.scale, self.offset = scale, offset

    def __call__(self, value, context):
        return self.scale * np.asarray(value) + self.offset


def test_constant_estimator_returns_paper_color_without_child_draw():
    estimator = build_estimator(
        {"estimator": "constant", "value": [0.97, 0.96, 0.95]}
    )
    assert isinstance(estimator, ConstantEstimator)
    draws = 0

    def draw():
        nonlocal draws
        draws += 1
        return np.zeros(3)

    stats = EstimateStats()
    result = estimator.estimate(draw, RNG(), StyleContext(), stats)
    np.testing.assert_allclose(result, [0.97, 0.96, 0.95])
    assert draws == 0
    assert stats.draws == 0
    assert stats.style_evaluations == 1


def test_unbiased_power_recurrence_uses_distinct_samples():
    samples = np.array([[1.0], [2.0], [4.0]])
    powers = unbiased_powers(samples, 3)[:, 0]
    np.testing.assert_allclose(powers, [1.0, 7.0 / 3.0, 14.0 / 3.0, 8.0])


def test_polynomial_estimator_is_unbiased_for_square():
    rng = np.random.default_rng(8)
    estimates = []
    estimator = PolynomialEstimator(np.array([0.0, 0.0, 1.0]), sample_count=2)
    for _ in range(30000):
        estimate = estimator.estimate(
            lambda: np.repeat(rng.exponential(0.7), 3), RNG(), StyleContext(), EstimateStats()
        )
        estimates.append(estimate[0])
    assert abs(np.mean(estimates) - 0.7**2) < 0.015


def test_degree_20_tie_dye_fit_is_stable_on_normalized_domain():
    function_spec = {
        "type": "tie_dye",
        "frequency_multipliers": [2.0, 2.15, 2.3],
        "input_shifts": [1.0, 1.13, 1.29],
        "amplitude": 0.5,
        "offset": 0.5,
        "inverted": True,
    }
    estimator = build_estimator({
        "estimator": "polynomial",
        "degree": 20,
        "sample_count": 32,
        "projection": "component",
        "fit_interval": [-1.0, 4.0],
        "normalized_domain": True,
        "evaluation_precision": "float64",
        "clamp_samples": True,
        "fit_samples": 4096,
        "function": function_spec,
    })
    function = build_function(function_spec)
    context = StyleContext()
    errors = []
    for value in np.linspace(-1.0, 4.0, 101):
        estimate = estimator.estimate(
            lambda value=value: np.repeat(value, 3),
            RNG(), context, EstimateStats(),
        )
        expected = function(np.repeat(value, 3), context)
        errors.append(estimate - expected)

    # This is the paper's deliberately finite degree-20 approximation, not an
    # exact cosine representation. It should nevertheless be a tight RGB fit.
    assert float(np.sqrt(np.mean(np.square(errors)))) < 0.03

    # The five-cycle fit has large, cancelling power-basis coefficients and
    # therefore explicitly requests Float64 evaluation on CUDA.
    assert np.max(np.abs(estimator.coefficients)) > 1e5
    assert estimator.evaluation_precision == "float64"


def test_direct_application_bias_matches_equation_21_for_square():
    rng = np.random.default_rng(10)
    samples = np.repeat(rng.normal(0.8, 0.3, size=(200000, 1)), 3, axis=1)
    count = 20
    groups = samples.reshape(-1, count, 3)
    measured = np.mean(groups.mean(axis=1)[:, 0] ** 2) - 0.8**2
    predicted = delta_method_bias(Square(), groups[0], StyleContext())[0]
    expected = 0.3**2 / count
    assert abs(measured - expected) < 6e-4
    assert abs(predicted - np.var(groups[0][:, 0], ddof=1) / count) < 1e-6


def test_delta_bias_correction_removes_quadratic_bias():
    generator = np.random.default_rng(4)
    estimator = DirectApplicationEstimator(Square(), samples=12, bias_correction="delta")
    values = [
        estimator.estimate(
            lambda: np.repeat(generator.normal(0.6, 0.25), 3), RNG(),
            StyleContext(), EstimateStats(),
        )[0]
        for _ in range(12000)
    ]
    assert abs(np.mean(values) - 0.6**2) < 0.006


def test_telescoping_is_exact_for_deterministic_affine_function():
    estimator = TelescopingEstimator(Affine(2.0, 0.3), continuation_probability=0.6)
    result = estimator.estimate(
        lambda: np.array([0.4, 0.5, 0.6]), RNG(), StyleContext(), EstimateStats()
    )
    np.testing.assert_allclose(result, [1.1, 1.3, 1.5])


def test_gamma_power_series_on_deterministic_input():
    estimator = GammaPowerSeriesEstimator(
        gamma=2.0, expansion_point=0.5, continuation_probability=0.65,
        oversampling=2,
    )
    values = [
        estimator.estimate(
            lambda: np.repeat(0.64, 3), RNG(seed), StyleContext(), EstimateStats()
        )[0]
        for seed in range(25000)
    ]
    assert abs(np.mean(values) - 0.8) < 0.006


def test_group_unbiased_multiplication_and_composition():
    linear = PolynomialEstimator(np.array([0.0, 1.0]), sample_count=1)
    square = PolynomialEstimator(np.array([0.0, 0.0, 1.0]), sample_count=2)
    product = MultiplicationEstimator(linear, square)
    composition = CompositionEstimator(outer=square, inner=linear)
    generator = np.random.default_rng(2)

    def draw():
        return np.repeat(generator.exponential(0.5), 3)

    product_values = [
        product.estimate(draw, RNG(i), StyleContext(), EstimateStats())[0]
        for i in range(15000)
    ]
    composition_values = [
        composition.estimate(draw, RNG(i), StyleContext(), EstimateStats())[0]
        for i in range(15000)
    ]
    assert abs(np.mean(product_values) - 0.5**3) < 0.01
    assert abs(np.mean(composition_values) - 0.5**2) < 0.015

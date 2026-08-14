from __future__ import annotations

from pathlib import Path

import numpy as np

from sre.config import load_config, load_config_data
from sre.estimators import (
    DirectApplicationEstimator,
    IdentityEstimator,
    PolynomialEstimator,
)
from sre.styles import (
    Cel,
    ColorMap_Nonlinear,
    CrossHatch,
    Halftone,
    StyleContext,
    TieDye,
    ToneHatch,
    ToneHalftone,
    build_function,
)


def test_material_level_and_first_hit_parameterization():
    config = load_config({
        "materials": {
            "paper": {
                "estimator": "direct",
                "samples": 4,
                "function": {"type": "cel", "levels": 3},
                "when": {"first_hit": True, "depths": [0, 2]},
            }
        }
    })
    active = StyleContext(depth=2, material_id="paper", occurrence=1)
    repeated = StyleContext(depth=2, material_id="paper", occurrence=2)
    assert isinstance(config.resolve("paper", "shape", active), DirectApplicationEstimator)
    assert isinstance(config.resolve("paper", "shape", repeated), IdentityEstimator)


def test_file_config_inheritance_deep_merges_objects_and_replaces_lists(tmp_path):
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    parent.write_text(
        '{"metadata":{"mode":"left","nested":{"a":1,"b":2}},'
        '"feature_lines":{"enabled":true,"auxiliary_samples":4,'
        '"types":[{"measurement":"depth"}]},'
        '"default":{"estimator":"identity"}}',
        encoding="utf-8",
    )
    child.write_text(
        '{"extends":"parent.json","metadata":{"mode":"right",'
        '"nested":{"b":3}},"feature_lines":{"types":['
        '{"measurement":"normal"}]}}',
        encoding="utf-8",
    )

    data = load_config_data(child)

    assert "extends" not in data
    assert data["metadata"] == {"mode": "right", "nested": {"a": 1, "b": 3}}
    assert data["feature_lines"]["auxiliary_samples"] == 4
    assert data["feature_lines"]["types"] == [{"measurement": "normal"}]


def test_cross_hatching_is_object_space_and_brightness_dependent():
    hatch = CrossHatch(scale=1.0, width=0.2)
    dark = hatch(np.zeros(3), StyleContext(position=np.zeros(3)))
    bright = hatch(np.ones(3), StyleContext(position=np.zeros(3)))
    assert np.mean(dark) < np.mean(bright)


def test_cel_uses_configurable_nonuniform_band_thresholds():
    cel = Cel(
        levels=4,
        thresholds=(0.1, 0.4, 0.85),
        palette=[
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.6, 0.6, 0.6],
            [0.95, 0.95, 0.95],
        ],
    )
    np.testing.assert_allclose(
        cel(np.repeat(0.3, 3), StyleContext()), [0.3, 0.3, 0.3]
    )
    np.testing.assert_allclose(
        cel(np.repeat(0.7, 3), StyleContext()), [0.6, 0.6, 0.6]
    )


def test_fig8_cel_matches_supplemental_s5_2_mapping():
    cel = Cel(
        levels=2,
        thresholds=(0.75,),
        band_values=(0.4, 0.95),
        brightness_mode="mean",
    )
    context = StyleContext()

    shadow = np.array([0.3, 0.45, 0.6])
    shadow_result = cel(shadow, context)
    np.testing.assert_allclose(shadow_result, shadow * (0.4 / np.mean(shadow)))
    assert abs(float(np.mean(shadow_result)) - 0.4) < 1e-12

    highlight = np.array([0.6, 0.8, 1.0])
    highlight_result = cel(highlight, context)
    np.testing.assert_allclose(
        highlight_result, highlight * (0.95 / np.mean(highlight))
    )
    assert abs(float(np.mean(highlight_result)) - 0.95) < 1e-12


def test_fig8_cel_uses_mean_rgb_not_rec709_luminance():
    cel = Cel(
        levels=2,
        thresholds=(0.75,),
        band_values=(0.4, 0.95),
        brightness_mode="mean",
    )
    value = np.array([1.0, 0.6, 0.8])
    assert np.mean(value) >= 0.75
    assert np.dot(value, [0.2126, 0.7152, 0.0722]) < 0.75
    assert abs(float(np.mean(cel(value, StyleContext()))) - 0.95) < 1e-12


def test_palette_cel_can_retain_gi_chroma_without_changing_band_luminance():
    cel = Cel(
        levels=2,
        thresholds=(0.5,),
        palette=[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
        chroma_strength=0.5,
    )
    result = cel(np.array([1.0, 0.2, 0.2]), StyleContext())
    assert not np.allclose(result, cel.palette[0])
    assert abs(float(np.dot(result, [0.2126, 0.7152, 0.0722])) - 0.2) < 1e-12


def test_palette_cel_can_reduce_color_bleed_in_highlight_bands():
    cel = Cel(
        levels=2,
        thresholds=(0.5,),
        palette=[[0.2, 0.3, 0.4], [1.0, 0.99, 0.94]],
        chroma_strength=0.5,
        chroma_weights=(1.0, 0.0),
    )
    result = cel(np.array([1.0, 0.8, 0.6]), StyleContext())
    np.testing.assert_allclose(result, cel.palette[1])


def test_cross_hatching_activates_nested_direction_families():
    hatch = CrossHatch(
        scale=1.0,
        width=0.04,
        directions=((1.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
        activation_thresholds=(0.2, 0.7),
        phase_offsets=(0.0, 0.0),
        scale_factors=(1.0, 1.0),
        family_widths=(1.0, 1.0),
        width_growth=0.0,
        edge_softness=0.0,
    )
    context = StyleContext(position=np.array([0.25, 0.0, 0.0]))
    middle_tone = hatch(np.repeat(0.5, 3), context)
    dark_tone = hatch(np.repeat(0.1, 3), context)
    np.testing.assert_allclose(middle_tone, hatch.paper)
    np.testing.assert_allclose(dark_tone, hatch.ink)


def test_cross_hatching_normalizes_plane_directions_for_even_spacing():
    hatch = CrossHatch(
        scale=1.0,
        width=0.03,
        directions=((10.0, 0.0, 0.0),),
        activation_thresholds=(0.0,),
        phase_offsets=(0.0,),
        scale_factors=(1.0,),
        family_widths=(1.0,),
        width_growth=0.0,
        edge_softness=0.0,
    )
    context = StyleContext(position=np.array([0.025, 0.0, 0.0]))
    np.testing.assert_allclose(hatch(np.zeros(3), context), hatch.ink)


def test_f11_red_robot_uses_one_stable_object_space_hatch_family():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    )
    estimator = config.materials["mat-f11-red"].estimator
    assert isinstance(estimator, DirectApplicationEstimator)
    hatch = estimator.function
    assert isinstance(hatch, CrossHatch)
    assert hatch.directions.shape == (1, 3)
    assert hatch.scale == 42.0
    assert hatch.width == 0.105
    np.testing.assert_allclose(hatch.activation_thresholds, [0.0])
    np.testing.assert_allclose(hatch.ink, [0.035, 0.0006, 0.0004])
    np.testing.assert_allclose(hatch.paper, [0.003, 0.0002, 0.006])
    red_lines = [
        line for line in config.feature_lines.types
        if line.include_materials == ("mat-f11-red",)
    ]
    assert len(red_lines) == 2
    for line in red_lines:
        assert line.line_hatch_scale == 42.0
        assert line.line_hatch_edge_softness == 0.012
        np.testing.assert_allclose(
            line.line_hatch_direction,
            np.array([0.66162164, 0.66162164, 0.35286487])
            / np.linalg.norm([0.66162164, 0.66162164, 0.35286487]),
        )
    assert all(line.measurement == "normal" for line in red_lines)
    assert all(line.normal_orientation_invariant for line in red_lines)
    assert red_lines[0].threshold == 0.48
    assert red_lines[0].stencil_radius == 8.0
    assert red_lines[0].include_silhouette
    assert red_lines[0].line_hatch_width == 0.240
    assert red_lines[1].threshold == 0.10
    assert red_lines[1].stencil_radius == 5.0
    assert not red_lines[1].include_silhouette
    assert red_lines[1].line_hatch_width == 0.220


def test_halftone_uses_a_three_dimensional_sphere_lattice():
    halftone = Halftone(
        scale=1.0,
        min_radius=0.1,
        max_radius=0.3,
        dot_threshold=0.0,
        edge_softness=0.0,
        min_ink_strength=1.0,
        phase=(0.0, 0.0, 0.0),
        orientation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    center = StyleContext(position=np.array([0.5, 0.5, 0.5]))
    displaced_z = StyleContext(position=np.array([0.5, 0.5, 0.0]))
    np.testing.assert_allclose(halftone(np.zeros(3), center), halftone.ink)
    np.testing.assert_allclose(
        halftone(np.zeros(3), displaced_z), halftone.paper
    )


def test_halftone_dot_radius_and_density_increase_in_dark_tones():
    halftone = Halftone(
        scale=1.0,
        min_radius=0.05,
        max_radius=0.4,
        radius_gamma=1.0,
        dot_threshold=0.0,
        edge_softness=0.0,
        min_ink_strength=1.0,
        phase=(0.0, 0.0, 0.0),
        orientation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    context = StyleContext(position=np.array([0.8, 0.5, 0.5]))
    light = halftone(np.repeat(0.8, 3), context)
    dark = halftone(np.repeat(0.1, 3), context)
    np.testing.assert_allclose(light, halftone.paper)
    np.testing.assert_allclose(dark, halftone.ink)


def test_all_paper_style_functions_construct_and_return_rgb():
    specs = [
        {"type": "identity"},
        {"type": "gamma"},
        {"type": "saturation"},
        {"type": "color_map", "colors": [[0, 0, 0], [1, 0.8, 0.2]]},
        {
            "type": "color_map_nonlinear",
            "colors": [[0, 0, 0], [1, 0.8, 0.2]],
        },
        {"type": "cel"},
        {"type": "crosshatch"},
        {"type": "halftone"},
        {"type": "tone_hatch"},
        {"type": "tone_halftone"},
        {"type": "gooch"},
        {"type": "tie_dye"},
    ]
    for spec in specs:
        result = build_function(spec)(np.array([0.2, 0.4, 0.6]), StyleContext())
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


def test_nonlinear_color_map_uses_smoothstep_not_linear_interpolation():
    function = ColorMap_Nonlinear(
        colors=[[0.0, 0.0, 0.0], [1.0, 0.8, 0.2]],
        input_range=(0.0, 1.0),
    )
    context = StyleContext()
    value = np.repeat(0.25, 3)
    linear_weight = 0.25
    smoothstep_weight = 0.25**2 * (3.0 - 2.0 * 0.25)
    linear_result = linear_weight * np.array([1.0, 0.8, 0.2])
    expected = smoothstep_weight * np.array([1.0, 0.8, 0.2])
    result = function(value, context)
    assert np.allclose(result, expected)
    assert not np.allclose(result, linear_result)


def test_tie_dye_is_a_component_wise_rgb_cosine():
    function = TieDye(
        frequencies=(5.0, 7.0, 11.0),
        phases=(0.0, 2.094395, 4.18879),
        amplitude=0.5,
        offset=0.5,
    )
    value = np.array([0.2, 0.4, 0.6])
    expected = 0.5 + 0.5 * np.cos(
        np.array([5.0, 7.0, 11.0]) * value
        + np.array([0.0, 2.094395, 4.18879])
    )
    np.testing.assert_allclose(function(value, StyleContext()), expected)


def test_fig11_tie_dye_matches_supplemental_s5_5_formula():
    function = TieDye()
    value = np.array([0.2, 0.4, 0.6])
    multipliers = np.array([2.0, 2.15, 2.3])
    shifts = np.array([1.0, 1.13, 1.29])
    expected = (-np.cos(np.pi * multipliers * (value + shifts)) + 1.0) / 2.0
    np.testing.assert_allclose(function(value, StyleContext()), expected)


def test_dragon_tie_dye_config_matches_supplemental_s5_5():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/dragon_tie_dye.json")
    estimator = config.materials["dragon"].estimator

    assert isinstance(estimator, PolynomialEstimator)
    assert estimator.degree == 20
    assert estimator.sample_count == 32
    assert estimator.projection == "component"
    assert estimator.fit_interval == (-1.0, 4.0)
    assert estimator.clamp_samples
    assert estimator.normalized_domain
    assert estimator.evaluation_precision == "float64"

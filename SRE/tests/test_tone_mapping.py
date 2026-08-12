from __future__ import annotations

from pathlib import Path

import numpy as np

from sre.config import load_config
from sre.styles import StyleContext, ToneHatch, ToneHalftone
from sre.tone_mapping import (
    ToneMappingConfig,
    linear_mls_inverse,
    multiscale_anchor_offsets,
)


def test_linear_mls_recovers_an_affine_image_coordinate() -> None:
    points = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    coordinates = np.column_stack((
        20.0 + 2.0 * points[:, 0] + 3.0 * points[:, 1],
        12.0 - points[:, 0] + 0.5 * points[:, 1],
    ))
    query = np.array([0.2, -0.35, 0.0])
    result = linear_mls_inverse(
        points,
        coordinates,
        query,
        [0.0, 0.0, 1.0],
        min_candidates=4,
    )
    expected = np.array([
        20.0 + 2.0 * query[0] + 3.0 * query[1],
        12.0 - query[0] + 0.5 * query[1],
    ])
    assert result.method == "linear_mls"
    np.testing.assert_allclose(result.coordinate, expected, atol=1e-10)


def test_sparse_tone_inverse_uses_nearest_neighbor_fallback() -> None:
    result = linear_mls_inverse(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        [[10.0, 20.0], [30.0, 40.0]],
        [0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        min_candidates=4,
    )
    assert result.method == "nearest"
    np.testing.assert_allclose(result.coordinate, [10.0, 20.0])


def test_multiscale_offsets_include_center_local_and_paper_radius() -> None:
    config = ToneMappingConfig(
        enabled=True,
        anchor_samples=16,
        min_radius=1.5,
        search_radius=256.0,
        radial_rings=4,
    )
    offsets = multiscale_anchor_offsets(config)
    radii = np.array([np.linalg.norm(offset) for offset in offsets])
    assert len(offsets) == 16
    assert radii[0] == 0.0
    assert np.any(np.isclose(radii, 1.5))
    assert np.any(np.isclose(radii, 256.0))


def test_tone_hatch_uses_lifted_coordinate_not_world_position() -> None:
    hatch = ToneHatch(
        spacing=8.0,
        angles_degrees=(0.0,),
        activation_thresholds=(0.0,),
        phase_offsets=(0.0,),
        family_widths=(1.0,),
        min_coverage=0.1,
        max_coverage=0.7,
        edge_softness=0.0,
    )
    value = np.zeros(3)
    ink = hatch(
        value,
        StyleContext(
            position=np.array([100.0, 100.0, 100.0]),
            tone_coordinate=np.array([0.0, 0.0]),
        ),
    )
    paper = hatch(
        value,
        StyleContext(
            position=np.array([100.0, 100.0, 100.0]),
            tone_coordinate=np.array([0.0, 4.0]),
        ),
    )
    np.testing.assert_allclose(ink, hatch.ink)
    np.testing.assert_allclose(paper, hatch.paper)


def test_tone_hatch_region_smoothly_limits_ground_pattern() -> None:
    hatch = ToneHatch(
        spacing=8.0,
        angles_degrees=(0.0,),
        activation_thresholds=(0.0,),
        phase_offsets=(0.0,),
        family_widths=(1.0,),
        min_coverage=0.1,
        max_coverage=0.7,
        edge_softness=0.0,
        region_center=(8.0, 16.0),
        region_radius=(8.0, 4.0),
        region_feather=0.25,
    )
    value = np.zeros(3)
    inside = hatch(
        value, StyleContext(tone_coordinate=np.array([8.0, 16.0]))
    )
    outside = hatch(
        value, StyleContext(tone_coordinate=np.array([28.0, 16.0]))
    )
    np.testing.assert_allclose(inside, hatch.ink)
    np.testing.assert_allclose(outside, hatch.paper)


def test_tone_hatch_activates_zero_to_three_families_by_darkness() -> None:
    hatch = ToneHatch(
        spacing=8.0,
        angles_degrees=(0.0, 90.0, 45.0),
        activation_thresholds=(0.2, 0.5, 0.8),
        phase_offsets=(0.0, 0.0, 0.0),
        family_widths=(1.0, 1.0, 1.0),
        min_coverage=0.1,
        max_coverage=0.2,
        darkness_gamma=1.0,
        edge_softness=0.0,
        max_value=1.0,
    )
    first_line = StyleContext(tone_coordinate=np.array([4.0, 0.0]))
    second_line = StyleContext(tone_coordinate=np.array([0.0, 4.0]))
    third_line = StyleContext(tone_coordinate=np.array([4.0, 4.0]))

    # Bright values fall below every threshold, including directly on a line.
    np.testing.assert_allclose(hatch(np.ones(3), first_line), hatch.paper)

    # Darkness 0.3 activates only the horizontal family.
    medium = np.full(3, 0.7)
    np.testing.assert_allclose(hatch(medium, first_line), hatch.ink)
    np.testing.assert_allclose(hatch(medium, second_line), hatch.paper)

    # Darkness 0.6 additionally activates the vertical family, but not the
    # diagonal one whose line passes through (4, 4).
    dark = np.full(3, 0.4)
    np.testing.assert_allclose(hatch(dark, second_line), hatch.ink)
    np.testing.assert_allclose(hatch(dark, third_line), hatch.paper)

    # Darkness 0.9 crosses the final threshold and enables all three families.
    darkest = np.full(3, 0.1)
    np.testing.assert_allclose(hatch(darkest, third_line), hatch.ink)


def test_tone_halftone_dot_radius_responds_to_expected_brightness() -> None:
    halftone = ToneHalftone(
        spacing=8.0,
        angle_degrees=0.0,
        min_radius=0.2,
        max_radius=3.5,
        edge_softness=0.0,
        phase=(0.0, 0.0),
    )
    # With zero phase the cell centre is (4, 4); query three pixels away.
    context = StyleContext(tone_coordinate=np.array([7.0, 4.0]))
    light = halftone(np.ones(3), context)
    dark = halftone(np.zeros(3), context)
    np.testing.assert_allclose(light, halftone.paper)
    np.testing.assert_allclose(dark, halftone.ink)


def test_tone_mapping_configuration_loads_with_style_configuration() -> None:
    config = load_config({
        "tone_mapping": {
            "enabled": True,
            "anchor_samples": 12,
            "search_radius": 64.0,
        },
        "default": {
            "estimator": "direct",
            "samples": 1,
            "function": {"type": "tone_hatch"},
        },
    })
    assert config.tone_mapping.enabled
    assert config.tone_mapping.anchor_samples == 12
    assert config.tone_mapping.search_radius == 64.0


def test_fig1_combines_geometric_normal_lines_and_mls_tone_mapping() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "llat_feature_lines.json"
    )
    assert config.feature_lines.enabled
    assert config.tone_mapping.enabled
    assert config.tone_mapping.anchor_samples == 16
    assert config.tone_mapping.search_radius == 512.0
    assert config.tone_mapping.max_depth == 4
    assert all(
        line.measurement == "normal" for line in config.feature_lines.types
    )
    assert all(
        line.max_depth <= 3 for line in config.feature_lines.types
    )
    assert config.materials["mat-材质"].estimator.__class__.__name__ == "IdentityEstimator"
    assert (
        config.materials["mat-Cromo_claro_blinn"]
        .estimator.__class__.__name__
        == "IdentityEstimator"
    )
    assert (
        config.materials["mat-llat-curved-reflector"]
        .estimator.__class__.__name__
        == "IdentityEstimator"
    )
    default_hatch = config.default.estimator.function
    np.testing.assert_allclose(
        default_hatch.activation_thresholds, [0.32, 0.60]
    )
    assert len(default_hatch.angles_degrees) == 2
    assert default_hatch.min_coverage == 0.0
    assert default_hatch.max_coverage == 0.30
    small_robot = config.materials["mat-lambert6"].estimator.function
    np.testing.assert_allclose(
        small_robot.activation_thresholds, [0.30, 0.60]
    )
    np.testing.assert_allclose(small_robot.paper, [0.940, 0.940, 0.936])
    reconstructed_robot = config.materials[
        "mat-fig1-small-robot"
    ].estimator.function
    np.testing.assert_allclose(
        reconstructed_robot.activation_thresholds, [0.30, 0.60]
    )
    rear_robot_shell = config.materials[
        "mat-lambert1.001"
    ].estimator.function
    np.testing.assert_allclose(
        rear_robot_shell.activation_thresholds, [0.42, 0.70]
    )
    assert rear_robot_shell.max_coverage == 0.24


def test_fig1_tone_hatches_use_three_brightness_bands() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "llat_feature_lines.json"
    )
    bindings = [config.default, *config.materials.values()]
    hatches = [
        binding.estimator.function
        for binding in bindings
        if isinstance(getattr(binding.estimator, "function", None), ToneHatch)
    ]
    assert hatches
    for hatch in hatches:
        # Two ordered families form exactly three bands: clean highlight,
        # one-family midtone, and two-family shadow cross-hatching.
        assert len(hatch.angles_degrees) == 2
        assert len(hatch.activation_thresholds) == 2
        assert hatch.activation_thresholds[0] < hatch.activation_thresholds[1]


def test_fig1_robot_thigh_panels_keep_their_original_material() -> None:
    scene = (
        Path(__file__).resolve().parents[1]
        / "scenes"
        / "sre_LLaT.xml"
    ).read_text(encoding="utf-8")
    assert scene.count(
        '<ref id="mat-llat-curved-reflector" name="bsdf"/>'
    ) == 0
    assert scene.count(
        '<ref id="mat-Robot_metal_color02_blinn" name="bsdf"/>'
    ) >= 2
    assert 'id="fig1-small-robot-torso"' in scene
    assert scene.count('id="fig1-small-robot-leg-') == 4
    assert '<bsdf type="twosided" id="mat-lambert1.001"' in scene

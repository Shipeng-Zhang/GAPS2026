from __future__ import annotations

from pathlib import Path

import numpy as np

from sre.config import load_config
from sre.feature_lines import (
    FeatureLineConfig,
    FeatureLineType,
    finite_difference_slope,
    minimal_rotation,
    normal_finite_difference_slope,
    parallel_transport_half_vector,
    view_oriented_frame,
)


def test_cuda_auxiliary_trace_batch_is_positive_and_defaults_to_four():
    assert FeatureLineConfig().cuda_auxiliary_batch_size == 4
    assert not FeatureLineConfig().resample_delta_reflections
    assert not FeatureLineConfig().resample_glossy_reflections
    try:
        FeatureLineConfig(cuda_auxiliary_batch_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-sized CUDA auxiliary batches must be rejected")


def test_f11_enables_material_local_delta_reflection_sampling():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    ).feature_lines
    assert config.resample_delta_reflections
    assert config.resample_glossy_reflections


def test_feature_line_material_depth_color_and_displaced_stencil_configuration():
    line = FeatureLineType(
        name="fig11_reflected_displaced",
        include_materials=("mat-f11-pink",),
        color=[1.0, 0.0, 0.4],
        depth_colors={1: [1.0, 0.7, 0.0]},
        stencil_offset=[0.75, -0.25],
    )
    assert line.applies_to_material("mat-f11-pink")
    assert not line.applies_to_material("mat-f11-blue")
    np.testing.assert_allclose(line.color_at(0), [1.0, 0.0, 0.4])
    np.testing.assert_allclose(line.color_at(1), [1.0, 0.7, 0.0])
    np.testing.assert_allclose(line.stencil_offset, [0.75, -0.25])


def test_feature_line_glossy_mix_can_be_overridden_per_dictionary():
    inherited = FeatureLineType(name="inherited")
    yellow_reflection = FeatureLineType(
        name="yellow_reflection",
        glossy_mix_strength=0.0,
    )
    assert inherited.glossy_strength(0.45) == 0.45
    assert yellow_reflection.glossy_strength(0.45) == 0.0
    try:
        FeatureLineType(name="invalid", glossy_mix_strength=1.01)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid per-line glossy strength must be rejected")


def test_feature_line_object_space_hatch_modulates_only_detected_strokes():
    line = FeatureLineType(
        name="hatched_edge",
        line_hatch_scale=1.0,
        line_hatch_width=0.1,
        line_hatch_direction=[1.0, 0.0, 0.0],
        line_hatch_edge_softness=0.0,
    )
    assert line.line_hatch_weight([0.02, 0.0, 0.0]) == 1.0
    assert line.line_hatch_weight([0.25, 0.0, 0.0]) == 0.0
    solid = FeatureLineType(name="solid_edge")
    assert solid.line_hatch_weight([100.0, -50.0, 2.0]) == 1.0


def test_feature_line_shape_filter_is_explicit_and_optional():
    line = FeatureLineType(
        name="face_ring",
        include_shapes=["elm__22", "elm__24"],
    )
    assert line.applies_to_shape("elm__22")
    assert line.applies_to_shape("elm__24")
    assert not line.applies_to_shape("elm__26")
    assert FeatureLineType(name="all_parts").applies_to_shape("anything")


def test_normal_shape_boundary_fallback_retains_auxiliary_shape_ids():
    config = FeatureLineConfig(
        enabled=True,
        types=(FeatureLineType(
            name="coplanar_ring",
            measurement="normal",
            normal_shape_boundary_fallback=True,
            max_depth=1,
        ),),
    )
    assert config.needs_measurement(0, "shape_id")
    assert config.needs_measurement(1, "shape_id")
    assert not config.needs_measurement(2, "shape_id")


def test_silhouette_dictionary_does_not_request_an_internal_metric():
    config = FeatureLineConfig(
        enabled=True,
        types=(FeatureLineType(name="outline", measurement="silhouette"),),
    )
    assert config.types[0].measurement == "silhouette"
    assert not config.needs_measurement(0, "depth", "normal", "curvature")


def test_shifted_stencil_expands_global_auxiliary_sampling_radius():
    config = load_config({
        "feature_lines": {
            "types": [{
                "name": "shifted",
                "stencil_radius": 2.0,
                "stencil_offset": [3.0, 4.0],
            }]
        }
    }).feature_lines
    assert config.max_radius == 7.0


def test_warped_stencil_center_and_bounding_disk_cover_full_displacement():
    line = FeatureLineType(
        name="wavy",
        stencil_radius=1.0,
        stencil_offset=[2.0, 0.0],
        stencil_warp_amplitude=[1.0, 2.0],
        stencil_warp_frequency=[np.pi / 2.0, np.pi / 2.0],
        stencil_warp_phase=[0.0, 0.0],
        stencil_warp_axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    np.testing.assert_allclose(line.stencil_center([1.0, 1.0, 0.0]), [3.0, 2.0])
    np.testing.assert_allclose(line.sampling_radius, 1.0 + np.sqrt(13.0))


def test_f11_magenta_robot_uses_continuous_displaced_stencils():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    ).feature_lines
    magenta = [
        line for line in config.types
        if line.applies_to_material("mat-f11-magenta")
    ]
    assert [line.name for line in magenta] == [
        "magenta_displaced_positive",
        "magenta_displaced_negative",
        "magenta_original_silhouette",
        "magenta_inner_normal",
        "magenta_anchor_normal",
        "magenta_fine_normal",
    ]
    assert all(np.any(line.stencil_warp_amplitude > 0.0) for line in magenta[:2])
    assert all(np.any(line.stencil_warp_amplitude > 0.0) for line in magenta)
    assert all(
        line.stencil_warp_profile == "ripple"
        for line in magenta
    )
    for line in magenta:
        np.testing.assert_allclose(line.stencil_warp_amplitude, [22.0, 16.0])
        np.testing.assert_allclose(line.stencil_warp_frequency, [52.0, 44.0])
    for line in magenta[1:2]:
        np.testing.assert_allclose(
            line.stencil_warp_amplitude,
            magenta[0].stencil_warp_amplitude,
        )
        np.testing.assert_allclose(
            line.stencil_warp_frequency,
            magenta[0].stencil_warp_frequency,
        )
        np.testing.assert_allclose(
            line.stencil_warp_phase,
            magenta[0].stencil_warp_phase,
        )
        np.testing.assert_allclose(
            line.stencil_warp_axes,
            magenta[0].stencil_warp_axes,
        )
    assert magenta[0].stencil_offset[0] > 0.0
    assert magenta[1].stencil_offset[0] < 0.0
    assert magenta[0].stencil_radius == 7.0
    assert magenta[1].stencil_radius == 6.0
    original = next(
        line for line in magenta if line.name == "magenta_original_silhouette"
    )
    assert original.stencil_radius == 5.0
    assert magenta[0].measurement == "normal"
    assert magenta[1].measurement == "normal"
    assert all(line.measurement == "normal" for line in magenta)
    original = next(
        line for line in magenta if line.name == "magenta_original_silhouette"
    )
    assert original.threshold == 0.16
    assert original.normal_orientation_invariant
    assert not original.relative_depth
    assert original.include_silhouette
    assert magenta[0].threshold == 0.20
    assert magenta[1].threshold == 0.18
    inner = next(line for line in magenta if line.name == "magenta_inner_normal")
    assert inner.measurement == "normal"
    assert inner.threshold == 0.24
    assert inner.stencil_radius == 2.2
    assert not inner.include_silhouette
    anchor = next(line for line in magenta if line.name == "magenta_anchor_normal")
    assert anchor.measurement == "normal"
    assert anchor.threshold == 0.22
    assert anchor.stencil_radius == 3.2
    np.testing.assert_allclose(anchor.stencil_offset, [2.15, -0.78])
    assert not anchor.include_silhouette
    assert magenta[0].active_at(1) and magenta[1].active_at(1)


def test_normal_finite_difference_normalizes_and_can_ignore_export_signs():
    np.testing.assert_allclose(
        normal_finite_difference_slope([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], 2.0),
        np.sqrt(2.0) / 2.0,
    )
    assert normal_finite_difference_slope(
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 1.0, True
    ) == 0.0


def test_feature_line_config_reports_smallest_mixture_support():
    config = load_config({
        "feature_lines": {
            "types": [
                {"name": "fine", "stencil_radius": 0.9},
                {"name": "wide", "stencil_radius": 6.0},
            ]
        }
    }).feature_lines
    assert config.min_radius == 0.9
    assert config.max_radius == 6.0


def test_square_stencil_uses_a_circumscribed_global_sampling_disk():
    config = load_config({
        "feature_lines": {
            "types": [{
                "name": "square",
                "stencil": "square",
                "stencil_radius": 3.0,
            }]
        }
    }).feature_lines
    assert config.sampling_stencil == "disk"
    np.testing.assert_allclose(config.max_radius, 3.0 * np.sqrt(2.0))


def test_paper_feature_line_defaults_use_sixteen_searches_and_disk_stencil():
    config = load_config({
        "feature_lines": {
            "types": [{"name": "curvature", "measurement": "normal"}]
        }
    }).feature_lines
    assert config.enabled
    assert config.auxiliary_samples == 16
    assert config.sampling_stencil == "disk"
    assert config.types[0].comparisons == 16


def test_feature_line_config_reports_only_depth_local_measurements():
    config = load_config({
        "feature_lines": {
            "types": [
                {"name": "primary_depth", "measurement": "depth", "max_depth": 0},
                {"name": "reflected_shape", "measurement": "shape_id", "min_depth": 1},
            ]
        }
    }).feature_lines
    assert [line.name for line in config.active_types(0)] == ["primary_depth"]
    assert config.needs_measurement(0, "depth")
    assert not config.needs_measurement(0, "shape_id")
    assert config.needs_measurement(1, "shape_id")


def test_f10_preset_loads_inline_chinese_comment_fields():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f10_lines.json"
    ).feature_lines
    assert config.auxiliary_samples == 16
    assert [line.comparisons for line in config.types] == [16, 16, 16]
    # Stencil radii are intentionally artist-tunable. The regression only
    # verifies that inline annotation fields are ignored and valid values load.
    assert all(line.stencil_radius > 0.0 for line in config.types)
    assert [line.measurement for line in config.types] == [
        "depth", "material_id", "curvature"
    ]


def test_f11_face_nested_seams_have_one_exclusive_subpixel_dictionary():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    ).feature_lines
    for shape_id in ("elm__22", "elm__24"):
        seam_lines = [
            line for line in config.types
            if line.applies_to_material("mat-f11-blue-face")
            and line.applies_to_shape(shape_id)
        ]
        # The lower-right arcs are the boundary between elm__22 and elm__24.
        # Both sides must share one subpixel geometric-normal dictionary only.
        # In particular, neither the other normal dictionaries nor the 3.2 px
        # true head silhouette may enlarge their sampling support and fill the
        # gap.
        assert [line.name for line in seam_lines] == ["blue_face_nested_seams"]
        seam = seam_lines[0]
        assert seam.measurement == "normal"
        assert seam.normal_shape_boundary_fallback
        # The right-hand arc has a much weaker geometric-normal gradient than
        # the left-hand side. Keep a small positive threshold so it closes
        # without classifying a truly planar (zero-gradient) face as a line.
        assert 0.06 <= seam.threshold <= 0.10
        assert 3.8 <= seam.stencil_radius <= 4.6
        assert not seam.include_silhouette
        assert seam.min_depth == 0
        assert seam.max_depth == 1
        assert seam.color_at(1)[1] > seam.color_at(0)[1]


def test_f11_blue_reflection_palette_keeps_internal_lines_at_depth_one():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    ).feature_lines
    by_name = {line.name: line for line in config.types}
    reflected_names = (
        "blue_face_nested_seams",
        "blue_face_right_connector",
        "blue_inner_shape",
        "blue_right_side_shape",
        "blue_shallow_depth",
        "blue_face_recess_and_corners",
        "blue_face_outer_silhouette",
        "blue_body_outer_silhouette",
        "blue_body_fine_normal",
    )
    for name in reflected_names:
        line = by_name[name]
        assert line.active_at(1)
        # The rough floor spreads reflected line energy over many pixels. Its
        # depth-one palette is deliberately brighter and more green/cyan than
        # the direct blue line, matching the Fig. 11 reflected glow.
        assert line.color_at(1)[1] > line.color_at(0)[1]
        assert line.color_at(1)[2] >= line.color_at(0)[2]
    for name in ("blue_face_outer_silhouette", "blue_body_outer_silhouette"):
        outline = by_name[name]
        assert outline.measurement == "normal"
        assert outline.include_silhouette
        # A zero normal threshold would classify every two-hit pair and fill
        # the complete robot with line color instead of drawing an outline.
        assert outline.threshold >= 0.5
        assert 7.5 <= outline.stencil_radius <= 10.5
    assert (
        by_name["blue_face_outer_silhouette"].stencil_radius
        > by_name["blue_body_outer_silhouette"].stencil_radius
    )
    body_detail = by_name["blue_body_fine_normal"]
    assert body_detail.measurement == "normal"
    assert not body_detail.include_silhouette
    assert 0.12 <= body_detail.threshold <= 0.20
    assert 1.0 <= body_detail.stencil_radius <= 1.4

    connector = by_name["blue_face_right_connector"]
    assert connector.measurement == "normal"
    assert connector.include_shapes == ("elm__21",)
    assert connector.threshold == 0.06
    assert connector.stencil_radius == 9.0
    assert not connector.include_silhouette
    assert not connector.normal_shape_boundary_fallback
    connector_lines = [
        line for line in config.types
        if line.applies_to_material("mat-f11-blue-face")
        and line.applies_to_shape("elm__21")
    ]
    assert [line.name for line in connector_lines] == [
        "blue_face_right_connector"
    ]

    panel_detail = by_name["blue_face_recess_and_corners"]
    assert panel_detail.measurement == "normal"
    assert panel_detail.include_shapes == ("elm__20",)
    assert panel_detail.threshold == 0.09
    assert panel_detail.stencil_radius == 2.0
    assert not panel_detail.include_silhouette
    assert not panel_detail.normal_shape_boundary_fallback

    purple = {
        line.name: line for line in config.types
        if line.applies_to_material("mat-f11-purple")
    }
    assert set(purple) == {
        "purple_square_outer", "purple_fine_internal", "purple_fine_curvature"
    }
    assert all(line.measurement == "normal" for line in purple.values())
    assert purple["purple_square_outer"].include_silhouette
    assert purple["purple_square_outer"].stencil_radius == 7.0
    assert purple["purple_fine_internal"].threshold == 0.10
    assert purple["purple_fine_internal"].stencil_radius == 1.35
    assert purple["purple_fine_curvature"].threshold == 0.08
    assert purple["purple_fine_curvature"].stencil_radius == 0.90
    assert not purple["purple_fine_internal"].include_silhouette
    assert not purple["purple_fine_curvature"].include_silhouette

    upper_right = by_name["blue_face_upper_right_detail"]
    assert upper_right.measurement == "normal"
    assert upper_right.include_shapes == ("elm__26",)
    assert upper_right.threshold == 0.045
    assert upper_right.stencil_radius == 3.2
    assert not upper_right.include_silhouette

    lower_body = by_name["blue_lower_body_panel_detail"]
    assert lower_body.measurement == "normal"
    assert lower_body.include_shapes == ("elm__36",)
    assert lower_body.threshold == 0.055
    assert lower_body.stencil_radius == 4.5
    assert lower_body.include_silhouette


def test_f11_fourth_and_fifth_robot_keep_colored_floor_reflections():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    ).feature_lines
    by_name = {line.name: line for line in config.types}

    for name in (
        "pink_direct_yellow_reflection",
        "pink_inner_shape",
        "pink_reflection_normal_fallback",
    ):
        line = by_name[name]
        direct = line.color_at(0)
        reflected = line.color_at(1)
        if line.min_depth == 0:
            assert direct[0] > 8.0 * direct[1]
        else:
            assert line.min_depth == line.max_depth == 1
            assert line.measurement == "normal"
            assert line.include_silhouette
        assert reflected[0] > reflected[1] > 20.0 * reflected[2]
        assert line.glossy_strength(config.glossy_line_strength) == 0.0

    pink_lines = [
        line for line in config.types
        if line.applies_to_material("mat-f11-pink")
    ]
    assert pink_lines
    assert all(line.measurement == "normal" for line in pink_lines)

    for name in ("red_hatched_outline", "red_hatched_internal"):
        line = by_name[name]
        reflected = line.color_at(1)
        assert line.active_at(1)
        assert reflected[0] > 20.0 * reflected[1]
        assert line.glossy_strength(config.glossy_line_strength) <= 0.1


def test_f11_mirror_fill_dictionaries_are_reflection_only():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "f11_lines.json"
    ).feature_lines
    by_name = {line.name: line for line in config.types}
    expected = {
        "pink_reflection_normal_fallback": {"mat-f11-pink"},
    }
    for name, materials in expected.items():
        line = by_name[name]
        assert line.min_depth == line.max_depth == 1
        assert line.measurement == "normal"
        assert line.include_silhouette
        assert set(line.include_materials) == materials
        assert not line.active_at(0)
        assert line.active_at(1)
    # Other robots use only the physical floor reflection; no extra
    # reflection-fill dictionaries are allowed to alter their direct styles.
    for removed in (
        "blue_reflection_silhouette", "blue_reflection_fill",
        "purple_reflection_silhouette", "purple_reflection_fill",
        "magenta_reflection_silhouette", "magenta_reflection_fill",
        "pink_reflection_fill",
    ):
        assert removed not in by_name


def test_parallel_transport_preserves_microfacet_tilt_and_chart_azimuth():
    normal = np.array([0.0, 0.0, 1.0])
    auxiliary_normal = np.array([0.35, 0.1, 0.931396])
    auxiliary_normal /= np.linalg.norm(auxiliary_normal)
    view = np.array([0.2, -0.1, 0.974679])
    view /= np.linalg.norm(view)
    auxiliary_view = np.array([-0.1, 0.25, 0.963068])
    auxiliary_view /= np.linalg.norm(auxiliary_view)
    half_vector = np.array([0.25, 0.35, 0.902774])
    half_vector /= np.linalg.norm(half_vector)

    transported = parallel_transport_half_vector(
        half_vector,
        normal,
        auxiliary_normal,
        view,
        auxiliary_view,
    )
    np.testing.assert_allclose(
        np.dot(transported, auxiliary_normal),
        np.dot(half_vector, normal),
        atol=1e-10,
    )

    def chart_angle(h, n, v):
        cosine = np.dot(h, n)
        tangent = h - cosine * n
        tangent /= np.linalg.norm(tangent)
        first, second = view_oriented_frame(n, v)
        return np.arctan2(np.dot(tangent, second), np.dot(tangent, first))

    np.testing.assert_allclose(
        chart_angle(transported, auxiliary_normal, auxiliary_view),
        chart_angle(half_vector, normal, view),
        atol=1e-10,
    )


def test_diffuse_fallback_minimal_rotation_aligns_normal_directions():
    source = np.array([0.0, 0.0, 1.0])
    target = np.array([0.0, 1.0, 0.0])
    rotated = minimal_rotation(source, source, target)
    np.testing.assert_allclose(rotated, target, atol=1e-10)


def test_finite_difference_is_two_point_lipschitz_lower_bound():
    assert finite_difference_slope(1.0, 3.0, 0.5) == 4.0
    np.testing.assert_allclose(
        finite_difference_slope([0, 0, 0], [3, 4, 0], 2.0), 2.5
    )

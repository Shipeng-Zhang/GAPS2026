from __future__ import annotations

import pytest

from sre.lighting_style import (
    BrightnessLevelControl,
    LightingStyleConfig,
    build_lighting_style,
)


def test_brightness_control_selects_configured_levels() -> None:
    control = BrightnessLevelControl(
        thresholds=(0.2, 0.7), gains=(0.72, 0.95, 1.05)
    )
    assert control.gain(0.1) == 0.72
    assert control.gain(0.2) == 0.95
    assert control.gain(0.69) == 0.95
    assert control.gain(0.7) == 1.05


def test_brightness_control_maps_mean_brightness_to_target_level() -> None:
    control = BrightnessLevelControl(
        thresholds=(0.2, 0.7),
        levels=(0.15, 0.36, 0.62),
        brightness_mode="mean",
    )
    assert control.brightness([0.1, 0.2, 0.3]) == pytest.approx(0.2)
    assert control.gain(0.1) == pytest.approx(1.5)
    assert control.gain(0.4) == pytest.approx(0.9)
    assert control.gain(0.8) == pytest.approx(0.775)


def test_primary_distance_control_uses_smooth_foreground_background_blend() -> None:
    style = LightingStyleConfig(
        enabled=True,
        near_distance=10.0,
        far_distance=20.0,
        near_gain=1.08,
        far_gain=0.72,
    )
    assert style.primary_distance_gain(8.0) == pytest.approx(1.08)
    assert style.primary_distance_gain(15.0) == pytest.approx(0.90)
    assert style.primary_distance_gain(22.0) == pytest.approx(0.72)


def test_lighting_style_builder_ignores_documentation_fields() -> None:
    style = build_lighting_style(
        {
            "_paper_section": "6.3",
            "emission": {"thresholds": [0.25], "gains": [0.9, 0.8]},
            "reflected": {"thresholds": [0.2], "gains": [0.7, 1.0]},
        }
    )
    assert style.enabled
    assert style.emission.thresholds == (0.25,)
    assert style.reflected.gains == (0.7, 1.0)


def test_lighting_style_builder_accepts_target_brightness_levels() -> None:
    style = build_lighting_style(
        {
            "reflected": {
                "thresholds": [0.2, 0.7],
                "levels": [0.15, 0.36, 0.62],
                "brightness_mode": "mean",
            }
        }
    )
    assert style.reflected.uses_target_levels
    assert style.reflected.levels == (0.15, 0.36, 0.62)
    assert style.reflected.brightness_mode == "mean"


def test_lighting_style_rejects_malformed_levels() -> None:
    with pytest.raises(ValueError, match="one more value"):
        BrightnessLevelControl(thresholds=(0.2, 0.7), gains=(1.0, 0.8))
    with pytest.raises(ValueError, match="strictly increasing"):
        BrightnessLevelControl(
            thresholds=(0.7, 0.2), gains=(1.0, 0.9, 0.8)
        )

"""Section 6.3 lighting controls used by the Fig. 1 combined style.

The paper applies small brightness-level functions to emission and reflected
lighting before the line/tone styles consume radiance.  These controls are
kept separate from ToneHatch so the canonical MLS mapping remains responsible
only for transporting image-space coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BrightnessLevelControl:
    """Piecewise-constant gain indexed by luminance level."""

    thresholds: tuple[float, ...] = ()
    gains: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        thresholds = tuple(float(value) for value in self.thresholds)
        gains = tuple(float(value) for value in self.gains)
        if len(gains) != len(thresholds) + 1:
            raise ValueError(
                "lighting brightness gains need one more value than thresholds"
            )
        if (
            any(not np.isfinite(value) or value < 0.0 for value in thresholds)
            or any(b <= a for a, b in zip(thresholds, thresholds[1:]))
        ):
            raise ValueError(
                "lighting brightness thresholds must be finite, non-negative, "
                "and strictly increasing"
            )
        if any(not np.isfinite(value) or value < 0.0 for value in gains):
            raise ValueError("lighting brightness gains must be finite and non-negative")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "gains", gains)

    def gain(self, luminance: float) -> float:
        """Select the configured gain for a brightness band."""
        value = max(float(luminance), 0.0)
        index = int(np.searchsorted(self.thresholds, value, side="right"))
        return self.gains[index]


@dataclass(frozen=True)
class LightingStyleConfig:
    """Transport-component controls matching the combined-style recipe in 6.3."""

    enabled: bool = False
    emission: BrightnessLevelControl = field(default_factory=BrightnessLevelControl)
    reflected: BrightnessLevelControl = field(default_factory=BrightnessLevelControl)
    near_distance: float = 0.0
    far_distance: float = 0.0
    near_gain: float = 1.0
    far_gain: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.near_distance,
            self.far_distance,
            self.near_gain,
            self.far_gain,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("lighting distance controls must be finite and non-negative")
        if self.far_distance and self.far_distance <= self.near_distance:
            raise ValueError("lighting far_distance must exceed near_distance")

    def primary_distance_gain(self, distance: float) -> float:
        if self.far_distance <= self.near_distance:
            return 1.0
        weight = np.clip(
            (float(distance) - self.near_distance)
            / (self.far_distance - self.near_distance),
            0.0,
            1.0,
        )
        weight = weight * weight * (3.0 - 2.0 * weight)
        return float((1.0 - weight) * self.near_gain + weight * self.far_gain)


def _control(spec: Mapping[str, object] | None) -> BrightnessLevelControl:
    values = dict(spec or {})
    values["thresholds"] = tuple(values.get("thresholds", ()))
    values["gains"] = tuple(values.get("gains", (1.0,)))
    return BrightnessLevelControl(**values)


def build_lighting_style(spec: Mapping[str, object] | None) -> LightingStyleConfig:
    if not spec:
        return LightingStyleConfig()
    values = {
        key: value for key, value in dict(spec).items()
        if not str(key).startswith("_")
    }
    values["emission"] = _control(values.get("emission"))
    values["reflected"] = _control(values.get("reflected"))
    values.setdefault("enabled", True)
    return LightingStyleConfig(**values)


__all__ = [
    "BrightnessLevelControl",
    "LightingStyleConfig",
    "build_lighting_style",
]

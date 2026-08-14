"""JSON configuration and material/depth parametrization for SRE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    from .estimators import Estimator, IdentityEstimator, build_estimator
    from .feature_lines import FeatureLineConfig, build_feature_lines
    from .lighting_style import LightingStyleConfig, build_lighting_style
    from .styles import StyleContext
    from .tone_mapping import ToneMappingConfig, build_tone_mapping
except ImportError:
    from estimators import Estimator, IdentityEstimator, build_estimator
    from feature_lines import FeatureLineConfig, build_feature_lines
    from lighting_style import LightingStyleConfig, build_lighting_style
    from styles import StyleContext
    from tone_mapping import ToneMappingConfig, build_tone_mapping

# 风格断言:根据光线跟踪或渲染时的“上下文条件”,判断当前是否应该应用某种特定的渲染风格。
@dataclass(frozen=True)
class StylePredicate:
    min_depth: int = 0 # 允许应用风格的最小递归深度
    max_depth: int | None = None # 允许应用风格的最大递归深度
    depths: tuple[int, ...] | None = None # 指定特定深度
    first_hit: bool = False # 是否仅在光线第一次击中物体时才生效
    max_occurrences: int | None = None # 允许最大击中次数

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any] | None) -> "StylePredicate":
        values = {
            key: value for key, value in dict(spec or {}).items()
            if not str(key).startswith("_")
        }
        if "depths" in values:
            values["depths"] = tuple(int(value) for value in values["depths"])
        return cls(**values)

    # 匹配逻辑
    def matches(self, context: StyleContext) -> bool:
        if context.depth < self.min_depth:
            return False
        if self.max_depth is not None and context.depth > self.max_depth:
            return False
        if self.depths is not None and context.depth not in self.depths:
            return False
        if self.first_hit and context.occurrence != 1:
            return False
        if self.max_occurrences is not None and context.occurrence > self.max_occurrences:
            return False
        return True


@dataclass
class MaterialStyle:
    estimator: Estimator # 估计器
    predicate: StylePredicate = field(default_factory=StylePredicate) # 风格判断

    # 选择估计器
    def select(self, context: StyleContext) -> Estimator:
        return self.estimator if self.predicate.matches(context) else IdentityEstimator()

# 风格化渲染配置
@dataclass
class SREConfig:
    materials: dict[str, MaterialStyle] = field(default_factory=dict)
    shapes: dict[str, MaterialStyle] = field(default_factory=dict)
    default: MaterialStyle = field(
        default_factory=lambda: MaterialStyle(IdentityEstimator())
    )
    feature_lines: FeatureLineConfig = field(default_factory=FeatureLineConfig)
    tone_mapping: ToneMappingConfig = field(default_factory=ToneMappingConfig)
    lighting_style: LightingStyleConfig = field(default_factory=LightingStyleConfig)

    def resolve(self, material_id: str, shape_id: str, context: StyleContext) -> Estimator:
        binding = self.shapes.get(shape_id, self.materials.get(material_id, self.default))
        return binding.select(context)

# 解析配置字典,并封装为MaterialStyle对象
def _binding(spec: Mapping[str, Any] | None) -> MaterialStyle:
    values = {
        key: value for key, value in dict(spec or {}).items()
        if not str(key).startswith("_")
    }
    predicate = StylePredicate.from_spec(values.pop("when", None)) # 弹出筛选条件
    return MaterialStyle(build_estimator(values), predicate) # 组装成MaterialStyle对象


def _deep_merge_config(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively overlay a small scene preset on a complete SRE config.

    Lists intentionally replace their parent value.  In particular, feature
    line dictionaries are ordered and cannot be merged safely by index.  This
    keeps the inherited Fig. 13 DoF/glossy preset compact without introducing
    a second, drifting copy of every tone-hatch material.
    """
    merged = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        inherited = merged.get(key)
        if isinstance(inherited, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_config(inherited, value)
        else:
            merged[key] = value
    return merged


def load_config_data(
    source: str | Path, _stack: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """Load JSON plus an optional relative ``extends`` chain.

    ``extends`` is deliberately a file-only convenience: runtime dictionaries
    remain self-contained and retain their previous validation semantics.
    """
    path = Path(source).resolve()
    if path in _stack:
        cycle = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Cyclic SRE config inheritance: {cycle}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"SRE config root must be an object: {path}")
    data = dict(data)
    parent_name = data.get("extends")
    if parent_name is None:
        return data
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ValueError("SRE config 'extends' must be a non-empty path string")
    parent_path = (path.parent / parent_name).resolve()
    parent = load_config_data(parent_path, (*_stack, path))
    return _deep_merge_config(parent, data)


def load_config(source: str | Path | Mapping[str, Any] | None) -> SREConfig:
    if source is None:
        data: dict[str, Any] = {}
    elif isinstance(source, Mapping):
        data = dict(source)
    else:
        data = load_config_data(source)
    unknown = set(data) - {
        "default", "materials", "shapes", "metadata", "feature_lines",
        "tone_mapping", "lighting_style",
    }
    if unknown:
        raise ValueError(f"Unknown SRE config fields: {sorted(unknown)}")
    return SREConfig(
        default=_binding(data.get("default")),
        materials={key: _binding(value) for key, value in data.get("materials", {}).items()},
        shapes={key: _binding(value) for key, value in data.get("shapes", {}).items()},
        feature_lines=build_feature_lines(data.get("feature_lines")),
        tone_mapping=build_tone_mapping(data.get("tone_mapping")),
        lighting_style=build_lighting_style(data.get("lighting_style")),
    )

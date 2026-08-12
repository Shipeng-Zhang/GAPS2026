#!/usr/bin/env python3
"""Run renderer-independent tests without requiring pytest."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import tests.test_estimators as estimator_tests
    import tests.test_feature_lines as feature_line_tests
    import tests.test_render_helpers as render_tests
    import tests.test_styles_config as style_tests
    import tests.test_tone_mapping as tone_tests

    count = 0
    for module in (
        estimator_tests, feature_line_tests, render_tests, style_tests,
        tone_tests,
    ):
        for name in sorted(value for value in dir(module) if value.startswith("test_")):
            print(f"{module.__name__}.{name}")
            getattr(module, name)()
            count += 1
    print(f"{count} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

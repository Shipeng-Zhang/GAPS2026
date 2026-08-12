from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("SRE_MITSUBA_TEST") != "1", reason="opt-in renderer smoke test")
def test_sre_integrator_renders_non_black_image():
    import mitsuba as mi

    variant = os.environ.get("SRE_VARIANT", "scalar_rgb")
    mi.set_variant(variant)
    if variant == "cuda_ad_rgb":
        from sre.cuda_integrator import register_cuda_integrator
        register_cuda_integrator()
        integrator = "sre_cuda"
    else:
        from sre.integrator import register_sre_integrator
        register_sre_integrator()
        integrator = "sre"
    root = Path(__file__).resolve().parents[1]
    scene = mi.load_file(
        str(root / "scenes/sre_cbox.xml"),
        integrator=integrator,
        style_config=str(root / "configs/cel_1.json"),
        spp="1",
        res="8",
        max_depth="3",
    )
    image = np.asarray(mi.render(scene, spp=1, seed=2))
    assert image.shape[:2] == (8, 8)
    assert np.all(np.isfinite(image[..., :3]))
    assert np.max(image[..., :3]) > 0.0

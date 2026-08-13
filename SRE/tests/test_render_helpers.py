from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sre.render import (
    DEFAULT_FEATURE_LINE_MAX_WAVEFRONT_SIZE,
    DEFAULT_MAX_WAVEFRONT_SIZE,
    DEFAULT_SRE_MAX_WAVEFRONT_SIZE,
    DEFAULT_TERMINAL_TONE_MAX_WAVEFRONT_SIZE,
    _feature_lines_require_shape_identity,
    _load_single_worker_job,
    _mean_squared_error,
    _parser,
    _recommended_cuda_wavefront,
    _recommended_spp_per_pass,
    _sample_passes,
    _split_horizontal,
    _split_streaming_tiles,
    _stage_windows_replacement_character_assets,
    _tile_pixel_budget,
    _worker_environment,
)


def test_windows_replacement_character_assets_use_temporary_ascii_aliases():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_directory = root / "scene"
        staging_directory = root / "staging"
        scene_directory.mkdir()
        staging_directory.mkdir()
        source = scene_directory / "broken_\ufffd.ply"
        source.write_bytes(b"ply-data")
        scene_path = scene_directory / "scene.xml"
        xml = ET.fromstring(
            '<scene><shape><string name="filename" '
            'value="broken_\ufffd.ply"/></shape></scene>'
        )

        count = _stage_windows_replacement_character_assets(
            xml, scene_path, staging_directory
        )

        staged = Path(xml.find(".//string").get("value", ""))
        assert count == 1
        assert staged.name == "asset_000.ply"
        assert staged.read_bytes() == b"ply-data"
        assert source.read_bytes() == b"ply-data"


def test_shape_filtered_feature_lines_disable_xml_mesh_merging():
    assert _feature_lines_require_shape_identity({
        "shapes": {"mesh-specific-part": {"estimator": "constant"}},
        "feature_lines": {"types": []},
    })
    assert _feature_lines_require_shape_identity({
        "feature_lines": {
            "types": [{
                "measurement": "normal",
                "include_shapes": ["elm__20", "elm__22"],
            }]
        }
    })
    assert _feature_lines_require_shape_identity({
        "feature_lines": {
            "types": [{
                "measurement": "normal",
                "normal_shape_boundary_fallback": True,
            }]
        }
    })
    assert _feature_lines_require_shape_identity({
        "feature_lines": {"types": [{"measurement": "shape_id"}]}
    })
    assert not _feature_lines_require_shape_identity({
        "feature_lines": {"types": [{"measurement": "normal"}]}
    })


def test_cli_defaults_keep_memory_bounded_and_are_positive():
    args = _parser().parse_args([])
    # The interactive defaults may be tuned independently, but a pass must
    # never contain more samples than the complete render and both budgets
    # must remain valid.
    assert args.spp == 32
    assert args.spp_per_pass == 1
    # Frozen pass replay makes one-sample graphs fast without constructing a
    # giant multi-SPP trace. Keep enough lanes to feed the GPU while bounding
    # nested feature/tone/SRE graph assembly.
    # ``main`` resolves this after reading the selected style configuration.
    assert args.max_wavefront_size is None
    assert DEFAULT_MAX_WAVEFRONT_SIZE > 0
    assert DEFAULT_TERMINAL_TONE_MAX_WAVEFRONT_SIZE > 0
    assert args.disable_jit_freezing is False


def test_auto_wavefront_matches_style_complexity():
    root = Path(__file__).resolve().parents[1]
    assert _recommended_cuda_wavefront(
        root / "configs" / "f13_tone.json"
    ) == DEFAULT_TERMINAL_TONE_MAX_WAVEFRONT_SIZE
    assert _recommended_cuda_wavefront(
        root / "configs" / "f10_lines.json"
    ) == DEFAULT_FEATURE_LINE_MAX_WAVEFRONT_SIZE
    assert _recommended_cuda_wavefront(
        root / "configs" / "identity.json"
    ) == DEFAULT_SRE_MAX_WAVEFRONT_SIZE
    assert _recommended_cuda_wavefront(
        root / "configs" / "llat_feature_lines.json"
    ) == 4096
    assert _recommended_spp_per_pass(
        root / "configs" / "f11_lines.json", 256
    ) == 32
    assert _recommended_spp_per_pass(
        root / "configs" / "f13_tone.json", 256
    ) == 16


def test_cli_can_force_single_sample_low_memory_passes():
    args = _parser().parse_args(["--spp", "32", "--spp-per-pass", "1"])
    assert args.spp == 32
    assert args.spp_per_pass == 1


def test_horizontal_crops_cover_the_image_once():
    crops = _split_horizontal(23, 17, 4)
    assert crops == [
        (0, 0, 23, 5),
        (0, 5, 23, 4),
        (0, 9, 23, 4),
        (0, 13, 23, 4),
    ]
    assert sum(width * height for _, _, width, height in crops) == 23 * 17


def test_horizontal_crops_do_not_create_empty_workers():
    crops = _split_horizontal(5, 2, 8)
    assert crops == [(0, 0, 5, 1), (0, 1, 5, 1)]


def test_streaming_tiles_cover_image_once_and_respect_pixel_budget():
    crops = _split_streaming_tiles(23, 17, 50)
    coverage = np.zeros((17, 23), dtype=np.int32)
    for x, y, width, height in crops:
        assert width * height <= 50
        coverage[y:y + height, x:x + width] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))


def test_streaming_tiles_balance_full_width_rows_across_gpus():
    crops = _split_streaming_tiles(1920, 640, 262_144, worker_count=4)
    assert len(crops) == 8
    assert all(width * height <= 262_144 for _, _, width, height in crops)
    assert [len(crops[index::4]) for index in range(4)] == [2, 2, 2, 2]


def test_primary_lane_budget_accounts_for_spp_per_pass():
    assert _tile_pixel_budget(1_048_576, 16) == 65_536
    assert _tile_pixel_budget(8, 32) == 1


def test_worker_environment_preserves_overrides_and_selects_device():
    with tempfile.TemporaryDirectory() as directory:
        environment = _worker_environment(
            "4",
            {
                "LD_LIBRARY_PATH": "/custom/runtime",
                "DRJIT_LIBLLVM_PATH": "/explicit/libLLVM.so",
                "SRE_CACHE_ROOT": directory,
            },
        )
        assert environment["CUDA_VISIBLE_DEVICES"] == "4"
        assert environment["DRJIT_LIBLLVM_PATH"] == "/explicit/libLLVM.so"
        assert "/custom/runtime" in environment["LD_LIBRARY_PATH"].split(":")
        assert environment["OPTIX_CACHE_PATH"].endswith("device-4")
        assert environment["CUDA_CACHE_PATH"].endswith("device-4")
        assert Path(environment["OPTIX_CACHE_PATH"]).is_dir()


def test_worker_environment_isolates_physical_gpu_caches():
    with tempfile.TemporaryDirectory() as directory:
        base = {"SRE_CACHE_ROOT": directory}
        first = _worker_environment("3", base)
        second = _worker_environment("7", base)
        assert first["OPTIX_CACHE_PATH"] != second["OPTIX_CACHE_PATH"]
        assert first["CUDA_CACHE_PATH"] != second["CUDA_CACHE_PATH"]


def test_fresh_worker_job_rejects_multiple_tiles():
    with tempfile.TemporaryDirectory() as directory:
        job_path = Path(directory) / "job.json"
        job_path.write_text(
            '[{"index": 0, "crop": [0, 0, 1, 1], '
            '"seed": 1, "tile_data": "tile.npy"}]',
            encoding="utf-8",
        )
        try:
            _load_single_worker_job(job_path)
        except ValueError as error:
            assert "exactly one tile" in str(error)
        else:
            raise AssertionError(
                "a CUDA worker must never accept multiple tiles"
            )


def test_sample_passes_include_a_short_final_pass():
    assert list(_sample_passes(5, 2, 7)) == [
        (2, 7),
        (2, 2654435776),
        (1, 1013904249),
    ]


def test_sample_passes_clamp_seed_and_reject_invalid_counts():
    assert list(_sample_passes(2, 8, 0xFFFFFFFF)) == [
        (2, 0xFFFFFFFF)
    ]
    for spp, spp_per_pass in ((0, 1), (1, 0)):
        try:
            list(_sample_passes(spp, spp_per_pass, 0))
        except ValueError:
            continue
        raise AssertionError("invalid sample counts must raise ValueError")


def test_mean_squared_error_uses_all_rgb_pixels():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    reference = np.zeros_like(image)
    image[0, 0, 0] = 1.0
    assert _mean_squared_error(image, reference) == 1.0 / 12.0


def test_mean_squared_error_rejects_shape_mismatch_and_nonfinite_values():
    valid = np.zeros((2, 2, 3), dtype=np.float32)
    invalid_cases = [
        np.zeros((1, 2, 3), dtype=np.float32),
        np.zeros((2, 2, 4), dtype=np.float32),
        np.full((2, 2, 3), np.nan, dtype=np.float32),
    ]
    for invalid in invalid_cases:
        try:
            _mean_squared_error(valid, invalid)
        except ValueError:
            continue
        raise AssertionError("invalid MSE input should raise ValueError")

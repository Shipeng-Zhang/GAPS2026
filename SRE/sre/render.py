from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from math import gcd
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


CHANNEL_NAMES = [
    "R",
    "G",
    "B",
    "sre_tree_nodes",
    "sre_style_evaluations",
    "sre_inner_variance",
    "sre_estimated_bias",
]

# Fig. 13's nested tone/expectation tree has much more per-lane state than a
# feature-line-only render. These are *primary lane* budgets, not bytes.
# Two-lane packing bounds the deepest Fig. 13 tree at eight times the primary
# width. 196k therefore stays below 16 GiB on the tested 24 GiB cards while
# reducing a 1080x786 three-GPU render to six jobs. Users with less than 16 GiB can
# explicitly select 65536 without changing samples or image semantics.
DEFAULT_MAX_WAVEFRONT_SIZE = 8294400
# First-hit-only tone has no exponential style subtree, but reflective Fig.13
# materials still allocate BSDF and frozen-OptiX state during the first pass.
# 424,440 primary lanes can force a forbidden allocation-cache flush while a
# frozen function is recording. 262k produces four balanced 212,220-lane tiles
# at 1080x786: measured at 76 s on two 24 GiB cards without changing samples.
DEFAULT_TERMINAL_TONE_MAX_WAVEFRONT_SIZE = 1036800
DEFAULT_SRE_MAX_WAVEFRONT_SIZE = 8294400
# Fig.11 throughput preset for a 24 GiB card: one 1980x1440 tile with 32
# primary samples in flight. The feature-line tracer prunes camera misses and
# stores compact auxiliary fields, so this remains far below the recursive
# tone/SRE memory cost while halving pass-level launch overhead.
DEFAULT_FEATURE_LINE_MAX_WAVEFRONT_SIZE = 8294400


def _sample_passes(
    spp: int, spp_per_pass: int, seed: int
) -> Iterator[tuple[int, int]]:
    """Yield bounded-memory outer passes as ``(sample_count, seed)``."""
    if spp < 1 or spp_per_pass < 1:
        raise ValueError("spp and spp_per_pass must be positive")
    rendered = 0
    pass_index = 0
    while rendered < spp:
        count = min(spp_per_pass, spp - rendered)
        pass_seed = (seed + pass_index * 0x9E3779B9) & 0xFFFFFFFF
        yield count, pass_seed
        rendered += count
        pass_index += 1


def _feature_lines_require_shape_identity(config_data: Mapping[str, Any]) -> bool:
    """Return whether XML optimization would invalidate line dictionaries."""
    feature_lines = config_data.get("feature_lines", {})
    if not isinstance(feature_lines, Mapping):
        return False
    line_types = feature_lines.get("types", ())
    if not isinstance(line_types, Sequence) or isinstance(line_types, (str, bytes)):
        return False
    return any(
        str(line.get("measurement", "")).lower() in {"shape", "shape_id"}
        or bool(line.get("include_shapes"))
        or bool(line.get("normal_shape_boundary_fallback", False))
        for line in line_types
        if isinstance(line, Mapping)
    )


def _load_scene(
    scene_path: Path,
    config_path: Path,
    integrator_name: str,
    spp: int,
    resolution: int,
    width: int,
    height: int,
    max_depth: int,
    crop: tuple[int, int, int, int] | None,
) -> Any:
    import mitsuba as mi

    # Mitsuba 3's optimized XML parser merges compatible PLY meshes that share
    # a BSDF. That is normally desirable, but destroys per-part ShapePtr
    # identities. They are consumed not only by the ``shape_id`` metric, but
    # also by ``include_shapes`` filters and the opt-in coplanar seam fallback.
    # In Fig. 11, robot 1's two face rings use both mechanisms. Looking only
    # for a shape_id metric caused those rings to disappear when robot 4 was
    # converted to geometric-normal finite differences.
    preserve_shape_ids = False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        preserve_shape_ids = _feature_lines_require_shape_identity(config_data)
    except (OSError, ValueError, TypeError, AttributeError):
        # Configuration loading in the integrator remains authoritative and
        # will report malformed files with its usual, more specific error.
        preserve_shape_ids = False

    substitutions = {
        "integrator": integrator_name,
        "style_config": str(config_path),
        "spp": str(spp),
        "res": str(resolution),
        "max_depth": str(max_depth),
    }
    if crop is None and width == resolution and height == resolution:
        return mi.load_file(
            str(scene_path), optimize=not preserve_shape_ids, **substitutions
        )

    # Film dimensions and crop parameters cannot be changed after scene
    # construction. Patch them in memory so arbitrary project scenes support
    # rectangular output and multi-GPU tiles without new XML defaults.
    root = ET.fromstring(scene_path.read_text(encoding="utf-8"))
    films = list(root.iter("film"))
    if not films:
        raise ValueError(f"Scene has no film to configure: {scene_path}")

    film_values = {"width": width, "height": height}
    if crop is not None:
        x, y, crop_width, crop_height = crop
        film_values.update({
            "crop_offset_x": x,
            "crop_offset_y": y,
            "crop_width": crop_width,
            "crop_height": crop_height,
        })
    for film in films:
        found = set()
        for child in list(film):
            name = child.get("name")
            if child.tag == "integer" and name in film_values:
                child.set("value", str(int(film_values[name])))
                found.add(name)
        for name, value in film_values.items():
            if name not in found:
                ET.SubElement(
                    film, "integer", name=name, value=str(int(value))
                )

    resolver = mi.file_resolver()
    resolver.prepend(str(scene_path.parent))
    xml = ET.tostring(root, encoding="unicode")
    if "$res" not in xml:
        substitutions.pop("res")
    return mi.load_string(
        xml, optimize=not preserve_shape_ids, **substitutions
    )


def _write_pixels(output_path: Path, pixels: np.ndarray) -> Path:
    import mitsuba as mi

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.ascontiguousarray(pixels[..., :3])
    if output_path.suffix.lower() == ".exr":
        names = CHANNEL_NAMES[: pixels.shape[-1]]
        mi.Bitmap(
            np.ascontiguousarray(pixels), channel_names=names
        ).write(str(output_path))
        preview = output_path.with_suffix(".png")
        mi.util.write_bitmap(str(preview), rgb)
        mi.Thread.wait_for_tasks()
        return preview
    mi.util.write_bitmap(str(output_path), rgb, write_async=False)
    return output_path


def _mean_squared_error(image: np.ndarray, reference: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if image.shape != reference.shape:
        raise ValueError(
            "MSE image shape mismatch: "
            f"rendered {image.shape}, reference {reference.shape}"
        )
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"MSE expects RGB images with shape (height, width, 3), got {image.shape}"
        )
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(reference)):
        raise ValueError("MSE images must contain only finite RGB values")
    return float(np.mean(np.square(image - reference), dtype=np.float64))


def _load_display_rgb(path: Path) -> np.ndarray:
    """Load a PNG preview and decode its sRGB values to linear RGB."""
    import mitsuba as mi

    if not path.is_file():
        raise FileNotFoundError(f"MSE image does not exist: {path}")
    bitmap = mi.Bitmap(str(path)).convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.Float32,
        srgb_gamma=False,
    )
    return np.array(bitmap, dtype=np.float32, copy=True)


def _preview_mse(preview: Path, reference: Path) -> float:
    return _mean_squared_error(
        _load_display_rgb(preview), _load_display_rgb(reference)
    )


def _config_mse_reference(
    config_path: Path, override: Path | None
) -> Path | None:
    if override is not None:
        return override.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("mse_reference")
    if value is None:
        return None
    reference = Path(str(value))
    if not reference.is_absolute():
        reference = config_path.parent / reference
    return reference.resolve()


def _requires_bounded_cuda_wavefront(config_path: Path) -> bool:
    """Whether disabling spatial streaming is unsafe for this style config."""
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if bool(data.get("feature_lines", {}).get("enabled", False)):
        return True
    if bool(data.get("tone_mapping", {}).get("enabled", False)):
        return True
    bindings = [data.get("default", {}), *data.get("materials", {}).values(),
                *data.get("shapes", {}).values()]
    return any(
        isinstance(binding, Mapping)
        and int(binding.get("samples", 1)) > 1
        for binding in bindings
    )


def _recommended_cuda_wavefront(config_path: Path) -> int:
    """Select a throughput-oriented budget without unbounding recursive DAGs."""
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    bindings = [
        data.get("default", {}),
        *data.get("materials", {}).values(),
        *data.get("shapes", {}).values(),
    ]
    if bool(data.get("tone_mapping", {}).get("enabled", False)):
        styled = [
            binding for binding in bindings
            if isinstance(binding, Mapping)
            and str(binding.get("estimator", "identity")).lower()
            not in {"identity", "constant"}
        ]
        terminal_direct = bool(styled) and all(
            str(binding.get("estimator", "identity")).lower() == "direct"
            and not bool(binding.get("recursive", True))
            for binding in styled
        )
        return (
            DEFAULT_TERMINAL_TONE_MAX_WAVEFRONT_SIZE
            if terminal_direct else DEFAULT_MAX_WAVEFRONT_SIZE
        )

    feature_lines = bool(
        data.get("feature_lines", {}).get("enabled", False)
    )
    recursive_estimator = any(_binding_is_recursive_heavy(binding) for binding in bindings)
    if feature_lines and not recursive_estimator:
        return DEFAULT_FEATURE_LINE_MAX_WAVEFRONT_SIZE
    return DEFAULT_SRE_MAX_WAVEFRONT_SIZE


def _recommended_spp_per_pass(config_path: Path, spp: int) -> int:
    """Select a throughput pass width without applying it to recursive tone."""
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    bindings = [
        data.get("default", {}),
        *data.get("materials", {}).values(),
        *data.get("shapes", {}).values(),
    ]
    feature_only = (
        bool(data.get("feature_lines", {}).get("enabled", False))
        and not bool(data.get("tone_mapping", {}).get("enabled", False))
        and not any(_binding_is_recursive_heavy(binding) for binding in bindings)
    )
    return min(int(spp), 32 if feature_only else 16)


def _binding_is_recursive_heavy(binding: Any) -> bool:
    """Identify estimators that can fan out a large recursive expectation.

    A one-sample direct style (used by Fig. 11's red hatch material) still
    traces one physical suffix, but it does not create the branching workload
    of polynomial/power-series estimators. Treating it as heavy forced the
    feature-line render onto a 16k lane budget and left most of a 24 GiB GPU
    idle.
    """
    if not isinstance(binding, Mapping):
        return False
    estimator = str(binding.get("estimator", "identity")).lower()
    if estimator in {"identity", "constant"}:
        return False
    if estimator == "direct":
        if not bool(binding.get("recursive", True)):
            return False
        return int(binding.get("samples", 1)) > 1
    return True


def _image_statistics(pixels: np.ndarray) -> dict[str, Any]:
    rgb = pixels[..., :3]
    finite = np.isfinite(rgb)
    return {
        "shape": list(pixels.shape),
        "finite_fraction": float(finite.mean()),
        "mean_rgb": [
            float(value) for value in np.nanmean(rgb, axis=(0, 1))
        ],
        "max_rgb": [
            float(value) for value in np.nanmax(rgb, axis=(0, 1))
        ],
        "aov_means": {
            "tree_nodes": float(np.nanmean(pixels[..., 3]))
            if pixels.shape[-1] > 3 else 0.0,
            "style_evaluations": float(np.nanmean(pixels[..., 4]))
            if pixels.shape[-1] > 4 else 0.0,
            "inner_variance": float(np.nanmean(pixels[..., 5]))
            if pixels.shape[-1] > 5 else 0.0,
            "estimated_bias": float(np.nanmean(pixels[..., 6]))
            if pixels.shape[-1] > 6 else 0.0,
        },
    }


def render_scene(
    scene_path: Path | str,
    config_path: Path | str,
    output_path: Path | str,
    spp: int = 16,
    resolution: int = 128,
    max_depth: int = 5,
    seed: int = 0,
    crop: tuple[int, int, int, int] | None = None,
    tile_data: Path | None = None,
    width: int | None = None,
    height: int | None = None,
    spp_per_pass: int = 1,
    mse_reference: Path | str | None = None,
    progress_label: str | None = None,
    jit_freezing: bool = True,
) -> dict[str, Any]:
    import mitsuba as mi

    if mi.variant() == "cuda_ad_rgb":
        try:
            from .cuda_integrator import register_cuda_integrator
        except ImportError:
            from cuda_integrator import register_cuda_integrator
        register_cuda_integrator()
        integrator_name = "sre_cuda"
    else:
        try:
            from .integrator import register_sre_integrator
        except ImportError:
            from integrator import register_sre_integrator
        register_sre_integrator()
        integrator_name = "sre"

    scene_path = Path(scene_path).resolve()
    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    # Dynamic DFS lane compaction intentionally reads compacted widths on the
    # host. Such seed-dependent control flow cannot be recorded by dr.freeze.
    # Preserve it for every nested SRE/feature/tone configuration; graph replay
    # is enabled only for simple fixed-topology identity/path renders.
    jit_freezing = bool(
        jit_freezing
        and mi.variant() == "cuda_ad_rgb"
        and not _requires_bounded_cuda_wavefront(config_path)
    )
    image_width = int(width if width is not None else resolution)
    image_height = int(height if height is not None else resolution)
    spp_per_pass = min(int(spp_per_pass), int(spp))
    if spp_per_pass < 1:
        raise ValueError("spp_per_pass must be positive")
    scene = _load_scene(
        scene_path,
        config_path,
        integrator_name,
        int(spp),
        int(resolution),
        image_width,
        image_height,
        int(max_depth),
        crop,
    )
    if jit_freezing:
        # Thin-lens cameras retain an internal aperture-warp state that is not
        # fully traversable by dr.freeze. Recording an identity/path pass then
        # captures a device variable created before recording and aborts on
        # replay. Nested SRE configurations already disable freezing above;
        # apply the same safe fallback to thin-lens identity diagnostics.
        sensor_parameters = mi.traverse(scene.sensors()[0])
        if "aperture_radius" in sensor_parameters:
            jit_freezing = False
    start = time.perf_counter()
    accumulated = None
    accumulated_device = None
    accumulated_shape: tuple[int, ...] | None = None
    pass_index = 0
    pass_seconds = []
    film = scene.sensors()[0].film()
    frozen_renderers: dict[int, Any] = {}
    profile_kernels = (
        mi.variant() == "cuda_ad_rgb"
        and os.environ.get("SRE_PROFILE_KERNELS") == "1"
    )
    if profile_kernels:
        import drjit as dr

        dr.kernel_history_clear()
        dr.set_flag(dr.JitFlag.KernelHistory, True)

    def render_pass(pass_spp: int, pass_seed: int) -> Any:
        """Render a pass while replaying an already traced CUDA graph.

        ``mi.render()`` wraps the integrator in an AD custom operation and
        retraces the complete Python/Dr.Jit graph on every call. This renderer
        never differentiates an image, so it can call the primal integrator
        directly and freeze it. The random seed remains an opaque runtime
        input: samples change, while the graph and compiled kernels are reused.
        """
        if mi.variant() != "cuda_ad_rgb" or not jit_freezing:
            return mi.render(scene, spp=pass_spp, seed=pass_seed)

        import drjit as dr

        frozen = frozen_renderers.get(pass_spp)
        if frozen is None:
            sensor = scene.sensors()[0]
            integrator = scene.integrator()
            @dr.freeze(
                state_fn=lambda seed: (scene, sensor, integrator),
                backend=dr.JitBackend.CUDA,
                limit=4,
                warn_after=3,
            )
            def frozen(seed: Any) -> Any:
                return integrator.render(
                    scene=scene,
                    sensor=sensor,
                    seed=seed,
                    spp=pass_spp,
                    develop=True,
                    evaluate=False,
                )

            frozen_renderers[pass_spp] = frozen
        opaque_seed = dr.opaque(mi.UInt32, int(pass_seed), 1)
        return frozen(opaque_seed)

    for pass_spp, pass_seed in _sample_passes(
        int(spp), spp_per_pass, int(seed)
    ):
        if progress_label is not None:
            print(
                f"[SRE worker] {progress_label} pass={pass_index + 1}/"
                f"{(int(spp) + spp_per_pass - 1) // spp_per_pass} "
                f"spp={pass_spp} seed={pass_seed} start",
                flush=True,
            )
        film.clear()
        pass_start = time.perf_counter()
        image = render_pass(pass_spp, pass_seed)
        if mi.variant() == "cuda_ad_rgb":
            import drjit as dr

            if accumulated_device is None:
                accumulated_shape = tuple(int(value) for value in image.shape)
                accumulated_device = dr.zeros(
                    mi.Float64, dr.width(image.array)
                )
            accumulated_device += mi.Float64(image.array) * pass_spp
            # Complete the image read before film.clear() starts the next pass.
            # This retains only one Float64 accumulation buffer on the device
            # and avoids a full PCIe transfer for every outer sample.
            dr.eval(accumulated_device)
            dr.sync_thread()
            pass_pixels = None
        else:
            pass_pixels = np.array(image, dtype=np.float32, copy=True)
        pass_seconds.append(time.perf_counter() - pass_start)
        if pass_pixels is not None:
            if accumulated is None:
                accumulated = np.zeros(pass_pixels.shape, dtype=np.float64)
            accumulated += pass_pixels * pass_spp
        pass_index += 1
        if progress_label is not None:
            print(
                f"[SRE worker] {progress_label} pass={pass_index}/"
                f"{(int(spp) + spp_per_pass - 1) // spp_per_pass} "
                f"seconds={pass_seconds[-1]:.3f} done",
                flush=True,
            )
        del image, pass_pixels
    elapsed = time.perf_counter() - start
    if profile_kernels:
        import drjit as dr

        history = dr.kernel_history()
        dr.set_flag(dr.JitFlag.KernelHistory, False)
        jit_entries = [
            entry for entry in history
            if entry.get("type") == dr.KernelType.JIT
        ]
        print(
            "[SRE profile] "
            f"kernels={len(history)} jit={len(jit_entries)} "
            f"optix={sum(bool(entry.get('uses_optix', False)) for entry in jit_entries)} "
            f"execution_ms={sum(float(entry.get('execution_time', 0.0)) for entry in history):.1f} "
            f"codegen_ms={sum(float(entry.get('codegen_time', 0.0)) for entry in jit_entries):.1f} "
            f"backend_ms={sum(float(entry.get('backend_time', 0.0)) for entry in jit_entries):.1f} "
            f"optix_backend_ms={sum(float(entry.get('backend_time', 0.0)) for entry in jit_entries if entry.get('uses_optix', False)):.1f} "
            f"cuda_backend_ms={sum(float(entry.get('backend_time', 0.0)) for entry in jit_entries if not entry.get('uses_optix', False)):.1f} "
            f"unique_hashes={len({entry.get('hash') for entry in jit_entries})} "
            f"disk_hits={sum(bool(entry.get('cache_disk', False)) for entry in jit_entries)}",
            flush=True,
        )
    if accumulated_device is not None:
        assert accumulated_shape is not None
        pixels = (
            np.array(accumulated_device, dtype=np.float64, copy=True)
            .reshape(accumulated_shape)
            / int(spp)
        ).astype(np.float32)
    else:
        pixels = (accumulated / int(spp)).astype(np.float32)

    if tile_data is not None:
        tile_data.parent.mkdir(parents=True, exist_ok=True)
        np.save(tile_data, pixels, allow_pickle=False)
        preview = output_path
    else:
        preview = _write_pixels(output_path, pixels)

    report = {
        "scene": str(scene_path),
        "config": str(config_path),
        "output": str(output_path),
        "preview": str(preview),
        "spp": int(spp),
        "spp_per_pass": spp_per_pass,
        "passes": pass_index,
        "pass_seconds": pass_seconds,
        "resolution": (
            image_width if image_width == image_height
            else [image_width, image_height]
        ),
        "width": image_width,
        "height": image_height,
        "max_depth": int(max_depth),
        "seed": int(seed),
        "variant": mi.variant(),
        "seconds": elapsed,
        "jit_freezing": bool(
            jit_freezing and mi.variant() == "cuda_ad_rgb"
        ),
        "jit_recordings": int(sum(
            getattr(renderer, "n_recordings", 0)
            for renderer in frozen_renderers.values()
        )),
        **_image_statistics(pixels),
    }
    if mse_reference is not None and tile_data is None:
        reference_path = Path(mse_reference).resolve()
        report["mse"] = _preview_mse(Path(preview), reference_path)
        report["mse_reference"] = str(reference_path)
        report["mse_color_space"] = "linear_rgb_decoded_from_srgb_png"
    if crop is not None:
        report["crop"] = list(crop)
    return report


def _visible_cuda_devices() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not value or value.strip() in {"-1", "NoDevFiles"}:
        return []
    return [device.strip() for device in value.split(",") if device.strip()]


def _split_horizontal(
    width: int, height: int, count: int
) -> list[tuple[int, int, int, int]]:
    count = min(count, height)
    base, remainder = divmod(height, count)
    crops = []
    y = 0
    for index in range(count):
        height = base + (1 if index < remainder else 0)
        crops.append((0, y, width, height))
        y += height
    return crops


def _tile_pixel_budget(max_wavefront_size: int, spp_per_pass: int) -> int:
    """Convert a primary-lane budget into a spatial pixel budget.

    Mitsuba expands every pixel by ``spp_per_pass`` before invoking the
    integrator. Keeping this product bounded makes peak CUDA and Dr.Jit host
    memory independent of the complete output resolution.
    """
    if max_wavefront_size < 1 or spp_per_pass < 1:
        raise ValueError("wavefront size and spp_per_pass must be positive")
    return max(1, max_wavefront_size // spp_per_pass)


def _split_streaming_tiles(
    width: int, height: int, max_pixels: int, worker_count: int = 1
) -> list[tuple[int, int, int, int]]:
    """Cover an image with bounded crops balanced across GPU slots."""
    if width < 1 or height < 1 or max_pixels < 1 or worker_count < 1:
        raise ValueError("tile dimensions and pixel budget must be positive")
    # Search a small 2-D grid instead of always making full-width strips. At
    # the low lane budgets required by nested SRE expectations, full-width
    # strips degenerate into hundreds of one-row scenes. The selected grid
    # minimizes tile count, then prefers compact (near-square) crops, while
    # keeping the number of jobs divisible by the worker count when possible.
    candidates = []
    for columns in range(1, min(width, max_pixels) + 1):
        widest = (width + columns - 1) // columns
        if widest > max_pixels:
            continue
        maximum_height = max(1, max_pixels // widest)
        rows = (height + maximum_height - 1) // maximum_height
        row_multiple = worker_count // gcd(columns, worker_count)
        rows = min(
            height,
            ((rows + row_multiple - 1) // row_multiple) * row_multiple,
        )
        tallest = (height + rows - 1) // rows
        tile_count = columns * rows
        aspect_penalty = abs(widest / tallest - 1.0)
        candidates.append((tile_count, aspect_penalty, columns, rows))
    if not candidates:
        raise ValueError("pixel budget is too small for a non-empty tile")
    _, _, columns, rows = min(candidates)

    x_edges = [(index * width) // columns for index in range(columns + 1)]
    y_edges = [(index * height) // rows for index in range(rows + 1)]
    crops = []
    for row in range(rows):
        y = y_edges[row]
        crop_height = y_edges[row + 1] - y
        for column in range(columns):
            x = x_edges[column]
            crop_width = x_edges[column + 1] - x
            crops.append((x, y, crop_width, crop_height))
    return crops


def _stop_workers(processes: Sequence[subprocess.Popen[Any]]) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()
    deadline = time.monotonic() + 5.0
    while running and time.monotonic() < deadline:
        running = [process for process in running if process.poll() is None]
        if running:
            time.sleep(0.05)
    for process in running:
        process.kill()
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _worker_command(
    args: argparse.Namespace,
    job_path: Path,
    worker_report: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--scene", str(args.scene.resolve()),
        "--config", str(args.config.resolve()),
        "--output", str(args.output.resolve()),
        "--spp", str(args.spp),
        "--spp-per-pass", str(args.spp_per_pass),
        "--resolution", str(args.resolution),
        "--width", str(args.width),
        "--height", str(args.height),
        "--max-depth", str(args.max_depth),
        "--seed", str(args.seed),
        "--variant", args.variant,
        "--_worker",
        "--_worker-job", str(job_path),
        "--_worker-report", str(worker_report),
    ]
    # Workers only produce tiles. The parent computes MSE once after assembly.
    command.append("--skip-mse")
    if args.drjit_debug:
        command.append("--drjit-debug")
    if args.worker_progress:
        command.append("--worker-progress")
    if args.disable_jit_freezing:
        command.append("--disable-jit-freezing")
    return command


def _worker_environment(
    device: str | None,
    base_environment: Mapping[str, str] | None = None,
    cuda_launch_blocking: bool = False,
) -> dict[str, str]:
    """Build an isolated GPU-worker environment for the active Conda prefix.

    Dr.Jit loads LLVM dynamically instead of consulting ``llvm-config``. On
    this system, an activated Conda environment still omits its ``lib``
    directory from ``LD_LIBRARY_PATH``, causing Dr.Jit to find the system LLVM
    12 even though LLVM 15 is installed in the environment. GPU rendering does
    not need the LLVM backend, but selecting the matching runtime avoids a
    misleading warning and also supplies LLVM 15's required Conda libstdc++.
    """
    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
    if cuda_launch_blocking:
        # Must be present before the worker imports/initializes CUDA.
        environment["CUDA_LAUNCH_BLOCKING"] = "1"

    # OptiX normally puts one SQLite/WAL cache in $HOME/.drjit. Concurrent
    # workers then serialize on the same writer lock. Give every physical GPU
    # a stable private cache: fresh processes on one device reuse it, while
    # different devices no longer contend with one another.
    cache_root = Path(environment.get(
        "SRE_CACHE_ROOT",
        str(Path(__file__).resolve().parent.parent / ".sre-cache"),
    )).resolve()
    device_label = "default" if device is None else "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(device)
    )
    optix_cache = cache_root / "optix" / f"device-{device_label}"
    cuda_cache = cache_root / "cuda" / f"device-{device_label}"
    optix_cache.mkdir(parents=True, exist_ok=True)
    cuda_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("OPTIX_CACHE_PATH", str(optix_cache))
    environment.setdefault("OPTIX_CACHE_MAXSIZE", str(2 * 1024**3))
    environment.setdefault("CUDA_CACHE_PATH", str(cuda_cache))
    environment.setdefault("CUDA_CACHE_MAXSIZE", str(1024**3))

    prefix_library = Path(sys.prefix) / "lib"
    llvm_library = prefix_library / "libLLVM.so"
    if llvm_library.is_file():
        environment.setdefault("DRJIT_LIBLLVM_PATH", str(llvm_library))
        prefix_text = str(prefix_library)
        current = environment.get("LD_LIBRARY_PATH", "")
        entries = [entry for entry in current.split(os.pathsep) if entry]
        if prefix_text not in entries:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                [prefix_text, *entries]
            )
    return environment


def _load_single_worker_job(path: Path) -> dict[str, Any]:
    """Load one tile job; arrays are rejected to forbid context reuse."""
    job = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(job, Mapping):
        raise ValueError("A fresh worker accepts exactly one tile job")
    required = {"index", "crop", "seed", "tile_data"}
    missing = required.difference(job)
    if missing:
        raise ValueError(
            "Tile job is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return dict(job)


def _render_multi_gpu(
    args: argparse.Namespace, devices: Sequence[str | None]
) -> dict[str, Any]:
    """Render every bounded tile in a fresh process and CUDA context.

    A Mitsuba scene owns C++/OptiX objects that can indirectly retain Dr.Jit
    arrays after Python references and allocator caches are cleared. Reusing a
    process for another tile therefore permits stale graph state to contaminate
    a later launch. One process per tile makes process exit the hard resource
    lifetime boundary; GPU slots are reused, CUDA contexts are not.
    """
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_spp_per_pass = min(args.spp_per_pass, args.spp)
    if args.max_wavefront_size:
        lane_limited_pixels = _tile_pixel_budget(
            args.max_wavefront_size, effective_spp_per_pass
        )
        balanced_pixels = (
            args.width * args.height + len(devices) - 1
        ) // len(devices)
        max_pixels = min(lane_limited_pixels, balanced_pixels)
        crops = _split_streaming_tiles(
            args.width, args.height, max_pixels, len(devices)
        )
    else:
        crops = _split_horizontal(args.width, args.height, len(devices))
        max_pixels = max(width * height for _, _, width, height in crops)
    actual_max_tile_pixels = max(
        width * height for _, _, width, height in crops
    )
    actual_max_primary_lanes = (
        actual_max_tile_pixels * effective_spp_per_pass
    )
    if (
        args.max_wavefront_size
        and actual_max_primary_lanes > args.max_wavefront_size
    ):
        raise RuntimeError(
            "Streaming scheduler exceeded its primary-lane budget: "
            f"{actual_max_primary_lanes} > {args.max_wavefront_size}"
        )
    if args.worker_progress:
        print(
            "[SRE scheduler] "
            f"devices={list(devices)} tiles={len(crops)} "
            f"lane_budget={args.max_wavefront_size} "
            f"max_primary_lanes={actual_max_primary_lanes}",
            flush=True,
        )
    devices = list(devices[: min(len(devices), len(crops))])
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="sre-multigpu-") as directory:
        temporary = Path(directory)
        active_workers: dict[int, dict[str, Any]] = {}
        device_summaries = [
            {
                "device": device if device is not None else "default",
                "tiles": 0,
                "seconds": 0.0,
            }
            for device in devices
        ]
        next_tile = 0

        def launch_tile(slot: int, tile_index: int) -> None:
            """Launch exactly one tile in a fresh CUDA process/context."""
            device = devices[slot]
            crop = crops[tile_index]
            worker_report = temporary / f"tile-{tile_index}-report.json"
            log_path = temporary / f"tile-{tile_index}.log"
            job_path = temporary / f"tile-{tile_index}-job.json"
            job_path.write_text(json.dumps({
                "index": tile_index,
                "crop": list(crop),
                "seed": (
                    args.seed + tile_index * 0x9E3779B9
                ) & 0xFFFFFFFF,
                "tile_data": str(temporary / f"tile-{tile_index}.npy"),
            }), encoding="utf-8")
            log = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    _worker_command(args, job_path, worker_report),
                    env=_worker_environment(
                        device,
                        cuda_launch_blocking=args.cuda_launch_blocking,
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            active_workers[slot] = {
                "device": device,
                "tile_index": tile_index,
                "crop": crop,
                "process": process,
                "report": worker_report,
                "log_path": log_path,
                "log": log,
            }
            if args.worker_progress:
                print(
                    f"[SRE scheduler] tile={tile_index} device={device!r} "
                    f"pid={process.pid} fresh-context start",
                    flush=True,
                )

        try:
            for slot in range(len(devices)):
                launch_tile(slot, next_tile)
                next_tile += 1

            failure: tuple[int, int, dict[str, Any]] | None = None
            while active_workers and failure is None:
                made_progress = False
                for slot, record in list(active_workers.items()):
                    code = record["process"].poll()
                    if code is None:
                        continue
                    made_progress = True
                    record["log"].close()
                    del active_workers[slot]
                    if code != 0:
                        failure = (slot, code, record)
                        break
                    tile_report = json.loads(
                        record["report"].read_text(encoding="utf-8")
                    )
                    if not isinstance(tile_report, Mapping):
                        raise RuntimeError(
                            "Fresh tile worker returned a non-object report"
                        )
                    device_summaries[slot]["tiles"] += 1
                    device_summaries[slot]["seconds"] += float(
                        tile_report["seconds"]
                    )
                    if args.worker_progress:
                        print(
                            f"[SRE scheduler] tile={record['tile_index']} "
                            f"device={record['device']!r} "
                            f"pid={record['process'].pid} context-destroyed done",
                            flush=True,
                        )
                    # The completed process has exited, which destroys its
                    # entire CUDA context and all scene-owned C++ resources.
                    # Only now may this physical GPU receive another tile.
                    if next_tile < len(crops):
                        launch_tile(slot, next_tile)
                        next_tile += 1
                if active_workers and failure is None and not made_progress:
                    time.sleep(0.05)

            if failure is not None:
                _stop_workers([
                    record["process"] for record in active_workers.values()
                ])
                slot, code, record = failure
                log_text = record["log_path"].read_text(
                    encoding="utf-8", errors="replace"
                )[-8000:]
                advice = (
                    "If the exit code is -9, lower --max-wavefront-size "
                    "(for example 1536); Linux likely killed the worker "
                    "after host-memory exhaustion."
                )
                if code == -6:
                    advice = (
                        "The first failing tile/pass is printed above when "
                        "--worker-progress is enabled. If the log contains "
                        "jit_flush_malloc_cache(), first retry with "
                        "--spp-per-pass 1 --max-wavefront-size 1536. To "
                        "locate a genuine out-of-bounds operation, use one "
                        "visible GPU, --max-wavefront-size 512, "
                        "--cuda-launch-blocking, and --drjit-debug."
                    )
                raise RuntimeError(
                    f"GPU worker {slot} on visible device "
                    f"{record['device']!r}, tile {record['tile_index']} "
                    f"exited with code {code}.\n"
                    f"{log_text}\n"
                    f"{advice}"
                )
        except BaseException:
            _stop_workers([
                record["process"] for record in active_workers.values()
            ])
            raise
        finally:
            for record in active_workers.values():
                if not record["log"].closed:
                    record["log"].close()

        first_tile = np.load(temporary / "tile-0.npy", allow_pickle=False)
        channel_count = first_tile.shape[-1]
        del first_tile
        pixels = np.empty(
            (args.height, args.width, channel_count), dtype=np.float32
        )
        for tile_index, crop in enumerate(crops):
            tile = np.load(
                temporary / f"tile-{tile_index}.npy", allow_pickle=False
            )
            x, y, width, height = crop
            expected = (height, width, channel_count)
            if tile.shape != expected:
                raise RuntimeError(
                    f"Tile {tile_index} returned {tile.shape}, expected {expected}"
                )
            pixels[y:y + height, x:x + width] = tile
            del tile
        workers = device_summaries

    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    preview = _write_pixels(output_path, pixels)
    elapsed = time.perf_counter() - started
    report = {
        "scene": str(args.scene.resolve()),
        "config": str(args.config.resolve()),
        "output": str(output_path),
        "preview": str(preview),
        "spp": args.spp,
        "spp_per_pass": effective_spp_per_pass,
        "passes": (args.spp + effective_spp_per_pass - 1)
        // effective_spp_per_pass,
        "resolution": (
            args.width if args.width == args.height
            else [args.width, args.height]
        ),
        "width": args.width,
        "height": args.height,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "variant": args.variant,
        "seconds": elapsed,
        "multi_gpu": len(devices) > 1,
        "devices": [
            device if device is not None else "default" for device in devices
        ],
        "tile_count": len(crops),
        "tile_pixel_limit": max_pixels,
        "max_tile_pixels": actual_max_tile_pixels,
        "max_primary_wavefront": (
            actual_max_tile_pixels * effective_spp_per_pass
        ),
        "workers": workers,
        **_image_statistics(pixels),
    }
    if args.mse_reference is not None:
        report["mse"] = _preview_mse(preview, args.mse_reference)
        report["mse_reference"] = str(args.mse_reference)
        report["mse_color_space"] = "linear_rgb_decoded_from_srgb_png"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("./scenes/sre_LLaT.xml"))
    parser.add_argument("--config", type=Path, default=Path("./configs/llat_feature_lines.json"))
    parser.add_argument("--output", type=Path, default=Path("./outputs/fig1.png"))
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument(
        "--spp-per-pass", type=int, default=1,
        help=(
            "Outer samples per CUDA graph. By default feature-line-only "
            "renders use 32 for throughput, while recursive tone/SRE uses 16."
        ),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    # Fig. 13 contains camera -> planar mirror -> curved reflector -> scene
    # paths. A depth of two terminates at the curved reflector, producing the
    # solid black disk and removing shaded robot reflections even at high SPP.
    # Keep the CLI default consistent with the scene/paper configuration.
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-wavefront-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--variant",choices=("cuda_ad_rgb", "scalar_rgb"),default="cuda_ad_rgb",)
    parser.add_argument(
        "--mse-reference",
        type=Path,
        default=None,
        help=(
            "Reference PNG for image-level MSE; defaults to metadata."
            "mse_reference in the style config"
        ),
    )
    parser.add_argument(
        "--skip-mse",
        action="store_true",
        help="Disable config-provided MSE evaluation for this render",
    )
    parser.add_argument(
        "--drjit-debug",
        action="store_true",
        help=(
            "Enable Dr.Jit's bounds/undefined-behavior checks. Use only for "
            "small single-GPU diagnostic renders because it is very slow"
        ),
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help=(
            "Synchronize CUDA launches so an illegal access is reported at "
            "the first failing operation instead of a later allocator call"
        ),
    )
    parser.add_argument(
        "--worker-progress",
        action="store_true",
        help="Print worker tile/pass boundaries to identify the failing job",
    )
    parser.add_argument(
        "--disable-jit-freezing",
        action="store_true",
        help=(
            "Disable CUDA graph recording/replay and retrace every pass; "
            "intended only for diagnostics"
        ),
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_crop", type=int, nargs=4, help=argparse.SUPPRESS)
    parser.add_argument("--_tile-data", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-report", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Configure the parent before its first Mitsuba/Dr.Jit import as well as
    # subprocesses. This prevents final image assembly from discovering the
    # system LLVM 12 after all CUDA workers have successfully exited.
    runtime_environment = _worker_environment(None)
    for name in ("DRJIT_LIBLLVM_PATH", "LD_LIBRARY_PATH"):
        if name in runtime_environment:
            os.environ[name] = runtime_environment[name]
    if args.cuda_launch_blocking:
        # This also covers direct internal-worker invocations. No Mitsuba or
        # Dr.Jit module has been imported by render.py at this point.
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args.scene = args.scene.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.mse_reference = (
        None
        if args.skip_mse
        else _config_mse_reference(args.config, args.mse_reference)
    )
    if args.mse_reference is not None and not args.mse_reference.is_file():
        raise FileNotFoundError(
            f"MSE reference image does not exist: {args.mse_reference}"
        )
    args.width = args.width if args.width is not None else args.resolution
    args.height = args.height if args.height is not None else args.resolution
    if args.spp_per_pass is None:
        args.spp_per_pass = _recommended_spp_per_pass(args.config, args.spp)
        if args.worker_progress:
            print(
                "[SRE scheduler] auto spp-per-pass "
                f"{args.spp_per_pass} for {args.config.name}",
                flush=True,
            )
    recommended_wavefront = _recommended_cuda_wavefront(args.config)
    if args.max_wavefront_size is None:
        args.max_wavefront_size = recommended_wavefront
        if args.worker_progress:
            print(
                "[SRE scheduler] auto wavefront "
                f"{args.max_wavefront_size} for {args.config.name}",
                flush=True,
            )
    if (
        args.variant == "cuda_ad_rgb"
        and args.max_wavefront_size == 0
        and _requires_bounded_cuda_wavefront(args.config)
    ):
        # Unbounded horizontal crops are useful for identity/path-tracing
        # diagnostics, but are never safe for feature/tone/nested SRE. Make a
        # mistaken zero fail safe instead of silently creating 100k+ lanes.
        args.max_wavefront_size = recommended_wavefront
        if args.worker_progress:
            print(
                "[SRE scheduler] unbounded wavefront requested for a nested "
                f"style; applying safe limit {recommended_wavefront}",
                flush=True,
            )
    if (
        args.spp < 1
        or args.spp_per_pass < 1
        or args.resolution < 1
        or args.width < 1
        or args.height < 1
        or args.max_depth < 1
        or args.max_wavefront_size < 0
    ):
        raise ValueError(
            "spp, spp-per-pass, resolution, width, height, and max-depth "
            "must be positive (max-wavefront-size may be zero)"
        )

    devices = _visible_cuda_devices()
    if args.variant == "cuda_ad_rgb" and not args._worker:
        scheduling_devices: list[str | None] = devices or [None]
        report = _render_multi_gpu(args, scheduling_devices)
    else:
        if args._worker and args._worker_job is not None:
            import mitsuba as mi

            if args._worker_report is None:
                raise ValueError("Streaming worker needs a report path")
            mi.set_variant(args.variant)
            if args.drjit_debug:
                import drjit as dr
                dr.set_flag(dr.JitFlag.Debug, True)
            job = _load_single_worker_job(args._worker_job)
            crop = tuple(int(value) for value in job["crop"])
            label = (
                f"pid={os.getpid()} tile={int(job['index'])} crop={crop} "
                f"primary_lanes<={crop[2] * crop[3] * min(args.spp_per_pass, args.spp)}"
            )
            if args.worker_progress:
                print(f"[SRE worker] {label} load", flush=True)
            report = render_scene(
                args.scene,
                args.config,
                args.output,
                args.spp,
                args.resolution,
                args.max_depth,
                int(job["seed"]),
                crop=crop,
                tile_data=Path(job["tile_data"]),
                width=args.width,
                height=args.height,
                spp_per_pass=args.spp_per_pass,
                mse_reference=None,
                progress_label=label if args.worker_progress else None,
                jit_freezing=not args.disable_jit_freezing,
            )
            args._worker_report.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            if args.worker_progress:
                print(f"[SRE worker] {label} process-exit", flush=True)
            return 0
        if args._worker and (
            args._crop is None
            or args._tile_data is None
            or args._worker_report is None
        ):
            raise ValueError("Incomplete internal GPU worker arguments")
        import mitsuba as mi
        mi.set_variant(args.variant)
        if args.drjit_debug:
            import drjit as dr
            dr.set_flag(dr.JitFlag.Debug, True)
        crop = tuple(args._crop) if args._crop is not None else None
        report = render_scene(
            args.scene,
            args.config,
            args.output,
            args.spp,
            args.resolution,
            args.max_depth,
            args.seed,
            crop=crop,
            tile_data=args._tile_data,
            width=args.width,
            height=args.height,
            spp_per_pass=args.spp_per_pass,
            mse_reference=args.mse_reference,
            progress_label="direct-worker" if args.worker_progress else None,
            jit_freezing=not args.disable_jit_freezing,
        )
        if args._worker:
            args._worker_report.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return 0

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

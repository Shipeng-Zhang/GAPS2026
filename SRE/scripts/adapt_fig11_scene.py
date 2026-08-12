"""Adapt the Blender-exported Fig. 11 XML to the local SRE renderer.

The source asset shares one Blender material between all five robots.  That is
fine for ordinary rendering, but feature-line dictionaries are material-bound,
so the shared material would make the five styles leak into each other.  This
script groups every exported submesh by robot and assigns one stable SRE BSDF
per robot while preserving the original shape IDs.
"""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "fig11.xml"

FLOOR_MATERIALS = {
    "mat-Material.020",
    "mat-Material.021",
    "mat-Material.022",
    "mat-Material.029",
    "mat-Material.030",
}

ROBOT_MATERIALS = {
    "blue": {
        "mat-Material.031", "mat-Material.032", "mat-Material.035",
        "mat-Material.041", "mat-Material.045", "mat-Material.046",
        "mat-Material.052", "default-bsdf",
    },
    "purple": {f"mat-Material.{index:03d}" for index in range(53, 59)},
    "magenta": {
        *(f"mat-Material.{index:03d}" for index in range(62, 70)),
        "mat-Material.087",
        "mat-Иллюстрация_без_названия (4)",
    },
    "pink": {f"mat-Material.{index:03d}" for index in range(72, 77)},
    "red": {f"mat-Material.{index:03d}" for index in range(78, 86)},
}

SRE_MATERIAL = {
    name: f"mat-f11-{name}" for name in ROBOT_MATERIALS
}


def object_key(shape: ET.Element) -> str:
    filename = shape.find("string[@name='filename']")
    if filename is None:
        return ""
    stem = Path(filename.get("value", "")).stem
    # Blender emits one PLY per material slot, e.g.
    # ``Cube_004-Material.050.ply``.  All slots of that object must share the
    # same owner, especially the ubiquitous Material.050 fragments.
    return re.sub(r"-(?:Material\.\d+|Иллюстрация.*)$", "", stem)


def material_ref(shape: ET.Element) -> ET.Element | None:
    return shape.find("ref[@name='bsdf']")


def make_bsdf(material_id: str, reflectance: float) -> ET.Element:
    bsdf = ET.Element("bsdf", type="diffuse", id=material_id, name=material_id)
    ET.SubElement(
        bsdf,
        "rgb",
        name="reflectance",
        value=f"{reflectance:.3f}",
    )
    return bsdf


def adapt() -> None:
    tree = ET.parse(SCENE)
    root = tree.getroot()
    if (
        root.find("integrator") is not None
        and root.find("integrator").get("type") == "$integrator"
        and root.find("bsdf[@id='mat-f11-blue']") is not None
    ):
        print(f"{SCENE} is already adapted; leaving it unchanged")
        return
    # Keep the exporter's 2.1 compatibility mode. Mitsuba's 3.0 XML upgrade
    # path may coalesce adjacent meshes that share a BSDF, which would erase
    # the per-part shape IDs required by the inner-line dictionary.
    root.set("version", "2.1.0")

    # Replace Blender's fixed path integrator with the substitutions expected
    # by sre/render.py.  A two-edge path is sufficient for direct lines plus
    # the once-reflected lines visible in the glossy floor.
    for child in list(root):
        if child.tag in {"default", "integrator"}:
            root.remove(child)
    defaults = [
        ("spp", "256"),
        ("res", "640"),
        ("max_depth", "2"),
        ("integrator", "sre_cuda"),
        ("style_config", "../configs/f11_lines.json"),
    ]
    insertion = 0
    for name, value in defaults:
        root.insert(insertion, ET.Element("default", name=name, value=value))
        insertion += 1
    integrator = ET.Element("integrator", type="$integrator", id="fig11-sre")
    ET.SubElement(integrator, "integer", name="max_depth", value="$max_depth")
    ET.SubElement(integrator, "integer", name="rr_depth", value="4")
    ET.SubElement(integrator, "float", name="rr_probability", value="0.95")
    ET.SubElement(
        integrator, "string", name="style_config", value="$style_config"
    )
    root.insert(insertion, integrator)

    sensor = root.find("sensor")
    if sensor is None:
        raise RuntimeError("fig11.xml has no sensor")
    # Preserve the user's rebuilt camera and framing.  The supplied production
    # meshes are centred at x=-4,-2,0,2,4; the previous synthetic-scene camera
    # targeted x=4 and consequently cropped/distorted this asset set.
    sampler = sensor.find("sampler")
    if sampler is None:
        sampler = ET.SubElement(sensor, "sampler", type="independent")
    sample_count = sampler.find("integer[@name='sample_count']")
    if sample_count is None:
        sample_count = ET.SubElement(sampler, "integer", name="sample_count")
    sample_count.set("value", "$spp")
    film = sensor.find("film")
    if film is None:
        raise RuntimeError("fig11.xml has no film")
    for name in ("width", "height"):
        field = film.find(f"integer[@name='{name}']")
        if field is None:
            field = ET.SubElement(film, "integer", name=name)
        field.set("value", "$res")
    if film.find("rfilter") is None:
        ET.SubElement(film, "rfilter", type="box")
    if film.find("string[@name='pixel_format']") is None:
        ET.SubElement(film, "string", name="pixel_format", value="rgb")
    if film.find("string[@name='component_format']") is None:
        ET.SubElement(
            film, "string", name="component_format", value="float32"
        )

    # Remove the directional light and the five separated floor tiles. Lines
    # are SRE radiance values and remain visible/reflected without illumination.
    for emitter in list(root.findall("emitter")):
        root.remove(emitter)
    shapes = list(root.findall("shape"))
    for shape in shapes:
        ref = material_ref(shape)
        if ref is not None and ref.get("id") in FLOOR_MATERIALS:
            root.remove(shape)

    # First infer each Blender object group's owner from its non-shared
    # material. Then use that mapping for the common Material.050 fragments.
    owner_by_key: dict[str, str] = {}
    for shape in root.findall("shape"):
        ref = material_ref(shape)
        if ref is None:
            continue
        for owner, material_ids in ROBOT_MATERIALS.items():
            if ref.get("id") in material_ids:
                owner_by_key[object_key(shape)] = owner
                break

    unresolved_common: list[ET.Element] = []
    for shape in root.findall("shape"):
        filename = shape.find("string[@name='filename']")
        ref = material_ref(shape)
        if filename is None or ref is None:
            continue
        filename.set(
            "value",
            f"fig11_meshes/meshes/{Path(filename.get('value', '')).name}",
        )
        owner = None
        for candidate, material_ids in ROBOT_MATERIALS.items():
            if ref.get("id") in material_ids:
                owner = candidate
                break
        if owner is None and ref.get("id") == "mat-Material.050":
            owner = owner_by_key.get(object_key(shape))
            if owner is None:
                unresolved_common.append(shape)
                continue
        if owner is None:
            raise RuntimeError(
                f"cannot classify {shape.get('id')} with {ref.get('id')}"
            )
        ref.set("id", SRE_MATERIAL[owner])

    # The final ten paired unicode-named pieces belong to the centre/magenta
    # robot but lack the '_Material_' naming convention used by the exporter.
    for shape in unresolved_common:
        ref = material_ref(shape)
        assert ref is not None
        ref.set("id", SRE_MATERIAL["magenta"])

    # All legacy BSDFs are now unused. Add compact, uniquely-valued definitions
    # so Mitsuba cannot alias the five material IDs during scene construction.
    for bsdf in list(root.findall("bsdf")):
        root.remove(bsdf)
    bsdfs = [
        make_bsdf("mat-f11-blue", 0.781),
        make_bsdf("mat-f11-purple", 0.773),
        make_bsdf("mat-f11-magenta", 0.765),
        make_bsdf("mat-f11-pink", 0.757),
        make_bsdf("mat-f11-red", 0.741),
    ]
    floor = ET.Element(
        "bsdf", type="roughconductor", id="mat-f11-floor",
        name="mat-f11-floor",
    )
    ET.SubElement(floor, "string", name="distribution", value="ggx")
    # The paper's reflections remain identifiable while being softened by
    # glossy transport.  0.04 avoids the featureless colour fog produced by
    # the old 0.12 setting at practical SPP values. The final 0.025 value keeps
    # a controlled glossy halo while preserving recognizable reflected parts.
    ET.SubElement(floor, "float", name="alpha", value="0.025")
    ET.SubElement(floor, "string", name="material", value="none")
    ET.SubElement(
        floor, "rgb", name="specular_reflectance", value="0.96"
    )
    bsdfs.append(floor)
    sensor_index = list(root).index(sensor)
    for offset, bsdf in enumerate(bsdfs):
        root.insert(sensor_index + 1 + offset, bsdf)

    floor_shape = ET.Element(
        "shape", type="rectangle", id="fig11-reflective-floor",
        name="fig11-reflective-floor",
    )
    floor_transform = ET.SubElement(floor_shape, "transform", name="to_world")
    ET.SubElement(floor_transform, "scale", x="5.6", y="4.0")
    ET.SubElement(floor_transform, "rotate", x="1", angle="-90")
    ET.SubElement(
        floor_transform, "translate", x="0.0", y="-0.01", z="0.4"
    )
    ET.SubElement(floor_shape, "ref", id="mat-f11-floor", name="bsdf")
    first_shape = next(
        (index for index, child in enumerate(root) if child.tag == "shape"),
        len(root),
    )
    root.insert(first_shape, floor_shape)

    ET.indent(tree, space="    ")
    tree.write(SCENE, encoding="unicode", xml_declaration=False)


if __name__ == "__main__":
    adapt()

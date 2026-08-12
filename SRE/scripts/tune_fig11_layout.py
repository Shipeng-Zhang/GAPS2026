"""Apply the paper-matched Fig. 11 object scale and reflection roughness."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "fig11.xml"

LAYOUT = {
    "mat-f11-blue": (-4.0, 1.35),
    "mat-f11-purple": (-2.0, 1.35),
    "mat-f11-magenta": (0.0, 1.38),
    "mat-f11-pink": (2.0, 1.35),
    "mat-f11-red": (4.0, 1.32),
}


def main() -> None:
    tree = ET.parse(SCENE)
    root = tree.getroot()
    floor_alpha = root.find("bsdf[@id='mat-f11-floor']/float[@name='alpha']")
    if floor_alpha is None:
        raise RuntimeError("fig11.xml has no mat-f11-floor roughness")
    floor_alpha.set("value", "0.025")

    changed = 0
    for shape in root.findall("shape"):
        ref = shape.find("ref[@name='bsdf']")
        if ref is None or ref.get("id") not in LAYOUT:
            continue
        if shape.find("transform[@name='to_world']") is not None:
            continue
        center_x, scale = LAYOUT[ref.get("id")]
        translate_x = center_x * (1.0 - scale)
        transform = ET.Element("transform", name="to_world")
        ET.SubElement(
            transform,
            "matrix",
            value=(
                f"{scale} 0 0 {translate_x} "
                f"0 {scale} 0 0 "
                f"0 0 {scale} 0 "
                "0 0 0 1"
            ),
        )
        filename = shape.find("string[@name='filename']")
        insertion = list(shape).index(filename) + 1 if filename is not None else 0
        shape.insert(insertion, transform)
        changed += 1

    if changed not in {0, 289}:
        raise RuntimeError(
            f"partially transformed Fig. 11 scene ({changed}/289 meshes)"
        )
    ET.indent(tree, space="    ")
    tree.write(SCENE, encoding="unicode", xml_declaration=False)
    print(
        "Fig. 11 layout already tuned"
        if changed == 0
        else f"Scaled {changed} robot meshes and set floor alpha to 0.025"
    )


if __name__ == "__main__":
    main()

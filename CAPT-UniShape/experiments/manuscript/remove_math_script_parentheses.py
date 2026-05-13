from __future__ import annotations

import sys
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "docs" / "CAPT-UniShape_Elsevier_revised_draft_reformatted.docx"

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"m": M}


def unwrap_math_delimiters_inside_scripts(root: etree._Element) -> int:
    """Remove visible grouping parentheses generated inside OMML sub/sup scripts.

    Word's UnicodeMath grouping syntax, such as x_(cond), may become a delimiter
    object inside the subscript/superscript. It renders as visible parentheses.
    For publication notation we want x with subscript cond, not x with subscript
    (cond), so the delimiter is unwrapped only when it is inside m:sub or m:sup.
    """

    changed = 0
    for script_tag in ("sub", "sup"):
        for script in root.xpath(f".//m:{script_tag}", namespaces=NS):
            for delimiter in list(script.xpath(".//m:d", namespaces=NS)):
                parent = delimiter.getparent()
                if parent is None:
                    continue
                insert_at = parent.index(delimiter)
                replacement = []
                for e in delimiter.xpath("./m:e", namespaces=NS):
                    replacement.extend(deepcopy(child) for child in e)
                if not replacement:
                    continue
                parent.remove(delimiter)
                for offset, node in enumerate(replacement):
                    parent.insert(insert_at + offset, node)
                changed += 1
    return changed


def patch_docx_in_place(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)

    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        root = etree.fromstring(zin.read("word/document.xml"))
        changed = unwrap_math_delimiters_inside_scripts(root)
        zout.writestr(
            "word/document.xml",
            etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
        )
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                continue
            zout.writestr(item, zin.read(item.filename))
    tmp.replace(path)
    return changed


def main() -> None:
    targets = [Path(arg).resolve() for arg in sys.argv[1:]] or [DEFAULT_TARGET]
    for path in targets:
        changed = patch_docx_in_place(path)
        print(f"{path} changed={changed}")


if __name__ == "__main__":
    main()

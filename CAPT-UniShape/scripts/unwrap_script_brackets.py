"""Strip stray m:d delimiters from inside m:sub / m:sup nodes (and the nested
sub/sup inside m:sSub, m:sSup, m:sSubSup, m:sPre).

Original docx had patterns like x_i^{(op)}, x_i^{(eis)}, x_i^{(cond)}, R^{(3×64)}
where the superscript content is wrapped in a default ()-delimiter. Word renders
these as parentheses around the script. We want plain script content with no
parentheses, so unwrap m:d -> use its m:e child directly.

Constraints:
- Only unwrap m:d nested INSIDE a script position (m:sub or m:sup). Body-level
  parentheses (function arguments, tuples, etc.) are NOT touched.
- Do not touch tables, body text, or paragraph structure.

The same docx is updated in place; a dated backup is created first.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import lxml.etree as ET

DOCX_PATH = Path(
    "/mnt/d/learn/论文所需材料/论文2/CAPT-UniShape/docs/"
    "CAPT-UniShape_Elsevier_revised_draft_reformatted_humanized.docx"
)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def qn(tag: str) -> str:
    return f"{{{W}}}{tag}"


def mqn(tag: str) -> str:
    return f"{{{M}}}{tag}"


SCRIPT_TAGS = (mqn("sub"), mqn("sup"))


def unwrap_script_delimiters(root: ET._Element) -> int:
    """For every m:sub / m:sup that directly contains a single m:d wrapper,
    replace the m:d with the children of its m:e child. Returns count unwrapped.
    """
    count = 0
    # Collect targets first to avoid iter() skipping after mutation.
    scripts = [el for el in root.iter() if el.tag in SCRIPT_TAGS]
    for script in scripts:
        d_children = [c for c in script if c.tag == mqn("d")]
        if not d_children:
            continue
        # Only act when the script position consists only of m:d wrappers (no
        # other math content) — that's the pattern produced by pandoc when the
        # source was "x^(op)".
        non_d = [c for c in script if c.tag != mqn("d")]
        if non_d:
            continue
        # Replace each m:d by the children of its m:e child, in original order.
        for d in d_children:
            e = d.find(mqn("e"))
            new_children = list(e) if e is not None else []
            idx = list(script).index(d)
            script.remove(d)
            for offset, ch in enumerate(new_children):
                script.insert(idx + offset, ch)
            count += 1
    return count


def main() -> None:
    if not DOCX_PATH.exists():
        sys.exit(f"DOCX not found: {DOCX_PATH}")

    backup = DOCX_PATH.with_name(
        DOCX_PATH.stem + f".backup_unwrap_script_{date.today().isoformat()}.docx"
    )
    if not backup.exists():
        shutil.copy2(DOCX_PATH, backup)
        print(f"[backup] {backup}")
    else:
        print(f"[backup-skip] {backup} (already exists)")

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        members = {n: zin.read(n) for n in zin.namelist()}

    root = ET.fromstring(members["word/document.xml"])
    n = unwrap_script_delimiters(root)
    print(f"[unwrap] unwrapped {n} m:d delimiters inside sub/sup")

    members["word/document.xml"] = ET.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    tmp = DOCX_PATH.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    tmp.replace(DOCX_PATH)
    print(f"[done] {DOCX_PATH}")


if __name__ == "__main__":
    main()

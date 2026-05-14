"""Add Word auto-numbering (SEQ) + bookmarks for figures, tables, references,
then replace inline "Fig. N" / "Table N" mentions in body text with REF
cross-reference fields.

Constraints (per user instructions):
- Do NOT touch table cell contents.
- Do NOT touch equations (m:oMath, m:oMathPara).
- Do NOT touch chapter headings (kept hand-numbered).
- Edit the docx in-place; make a dated backup first.

Result: figure/table captions and reference list items use SEQ fields so Word
re-numbers them automatically; in-text mentions of "Fig. N" and "Table N"
become REF cross-reference fields linked to the caption bookmarks.
"""
from __future__ import annotations

import re
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
NS = {"w": W, "m": M}


def qn(tag: str) -> str:
    return f"{{{W}}}{tag}"


# Style id whose name is "图表" (used as Caption alias). af7 is also a caption variant.
CAPTION_STYLE_IDS = {"aff9", "af7"}
CAPTION_PRIMARY_ID = "aff9"

CAPTION_HEAD_RE = re.compile(r"^\s*(Fig\.|Figure|Table)\s+(\d+)\.\s+")
INLINE_FIG_RE = re.compile(r"\bFig\.?\s+(\d+)\b")
INLINE_TABLE_RE = re.compile(r"\bTable\s+(\d+)\b")
REF_BRACKET_RE = re.compile(r"\[(\d+)\]")


def text_of(p: ET._Element) -> str:
    """Concatenate all w:t text in a paragraph (used for matching only)."""
    return "".join((t.text or "") for t in p.findall(f".//{qn('t')}"))


def get_pstyle_id(p: ET._Element) -> str:
    ppr = p.find(qn("pPr"))
    if ppr is None:
        return ""
    pst = ppr.find(qn("pStyle"))
    return pst.get(qn("val")) if pst is not None else ""


def set_pstyle_id(p: ET._Element, style_id: str | None) -> None:
    ppr = p.find(qn("pPr"))
    if ppr is None:
        ppr = ET.SubElement(p, qn("pPr"))
        p.insert(0, ppr)
    pst = ppr.find(qn("pStyle"))
    if style_id is None:
        if pst is not None:
            ppr.remove(pst)
        return
    if pst is None:
        pst = ET.SubElement(ppr, qn("pStyle"))
        ppr.insert(0, pst)
    pst.set(qn("val"), style_id)


def clone_rpr_for_field(p: ET._Element) -> ET._Element | None:
    """Pick a representative run's rPr to mimic font when inserting field runs.

    Returns a fresh copy of the first w:r/w:rPr we find, or None.
    """
    for r in p.findall(qn("r")):
        rpr = r.find(qn("rPr"))
        if rpr is not None:
            return ET.fromstring(ET.tostring(rpr))
    return None


def make_run_text(text: str, rpr_template: ET._Element | None) -> ET._Element:
    r = ET.Element(qn("r"))
    if rpr_template is not None:
        r.append(ET.fromstring(ET.tostring(rpr_template)))
    t = ET.SubElement(r, qn("t"))
    if text != text.strip() or "  " in text:
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def make_field_simple(
    instr: str,
    display: str,
    rpr_template: ET._Element | None,
) -> ET._Element:
    """Build <w:fldSimple w:instr="..."><w:r>...display...</w:r></w:fldSimple>."""
    fld = ET.Element(qn("fldSimple"))
    fld.set(qn("instr"), instr)
    fld.append(make_run_text(display, rpr_template))
    return fld


def make_bookmark_start(bid: int, name: str) -> ET._Element:
    bm = ET.Element(qn("bookmarkStart"))
    bm.set(qn("id"), str(bid))
    bm.set(qn("name"), name)
    return bm


def make_bookmark_end(bid: int) -> ET._Element:
    bm = ET.Element(qn("bookmarkEnd"))
    bm.set(qn("id"), str(bid))
    return bm


def strip_runs(p: ET._Element) -> list[ET._Element]:
    """Remove and return all non-pPr children (runs, math, fields, etc.)."""
    children = []
    for c in list(p):
        if c.tag == qn("pPr"):
            continue
        p.remove(c)
        children.append(c)
    return children


def dedup_caption_text(text: str) -> str:
    """Some captions had their content pasted twice; collapse the doubled form.

    Handles two shapes:
    1. Exact half-doubled string: first half == second half.
    2. Head-repeated: "Fig. N. body. Fig. N. body." — re-emit one copy.
    """
    s = text.strip()
    # half-doubled
    if len(s) >= 4 and len(s) % 2 == 0:
        half = len(s) // 2
        if s[:half].rstrip() == s[half:].lstrip():
            return s[:half].rstrip()
    # head-repeated
    head_m = re.match(r"^((?:Fig\.|Figure|Table)\s+\d+\.\s+)", s)
    if head_m:
        head = head_m.group(1)
        rest = s[len(head):]
        idx = rest.find(head)
        if idx > 0:
            first_body = rest[:idx].rstrip()
            second_body = rest[idx + len(head):].rstrip()
            if first_body == second_body:
                return head + first_body
    return s


def cleanup_caption_paragraphs(paragraphs: list[ET._Element]) -> dict:
    """Fix mis-styled paragraphs and remove duplicated caption text.

    Constraints:
    - Never delete m:oMath / w:drawing / other non-text children — captions like
      Fig. 1 contain inline math (x^op, x^cond, ...). Only normalize leading w:t
      whitespace and collapse duplicated caption heads when detected.
    """
    stats = {"unstyled_body": 0, "deduped": 0, "style_normalized": 0, "leading_space_trimmed": 0}
    for p in paragraphs:
        sid = get_pstyle_id(p)
        text = text_of(p)
        head = CAPTION_HEAD_RE.match(text)
        looks_like_caption = bool(head)

        # If a paragraph is styled as caption but doesn't start with "Fig. N." / "Table N.",
        # demote it to body text (Normal).
        if sid in CAPTION_STYLE_IDS and not looks_like_caption:
            set_pstyle_id(p, None)
            stats["unstyled_body"] += 1
            continue

        # Normalize caption variants to the primary caption style id.
        if sid in CAPTION_STYLE_IDS and sid != CAPTION_PRIMARY_ID:
            set_pstyle_id(p, CAPTION_PRIMARY_ID)
            stats["style_normalized"] += 1

        if not looks_like_caption:
            continue

        # Try to dedupe doubled caption text. We do this only when the paragraph
        # contains pure text (no oMath / drawing) — those captions can be safely
        # rebuilt; otherwise we leave the content as-is to preserve math runs.
        has_non_text = any(c.tag != qn("pPr") and c.tag != qn("r") for c in p)
        deduped = dedup_caption_text(text)
        if deduped != text.strip() and not has_non_text:
            rpr = clone_rpr_for_field(p)
            for c in list(p):
                if c.tag != qn("pPr"):
                    p.remove(c)
            p.append(make_run_text(deduped, rpr))
            stats["deduped"] += 1
            continue

        # Trim leading whitespace from the first w:t without touching other runs.
        if text != text.lstrip():
            first_r = p.find(qn("r"))
            if first_r is not None:
                first_t = first_r.find(qn("t"))
                if first_t is not None and first_t.text:
                    first_t.text = first_t.text.lstrip()
                    stats["leading_space_trimmed"] += 1
    return stats


def caption_bookmark_name(kind: str, number: int) -> str:
    return f"_Ref_{kind}_{number}"


SEQ_FIELD_NAME = {"Fig": "Figure", "Table": "Table", "Ref": "Reference"}


def replace_caption_with_seq(
    p: ET._Element, kind: str, number: int, bookmark_id: int
) -> bool:
    """Replace the caption head ("Fig. N. " / "Table N. ") with bookmarked SEQ
    field, preserving any subsequent text runs, math, and drawings in the
    original order.
    """
    rpr_template = clone_rpr_for_field(p)

    text = text_of(p)
    m = CAPTION_HEAD_RE.match(text)
    if not m:
        return False

    label_word = "Fig." if kind == "Fig" else "Table"
    seq_name = SEQ_FIELD_NAME[kind]
    bm_name = caption_bookmark_name(kind, number)

    head_end = m.end()
    # Walk children in order, consuming exactly head_end characters worth of
    # plain w:t text from leading w:r elements. After head_end is consumed,
    # keep every remaining element verbatim.
    children = strip_runs(p)
    consumed = 0
    tail_children: list[ET._Element] = []
    for c in children:
        if consumed >= head_end:
            tail_children.append(c)
            continue
        if c.tag != qn("r"):
            # Non-text leading element before head fully consumed — keep as tail.
            tail_children.append(c)
            continue
        # Count text in this run
        ts = c.findall(qn("t"))
        run_text = "".join((t.text or "") for t in ts)
        if not run_text:
            # empty / non-text run, keep it before tail
            tail_children.append(c)
            continue
        end = consumed + len(run_text)
        if end <= head_end:
            # entirely within head -> drop this run
            consumed = end
            continue
        # Partial: split — keep the post-head portion of this run as a new run
        cut = head_end - consumed
        leftover = run_text[cut:]
        consumed = head_end
        if leftover:
            # Reuse this run's rPr if any
            rpr_local = c.find(qn("rPr"))
            new_r = make_run_text(leftover, rpr_local if rpr_local is not None else rpr_template)
            tail_children.append(new_r)

    p.append(make_bookmark_start(bookmark_id, bm_name))
    p.append(make_run_text(f"{label_word} ", rpr_template))
    p.append(
        make_field_simple(
            f" SEQ {seq_name} \\* ARABIC ",
            str(number),
            rpr_template,
        )
    )
    p.append(make_bookmark_end(bookmark_id))
    p.append(make_run_text(". ", rpr_template))
    for c in tail_children:
        p.append(c)
    return True


def process_captions(paragraphs: list[ET._Element], bid_counter: list[int]) -> dict:
    """Iterate caption-styled paragraphs in document order, assign SEQ + bookmark.

    Figure and Table have independent counters (SEQ Figure / SEQ Table).
    """
    stats = {"fig": 0, "table": 0, "skipped": 0}
    fig_n = 0
    tbl_n = 0
    for p in paragraphs:
        sid = get_pstyle_id(p)
        if sid not in CAPTION_STYLE_IDS:
            continue
        text = text_of(p)
        m = CAPTION_HEAD_RE.match(text)
        if not m:
            stats["skipped"] += 1
            continue
        label = m.group(1)
        kind = "Fig" if label.startswith("Fig") else "Table"
        if kind == "Fig":
            fig_n += 1
            number = fig_n
            stats["fig"] += 1
        else:
            tbl_n += 1
            number = tbl_n
            stats["table"] += 1
        bid_counter[0] += 1
        replace_caption_with_seq(p, kind, number, bid_counter[0])
    return stats


def find_references_paragraph(paragraphs: list[ET._Element]) -> int:
    """Return index of the first paragraph after the 'References' heading."""
    in_refs = False
    for i, p in enumerate(paragraphs):
        sid = get_pstyle_id(p)
        text = text_of(p).strip()
        if sid.startswith("1") or sid in {"Heading1", "Heading 1", "1"} or re.match(r"^Heading\b", sid):
            pass
        if "References" == text:
            in_refs = True
            continue
        if in_refs and text:
            return i
    return -1


def split_and_number_references(
    body: ET._Element,
    paragraphs: list[ET._Element],
    bid_counter: list[int],
) -> dict:
    """Find the single References-content paragraph, split it into individual
    reference paragraphs, and replace each "[N]" head with SEQ Reference + bookmark.
    """
    stats = {"refs": 0}
    ref_para_idx = find_references_paragraph(paragraphs)
    if ref_para_idx < 0:
        return stats
    ref_p = paragraphs[ref_para_idx]
    full_text = text_of(ref_p).strip()
    # Split before each "[N] " marker; keep the marker so we can map number -> body
    parts = re.split(r"(?=\[\d+\]\s)", full_text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return stats

    rpr_template = clone_rpr_for_field(ref_p)
    parent = ref_p.getparent()
    ref_index = list(parent).index(ref_p)

    # Capture the pPr from original so each new ref paragraph keeps the same style.
    orig_ppr = ref_p.find(qn("pPr"))

    # Remove the original paragraph; we'll insert the new ones at its index.
    parent.remove(ref_p)

    insert_at = ref_index
    for part in parts:
        m = re.match(r"^\[(\d+)\]\s*(.*)$", part)
        if not m:
            continue
        number = int(m.group(1))
        body_text = m.group(2)

        new_p = ET.Element(qn("p"))
        if orig_ppr is not None:
            new_p.append(ET.fromstring(ET.tostring(orig_ppr)))

        bid_counter[0] += 1
        bm_name = caption_bookmark_name("Ref", number)
        new_p.append(make_bookmark_start(bid_counter[0], bm_name))
        new_p.append(make_run_text("[", rpr_template))
        new_p.append(
            make_field_simple(
                " SEQ Reference \\* ARABIC ",
                str(number),
                rpr_template,
            )
        )
        new_p.append(make_run_text("] ", rpr_template))
        new_p.append(make_bookmark_end(bid_counter[0]))
        new_p.append(make_run_text(body_text, rpr_template))

        parent.insert(insert_at, new_p)
        insert_at += 1
        stats["refs"] += 1
    return stats


def is_inside_table(p: ET._Element) -> bool:
    """Walk ancestors to determine if the paragraph sits inside a w:tbl."""
    el = p.getparent()
    while el is not None:
        if el.tag == qn("tbl"):
            return True
        el = el.getparent()
    return False


def replace_inline_fig_table_in_paragraph(p: ET._Element) -> int:
    """Within a body paragraph (non-caption, non-table), replace ' Fig. N ' and
    ' Table N ' tokens with a REF cross-reference field. Returns # of replacements.

    Implementation note: we walk children sequentially; for each w:r whose w:t
    text matches a pattern, we split the run into prefix-text + literal label +
    SEQ-field + suffix-text. We only touch plain w:t text (no math, no nested).
    """
    rpr_template = clone_rpr_for_field(p)
    replacements = 0

    children = list(p)
    new_children: list[ET._Element] = []

    for c in children:
        if c.tag != qn("r"):
            new_children.append(c)
            continue
        # gather text and rPr
        rpr = c.find(qn("rPr"))
        # only handle simple runs: one w:t, no nested fld
        ts = c.findall(qn("t"))
        if len(ts) != 1 or c.find(qn("fldChar")) is not None or c.find(qn("instrText")) is not None:
            new_children.append(c)
            continue
        text = ts[0].text or ""
        # Search for both inline Fig and Table.
        pattern = re.compile(r"(\bFig\.?\s+(\d+)\b|\bTable\s+(\d+)\b)")
        if not pattern.search(text):
            new_children.append(c)
            continue

        # Use this run's rPr as the template for the inserted runs.
        local_rpr = rpr if rpr is not None else rpr_template

        pos = 0
        for m in pattern.finditer(text):
            pre = text[pos : m.start()]
            if pre:
                new_children.append(make_run_text(pre, local_rpr))
            full = m.group(0)
            if m.group(2):  # Fig.
                number = int(m.group(2))
                # Decide label: keep "Fig. " or "Fig " as written
                label = full[: full.rfind(m.group(2))]
                bm = caption_bookmark_name("Fig", number)
            else:  # Table
                number = int(m.group(3))
                label = full[: full.rfind(m.group(3))]
                bm = caption_bookmark_name("Table", number)
            # Append label run + REF field
            new_children.append(make_run_text(label, local_rpr))
            new_children.append(
                make_field_simple(f" REF {bm} \\h ", str(number), local_rpr)
            )
            pos = m.end()
            replacements += 1
        suffix = text[pos:]
        if suffix:
            new_children.append(make_run_text(suffix, local_rpr))

    if replacements:
        # rebuild paragraph children in order, keeping pPr first
        ppr = p.find(qn("pPr"))
        for c in list(p):
            p.remove(c)
        if ppr is not None:
            p.append(ppr)
        for c in new_children:
            p.append(c)
    return replacements


def process_inline_refs(paragraphs: list[ET._Element]) -> dict:
    """Walk every paragraph; replace inline Fig./Table mentions in body text
    only. Skip captions, table cells, and the References list.
    """
    stats = {"replacements": 0, "paragraphs_changed": 0, "skipped_captions": 0, "skipped_in_table": 0}
    in_refs = False
    for p in paragraphs:
        sid = get_pstyle_id(p)
        text = text_of(p).strip()
        if text == "References":
            in_refs = True
            continue
        if in_refs:
            continue
        if sid in CAPTION_STYLE_IDS:
            stats["skipped_captions"] += 1
            continue
        if is_inside_table(p):
            stats["skipped_in_table"] += 1
            continue
        n = replace_inline_fig_table_in_paragraph(p)
        if n:
            stats["replacements"] += n
            stats["paragraphs_changed"] += 1
    return stats


def main() -> None:
    if not DOCX_PATH.exists():
        sys.exit(f"DOCX not found: {DOCX_PATH}")

    backup = DOCX_PATH.with_name(
        DOCX_PATH.stem + f".backup_auto_numbering_{date.today().isoformat()}.docx"
    )
    if not backup.exists():
        shutil.copy2(DOCX_PATH, backup)
        print(f"[backup] {backup}")
    else:
        print(f"[backup-skip] {backup} (already exists)")

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        members = {n: zin.read(n) for n in zin.namelist()}

    doc_xml = members["word/document.xml"]
    root = ET.fromstring(doc_xml)
    body = root.find(qn("body"))
    paragraphs = body.findall(qn("p"))

    bid_counter = [200]  # avoid colliding with pandoc's bookmark ids (0..32)

    cleanup_stats = cleanup_caption_paragraphs(paragraphs)
    print(f"[cleanup] {cleanup_stats}")

    # refresh paragraph list (no structural change, but defensive)
    paragraphs = body.findall(qn("p"))
    caption_stats = process_captions(paragraphs, bid_counter)
    print(f"[captions] {caption_stats}")

    paragraphs = body.findall(qn("p"))
    ref_stats = split_and_number_references(body, paragraphs, bid_counter)
    print(f"[references] {ref_stats}")

    paragraphs = body.findall(qn("p"))
    inline_stats = process_inline_refs(paragraphs)
    print(f"[inline-refs] {inline_stats}")

    new_doc_xml = ET.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    members["word/document.xml"] = new_doc_xml

    tmp = DOCX_PATH.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    tmp.replace(DOCX_PATH)
    print(f"[done] {DOCX_PATH}")


if __name__ == "__main__":
    main()

import io
import json
import re
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from PIL import Image
import pymupdf as fitz
from .exceptions import handle_error

# Rendering configuration
RENDER_DPI = 200
CROP_PAD_PX = 8
VLM_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

# Compute header and footer margins
def find_margins(page: fitz.Page, pno: int) -> Tuple[float, float]:
    page_h = page.rect.height
    if pno == 0:
        return 0.04 * page_h, 0.96 * page_h

    top_y = 0.05 * page_h
    bot_y = 0.95 * page_h
    for block in page.get_text("blocks"):
        text = str(block[4]).strip()
        y0, y1 = float(block[1]), float(block[3])
        block_h = y1 - y0
        if y1 < 0.08 * page_h and block_h <= 24:
            if len(text.split()) <= 6 or re.search(r"arxiv|preprint|proceedings", text, re.I):
                top_y = max(top_y, y1 + 4.0)
        if y0 > 0.92 * page_h and block_h <= 24:
            if len(text.split()) <= 4 or re.search(r"^\d+$|copyright", text, re.I):
                bot_y = min(bot_y, y0 - 4.0)
    return top_y, bot_y

# Normalize bounding box coordinates
def normalize_box(box: Any) -> Optional[List[float]]:
    if isinstance(box, dict):
        box = [box.get("ymin", 0), box.get("xmin", 0), box.get("ymax", 0), box.get("xmax", 0)]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return [float(v) for v in box]
        except (ValueError, TypeError):
            return None
    return None

# Build caption inventory from text
def scan_captions(doc: fitz.Document) -> Dict[int, List[Dict[str, str]]]:
    pattern = re.compile(
        r"^(Figure|Fig\.?|Table|Tab\.?)\s+([0-9A-Za-z]+(?:\.[0-9]+)?)"
        r"(?:\s*[:.\-–—]|\s*\n|$)",
        re.I
    )
    inventory: Dict[int, List[Dict[str, str]]] = {}
    for pno, page in enumerate(doc):
        for block in page.get_text("blocks"):
            text = str(block[4]).strip()
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if not lines:
                continue
            match = pattern.match(lines[0])
            if match:
                raw_kind = match.group(1).lower()
                kind = "figure" if raw_kind.startswith("fig") else "table"
                elem_id = match.group(2)
                if pno not in inventory:
                    inventory[pno] = []
                inventory[pno].append({"type": kind, "id": elem_id, "caption": text})
    return inventory

# Query Gemini for detected elements
def _query_vlm(client: Any, page_img: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    for model_name in VLM_MODELS:
        for attempt in range(2):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=[page_img, prompt],
                    config={"response_mime_type": "application/json"}
                )
                if res and res.text:
                    raw_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", res.text.strip())
                    parsed = json.loads(raw_json)
                    if isinstance(parsed, dict):
                        for val in parsed.values():
                            if isinstance(val, list):
                                return val
                    elif isinstance(parsed, list):
                        return parsed
                break
            except Exception as err:
                if any(k in str(err).lower() for k in ["429", "quota", "resource_exhausted"]):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    return []

# Detect all elements on page
def detect_page_elements(client: Any, page_img: Image.Image, pno: int) -> List[Dict[str, Any]]:
    prompt = (
        "Identify every figure, plot, chart, diagram, and table on this academic paper page.\n"
        "For each element output:\n"
        "  type: 'figure' or 'table'\n"
        "  id: identifier exactly as printed (e.g. '1', '2', 'A1')\n"
        "  caption: the full caption text of this element\n"
        "  box_2d: [ymin, xmin, ymax, xmax] in 0-1000 scale.\n"
        "           The box MUST tightly enclose the ENTIRE element:\n"
        "           every row, column, header, footnote, and the caption line.\n"
        "Return ONLY valid JSON:\n"
        "{\"elements\": [{\"type\": \"figure\"|\"table\", \"id\": \"...\", "
        "\"caption\": \"...\", \"box_2d\": [ymin, xmin, ymax, xmax]}]}"
    )
    return _query_vlm(client, page_img, prompt)

# Targeted detection for missed element
def detect_single_element(
    client: Any, page_img: Image.Image, elem_type: str, elem_id: str
) -> Optional[Dict[str, Any]]:
    kind = "Figure" if elem_type == "figure" else "Table"
    prompt = (
        f"On this academic paper page there is {kind} {elem_id}.\n"
        f"Find {kind} {elem_id} and return its exact bounding box enclosing all rows, columns, headers, and caption.\n"
        "Return ONLY valid JSON:\n"
        "{\"elements\": [{\"type\": \"" + elem_type + "\", \"id\": \"" + elem_id + "\", "
        "\"caption\": \"...\", \"box_2d\": [ymin, xmin, ymax, xmax]}]}"
    )
    results = _query_vlm(client, page_img, prompt)
    for item in results:
        item_type = str(item.get("type", "")).lower()
        item_id = str(item.get("id", "")).strip()
        if elem_type in item_type and item_id == elem_id:
            return item
    return results[0] if results else None

# Crop PIL image by coordinates
def crop_box(
    page_img: Image.Image,
    ymin: float, xmin: float, ymax: float, xmax: float,
    pad: int = CROP_PAD_PX
) -> Image.Image:
    width, height = page_img.size
    x0 = max(0, int((xmin / 1000.0) * width) - pad)
    y0 = max(0, int((ymin / 1000.0) * height) - pad)
    x1 = min(width, int((xmax / 1000.0) * width) + pad)
    y1 = min(height, int((ymax / 1000.0) * height) + pad)
    return page_img.crop((x0, y0, x1, y1))

# Process single detected bounding box
def _process_item(
    item: Dict[str, Any],
    pno: int,
    page: fitz.Page,
    page_img: Image.Image,
    out_dir: Path,
    seen_ids: Set[str],
    figs: List[Dict],
    tabs: List[Dict],
    page_map: Dict[int, List[fitz.Rect]],
    scale_factor: float
) -> None:
    raw_type = str(item.get("type", "")).lower()
    raw_id = str(item.get("id", "")).strip()
    caption = str(item.get("caption", "")).strip()
    box = normalize_box(item.get("box_2d"))

    if not box or len(box) != 4:
        return

    ymin, xmin, ymax, xmax = box
    if not raw_id or raw_id.lower() == "none":
        match = re.search(r"(?:table|tab\.?|figure|fig\.?)\s*([0-9a-zA-Z]+)", caption, re.I)
        if match:
            raw_id = match.group(1)

    if not raw_id:
        return

    cropped_img = crop_box(page_img, ymin, xmin, ymax, xmax)
    if cropped_img.width < 25 or cropped_img.height < 25:
        return

    elem_id = int(raw_id) if raw_id.isdigit() else raw_id
    elem_kind = "figure" if "fig" in raw_type else "table"
    unique_key = f"{elem_kind}_{pno}_{elem_id}"

    if unique_key in seen_ids:
        return
    seen_ids.add(unique_key)

    img_w, img_h = page_img.size
    px0 = max(0, int((xmin / 1000.0) * img_w) - CROP_PAD_PX)
    py0 = max(0, int((ymin / 1000.0) * img_h) - CROP_PAD_PX)
    px1 = min(img_w, int((xmax / 1000.0) * img_w) + CROP_PAD_PX)
    py1 = min(img_h, int((ymax / 1000.0) * img_h) + CROP_PAD_PX)

    doc_rect = fitz.Rect(
        px0 * scale_factor, py0 * scale_factor,
        px1 * scale_factor, py1 * scale_factor
    )
    page_map[pno].append(doc_rect)

    file_name = f"{elem_kind}_{elem_id}.png"
    save_path = out_dir / file_name
    cropped_img.save(str(save_path))

    record = {
        "id": elem_id,
        "type": elem_kind,
        "page": pno + 1,
        "caption": caption or f"{elem_kind.capitalize()} {elem_id}",
        "path": str(save_path),
        "bbox": [doc_rect.x0, doc_rect.y0, doc_rect.x1, doc_rect.y1]
    }

    if elem_kind == "figure":
        figs.append(record)
    else:
        tabs.append(record)

# Extract figures and tables from PDF
@handle_error("Paper elements extraction")
def extract_paper(
    client: Any,
    pdf_path: str,
    out_dir: str = "uploaded_docs/extracted_figures",
    dpi: int = RENDER_DPI
) -> Tuple[fitz.Document, List[Dict], List[Dict], Dict[int, List[fitz.Rect]]]:
    doc = fitz.open(pdf_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for old_file in out_path.glob("*"):
        if old_file.is_file():
            old_file.unlink(missing_ok=True)

    figs: List[Dict] = []
    tabs: List[Dict] = []
    page_map: Dict[int, List[fitz.Rect]] = {pno: [] for pno in range(len(doc))}
    inventory = scan_captions(doc)

    candidate_pages = [
        pno for pno, page in enumerate(doc)
        if len(page.get_images()) > 0 or len(page.get_drawings()) > 4 or pno in inventory
    ]

    page_images: Dict[int, Image.Image] = {}
    for pno in candidate_pages:
        pix = doc[pno].get_pixmap(dpi=dpi)
        page_images[pno] = Image.open(io.BytesIO(pix.tobytes("png")))

    page_elements: Dict[int, List[Dict]] = {}
    max_workers = min(2, max(1, len(candidate_pages)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(detect_page_elements, client, page_images[pno], pno): pno
            for pno in candidate_pages
        }
        for future in concurrent.futures.as_completed(futures):
            pno = futures[future]
            try:
                page_elements[pno] = future.result()
            except Exception:
                page_elements[pno] = []

    seen_ids: Set[str] = set()
    scale_factor = 72.0 / dpi

    for pno in range(len(doc)):
        page_img = page_images.get(pno)
        if page_img is None:
            continue
        for item in page_elements.get(pno, []):
            _process_item(
                item, pno, doc[pno], page_img, out_path,
                seen_ids, figs, tabs, page_map, scale_factor
            )

    extracted_figs = {str(f["id"]) for f in figs}
    extracted_tabs = {str(t["id"]) for t in tabs}
    for pno, items in inventory.items():
        for inv_item in items:
            kind = inv_item["type"]
            elem_id = inv_item["id"]
            is_missing = (kind == "figure" and elem_id not in extracted_figs) or \
                         (kind == "table" and elem_id not in extracted_tabs)
            if is_missing:
                page_img = page_images.get(pno)
                if page_img is None:
                    pix = doc[pno].get_pixmap(dpi=dpi)
                    page_img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_images[pno] = page_img
                recovered = detect_single_element(client, page_img, kind, elem_id)
                if recovered:
                    _process_item(
                        recovered, pno, doc[pno], page_img, out_path,
                        seen_ids, figs, tabs, page_map, scale_factor
                    )
                time.sleep(0.3)

    def sort_key(item: Dict) -> Tuple[int, Any]:
        val = item["id"]
        return (0, int(val)) if str(val).isdigit() else (1, str(val))

    figs.sort(key=sort_key)
    tabs.sort(key=sort_key)
    return doc, figs, tabs, page_map

# Extract clean continuous prose text
@handle_error("Paper prose extraction")
def extract_prose(doc: fitz.Document, page_map: Dict[int, List[fitz.Rect]]) -> Tuple[str, List[Dict]]:
    page_docs: List[Dict] = []
    text_blocks: List[str] = []

    for pno, page in enumerate(doc):
        top_y, bot_y = find_margins(page, pno)
        blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]
        exclusions = page_map.get(pno, [])
        lines: List[str] = []

        for block in blocks:
            box = fitz.Rect(block[:4])
            if box.y1 <= top_y or box.y0 >= bot_y:
                continue
            if any((box & ex).get_area() > 0.40 * box.get_area() for ex in exclusions):
                continue

            cleaned = []
            for ln in str(block[4]).split("\n"):
                ln = ln.strip()
                if not ln or re.match(r"^arXiv:\d+\.\d+", ln, re.I) or re.match(r"^\d+$", ln):
                    continue
                cleaned.append(ln)

            chunk = " ".join(" ".join(cleaned).split())
            if chunk:
                lines.append(chunk)
                text_blocks.append(chunk)

        if lines:
            page_docs.append({"page": pno + 1, "text": "\n\n".join(lines)})

    stitched: List[str] = []
    for chunk in text_blocks:
        if not stitched:
            stitched.append(chunk)
            continue
        prev = stitched[-1]
        if prev.endswith("-") and not prev.endswith(" -"):
            stitched[-1] = prev[:-1] + chunk
        elif not re.search(r'[.!?:"]\s*$', prev) and re.match(r'^[a-z0-9,;)]', chunk):
            stitched[-1] = prev + " " + chunk
        else:
            stitched.append(chunk)

    clean_prose = [re.sub(r'(\b[a-zA-Z]+)-\s+([a-zA-Z]+\b)', r'\1\2', p) for p in stitched]
    return "\n\n".join(clean_prose), page_docs

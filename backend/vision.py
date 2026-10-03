from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

import time

# Supported Gemini vision models
VISION_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

from .model_manager import get_candidate_models, record_model_success, record_model_failure

# Generate content with Gemini vision
def call_gemini(client: Any, contents: List[Any]) -> str:
    for model_name in get_candidate_models(VISION_MODELS):
        try:
            res = client.models.generate_content(model=model_name, contents=contents)
            text = getattr(res, "text", "") or ""
            if text.strip():
                record_model_success(model_name)
                return text.strip()
        except Exception as err:
            record_model_failure(model_name, err)
            err_str = str(err).lower()
            if any(k in err_str for k in ["429", "resource_exhausted", "quota", "404", "not_found"]):
                continue
            time.sleep(0.5)
            continue
    return ""

# Parse table into text
def parse_table(client: Any, tab_item: Dict, cache_dir: str = "uploaded_docs/extracted_figures") -> str:
    cache_path = Path(cache_dir) / f"table_{tab_item['id']}_parsed.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()

    prompt = (
        f"You are an expert academic table parser for RAG retrieval.\n"
        f"Analyze this table image visually and analytically.\n"
        f"Do NOT just repeat the caption; read every row, column header, subheader, and cell directly from the image.\n"
        f"Transcribe the entire table into a clean structured list:\n\n"
        f"Table {tab_item['id']}: <Caption Title>\n"
        f"- <Column1>: <Value> | <Column2>: <Value> | <Column3>: <Value> ...\n\n"
        f"Rules:\n"
        f"1. Every row MUST be an itemized bullet point with accurate key-value column pairs.\n"
        f"2. Read numbers, units, and headers accurately from the visual table cells.\n"
        f"3. Output ONLY the itemized textual list with header.\n"
        f"Caption Context: {tab_item['caption']}"
    )

    try:
        with Image.open(tab_item["path"]) as img:
            img.load()
            parsed_text = call_gemini(client, [img, prompt])
        if parsed_text:
            cache_path.write_text(parsed_text, encoding="utf-8")
            return parsed_text
    except Exception:
        pass
    return f"Table {tab_item['id']}: {tab_item['caption']}"

# Summarize figure architecture and charts
def summarize_figure(client: Any, fig_item: Dict, cache_dir: str = "uploaded_docs/extracted_figures") -> str:
    cache_path = Path(cache_dir) / f"figure_{fig_item['id']}_summary.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()

    prompt = (
        f"Perform a thorough visual and analytical examination of this research paper figure for retrieval.\n"
        f"Do NOT simply restate or summarize the caption text.\n"
        f"Visually and analytically analyze what is depicted inside the image:\n"
        f"1. Architecture diagrams: Trace the visual pipeline, modules, inputs, outputs, unfreezing states, and arrows.\n"
        f"2. Plots & Charts: State the x-axis, y-axis, lines/bars, metrics, baseline comparisons, and numerical trends.\n"
        f"3. Qualitative examples: Describe the visual entities, rows, columns, prompt labels, and compare quality differences.\n"
        f"Figure Caption Context: {fig_item['caption']}"
    )

    try:
        with Image.open(fig_item["path"]) as img:
            img.load()
            summary_text = call_gemini(client, [img, prompt])
        if summary_text:
            cache_path.write_text(summary_text, encoding="utf-8")
            return summary_text
    except Exception:
        pass
    return f"Figure {fig_item['id']}: {fig_item['caption']}"

import os
import shutil
import concurrent.futures
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from .exceptions import handle_error
from .extractor import extract_paper, extract_prose
from .vision import parse_table, summarize_figure
from .rag import build_index, get_retriever

load_dotenv()

# Initialize Gemini API client
def get_genai_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)

# Execute multimodal ingestion pipeline
@handle_error("End-to-end ingestion pipeline")
def run_pipeline(
    pdf_source: str,
    upload_dir: str = "uploaded_docs",
    chroma_dir: str = "./chroma_db",
    paper_name: Optional[str] = None,
    status_cb: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    def update(stage: str, prog: float):
        if status_cb:
            status_cb(stage, prog)

    title = paper_name or Path(pdf_source).stem.replace("_", " ")
    up_path = Path(upload_dir)
    fig_dir = up_path / "extracted_figures"
    up_path.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for old_file in fig_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink(missing_ok=True)

    dest_pdf = up_path / "document.pdf"
    if Path(pdf_source).resolve() != dest_pdf.resolve():
        shutil.copy2(pdf_source, str(dest_pdf))

    client = get_genai_client()

    # Extract visual elements via VLM
    update("Extracting figures and tables with VLM...", 0.15)
    doc, figs, tabs, page_map = extract_paper(client, str(dest_pdf), out_dir=str(fig_dir))

    # Extract clean prose text
    update("Extracting clean continuous prose text...", 0.30)
    clean_prose, page_docs = extract_prose(doc, page_map)
    doc.close()

    # Process visual elements (tables and figures) concurrently
    total_tabs = len(tabs)
    total_figs = len(figs)
    total_items = total_tabs + total_figs

    if total_items > 0:
        update(f"Analyzing {total_tabs} tables and {total_figs} figures concurrently...", 0.45)
        max_workers = min(8, max(1, total_items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            tab_futures = {
                pool.submit(parse_table, client, t, str(fig_dir)): ("table", t)
                for t in tabs
            }
            fig_futures = {
                pool.submit(summarize_figure, client, f, str(fig_dir)): ("figure", f)
                for f in figs
            }
            all_futures = {**tab_futures, **fig_futures}
            completed = 0
            for future in concurrent.futures.as_completed(all_futures):
                item_type, item = all_futures[future]
                try:
                    res = future.result()
                    if item_type == "table":
                        item["parsed_text"] = res
                    else:
                        item["summary"] = res
                except Exception:
                    if item_type == "table":
                        item["parsed_text"] = f"Table {item.get('id', '')}: {item.get('caption', '')}"
                    else:
                        item["summary"] = f"Figure {item.get('id', '')}: {item.get('caption', '')}"
                completed += 1
                update(f"Analyzed visual element {completed}/{total_items}...", 0.45 + 0.40 * (completed / total_items))

    # Index into vector database
    update("Building persistent Chroma vector store...", 0.90)
    vstore = build_index(page_docs, tabs, figs, chroma_dir=chroma_dir, paper_name=title)
    retriever = get_retriever(vstore, k=8)

    update("Ingestion and indexing complete!", 1.0)
    return {
        "pdf_path": str(dest_pdf),
        "doc": doc,
        "figs": figs,
        "tabs": tabs,
        "page_docs": page_docs,
        "clean_prose": clean_prose,
        "vstore": vstore,
        "retriever": retriever,
        "paper_name": title
    }

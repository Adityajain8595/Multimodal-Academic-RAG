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

    # Transcribe tables in parallel
    total_tabs = len(tabs)
    if total_tabs > 0:
        update(f"Transcribing {total_tabs} tables into markdown...", 0.45)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, total_tabs)) as pool:
            futures = {
                pool.submit(parse_table, client, t, str(fig_dir)): t
                for t in tabs
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                t = futures[future]
                t["parsed_text"] = future.result()
                completed += 1
                update(f"Transcribing table {completed}/{total_tabs}...", 0.45 + 0.15 * (completed / total_tabs))

    # Summarize figures in parallel
    total_figs = len(figs)
    if total_figs > 0:
        update(f"Analyzing {total_figs} figures visually...", 0.60)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, total_figs)) as pool:
            futures = {
                pool.submit(summarize_figure, client, f, str(fig_dir)): f
                for f in figs
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                f = futures[future]
                f["summary"] = future.result()
                completed += 1
                update(f"Analyzing figure {completed}/{total_figs}...", 0.60 + 0.25 * (completed / total_figs))

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

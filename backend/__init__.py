# Modular multimodal backend package
from .exceptions import RAGError, handle_error
from .extractor import find_margins, extract_paper, extract_prose
from .vision import call_gemini, parse_table, summarize_figure
from .rag import build_index, get_retriever, create_chain, ask_paper, rewrite_query
from .pipeline import run_pipeline
from .model_manager import get_candidate_models, record_model_success, record_model_failure

__all__ = [
    "RAGError", "handle_error",
    "find_margins", "extract_paper", "extract_prose",
    "call_gemini", "parse_table", "summarize_figure",
    "build_index", "get_retriever", "create_chain", "ask_paper", "rewrite_query",
    "run_pipeline",
    "get_candidate_models", "record_model_success", "record_model_failure"
]

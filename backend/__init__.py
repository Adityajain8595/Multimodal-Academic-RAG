from .exceptions import RAGError, handle_error
from .extractor import extract_paper, extract_prose, find_margins
from .model_manager import (
    VISION_MODELS,
    get_candidate_models,
    record_model_failure,
    record_model_success,
)
from .pipeline import run_pipeline
from .rag import (
    ask_paper,
    build_index,
    create_chain,
    get_retriever,
    load_vector_store,
    rewrite_query,
    sanitize_answer,
)
from .vision import call_gemini, parse_table, summarize_figure

__all__ = [
    "RAGError",
    "handle_error",
    "find_margins",
    "extract_paper",
    "extract_prose",
    "call_gemini",
    "parse_table",
    "summarize_figure",
    "build_index",
    "get_retriever",
    "load_vector_store",
    "create_chain",
    "ask_paper",
    "rewrite_query",
    "sanitize_answer",
    "run_pipeline",
    "VISION_MODELS",
    "get_candidate_models",
    "record_model_success",
    "record_model_failure",
]


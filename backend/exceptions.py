from functools import wraps
from typing import Callable, Any

# Custom domain exception classes
class RAGError(Exception):
    def __init__(self, msg: str, code: str = "ERR_RAG"):
        self.msg = msg
        self.code = code
        super().__init__(f"[{code}] {msg}")


class PDFError(RAGError):
    def __init__(self, msg: str):
        super().__init__(msg, code="ERR_PDF")


class VisionError(RAGError):
    def __init__(self, msg: str):
        super().__init__(msg, code="ERR_VISION")


class VectorError(RAGError):
    def __init__(self, msg: str):
        super().__init__(msg, code="ERR_VECTOR")


# Central error handler decorator
def handle_error(stage_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return fn(*args, **kwargs)
            except RAGError:
                raise
            except Exception as err:
                err_type = type(err).__name__
                clean_msg = f"{stage_name} failed ({err_type}): {str(err).strip()}"
                
                lower_stage = stage_name.lower()
                if any(k in lower_stage for k in ["pdf", "page", "margin"]):
                    raise PDFError(clean_msg) from err
                if any(k in lower_stage for k in ["vision", "table", "figure", "gemini"]):
                    raise VisionError(clean_msg) from err
                if any(k in lower_stage for k in ["vector", "chroma", "retrieval"]):
                    raise VectorError(clean_msg) from err
                raise RAGError(clean_msg) from err
        return wrapper
    return decorator

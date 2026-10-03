from functools import wraps
from typing import Callable, Any

# Base domain exception
class RAGError(Exception):
    pass

# Stage execution wrapper
def handle_error(stage: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return fn(*args, **kwargs)
            except RAGError:
                raise
            except Exception as err:
                raise RAGError(f"{stage} failed: {err}") from err
        return wrapper
    return decorator

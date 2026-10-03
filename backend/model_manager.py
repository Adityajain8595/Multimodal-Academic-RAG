import time
import threading
from typing import List, Dict, Any

# Dynamic in-memory cooldown registry for Gemini models
_model_cooldowns: Dict[str, float] = {}
_cooldown_lock = threading.Lock()

def get_candidate_models(model_list: List[str]) -> List[str]:
    """
    Returns candidate models preserving original list priority,
    temporarily prioritizing models not currently in cooldown.
    """
    now = time.time()
    with _cooldown_lock:
        available = [m for m in model_list if _model_cooldowns.get(m, 0) <= now]
        if available:
            return available
        # If all models are temporarily in cooldown, try the one that will expire soonest
        return sorted(model_list, key=lambda m: _model_cooldowns.get(m, 0))

def record_model_success(model_name: str) -> None:
    """Clear cooldown on successful API response."""
    with _cooldown_lock:
        _model_cooldowns.pop(model_name, None)

def record_model_failure(model_name: str, err: Exception) -> None:
    """Record cooldown window based on error type to prevent wasted retries."""
    err_str = str(err).lower()
    now = time.time()
    with _cooldown_lock:
        if any(k in err_str for k in ["exceeded your current quota", "check your plan and billing"]):
            _model_cooldowns[model_name] = now + 60.0
        elif any(k in err_str for k in ["429", "resource_exhausted", "quota"]):
            # Short cooldown for rate-limit / concurrency spikes
            _model_cooldowns[model_name] = now + 5.0
        elif any(k in err_str for k in ["404", "not_found", "no longer available"]):
            _model_cooldowns[model_name] = now + 300.0
        else:
            _model_cooldowns[model_name] = now + 5.0

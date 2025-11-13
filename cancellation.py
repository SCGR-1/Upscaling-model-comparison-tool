"""Cancellation handling for async operations."""
import threading

class CancellationError(Exception):
    """Raised when processing is cancelled by user."""
    pass

_cancellation_flags = {}  # {request_id: threading.Event}

def check_cancellation(request_id: str):
    """Check if the request has been cancelled."""
    if request_id and _cancellation_flags.get(request_id, threading.Event()).is_set():
        raise CancellationError("Processing cancelled by user")

def register_cancellation(request_id: str):
    """Register a request for cancellation tracking."""
    if request_id:
        _cancellation_flags[request_id] = threading.Event()

def cancel_request(request_id: str):
    """Cancel a request by setting its flag."""
    if request_id in _cancellation_flags:
        _cancellation_flags[request_id].set()
        return True
    return False

def unregister_cancellation(request_id: str):
    """Remove cancellation tracking for a request."""
    _cancellation_flags.pop(request_id, None)

def is_cancelled(request_id: str) -> bool:
    """Check if a request is cancelled without raising."""
    if request_id:
        return _cancellation_flags.get(request_id, threading.Event()).is_set()
    return False


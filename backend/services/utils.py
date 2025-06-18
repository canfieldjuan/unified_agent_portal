# backend/services/utils.py

import structlog
from functools import wraps
from fastapi.responses import JSONResponse
from typing import Callable

logger = structlog.get_logger()

def handle_errors(func: Callable) -> Callable:
    """
    Decorator to gracefully handle exceptions in API routes and return a consistent
    JSON error response. Logs the error for debugging.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in API route '{func.__name__}'", exc_info=True, error_type=type(e).__name__, error_message=str(e))
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "response": f"An internal server error occurred: {type(e).__name__}",
                    "detail": str(e),
                    "model": "Error Handler",
                    "provider": "System",
                    "cost": 0.0,
                    "response_time": 0.0,
                    "reasoning": "Error occurred during API processing."
                }
            )
    return wrapper
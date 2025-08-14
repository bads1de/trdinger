from datetime import datetime
from typing import Any, Dict, Optional

def now_iso() -> str:
    return datetime.now().isoformat()

def error_response(message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None, context: Optional[str] = None) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "success": False,
        "message": message,
        "timestamp": now_iso(),
    }
    if error_code:
        response["error_code"] = error_code
    if details:
        response["details"] = details
    if context:
        response["context"] = context
    return response

def api_response(success: bool, message: str, status: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response: Dict[str, Any] = {"success": success, "message": message}
    if status:
        response["status"] = status
    if data is not None:
        response["data"] = data
    response["timestamp"] = now_iso()
    return response

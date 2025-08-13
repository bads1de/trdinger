from datetime import datetime
from typing import Any, Dict, Optional

def now_iso() -> str:
    return datetime.now().isoformat()

def make_error_response(message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None, context: Optional[str] = None) -> Dict[str, Any]:
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

def make_api_response(success: bool, message: str, status: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response: Dict[str, Any] = {"success": success, "message": message}
    if status:
        response["status"] = status
    if data is not None:
        response["data"] = data
    response["timestamp"] = now_iso()
    return response

def make_list_response(success: bool, message: str, items: list, status: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response_data = {"items": items}
    if metadata:
        response_data.update(metadata)
    return make_api_response(success=success, message=message, status=status, data=response_data)

def make_success_response(data: Optional[Dict[str, Any]] = None, message: str = "成功", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response_data = data
    if metadata:
        if isinstance(data, dict):
            response_data = {**data, "metadata": metadata}
        else:
            response_data = {"data": data, "metadata": metadata}
    return make_api_response(success=True, message=message, data=response_data)
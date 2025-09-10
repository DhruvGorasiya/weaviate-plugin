from typing import Any, Dict, List, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)

def format_search_results(results: List[Dict[str, Any]], include_metadata: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results or []:
        item = {
            "uuid": r.get("uuid"),
            "properties": r.get("properties") or {},
        }
        if include_metadata:
            meta = r.get("metadata") or {}
            # Optional: normalize
            # If distance present, convert to relevance score in [0,1]
            if "distance" in meta and "relevance" not in meta:
                try:
                    d = float(meta["distance"])
                    meta["relevance"] = max(0.0, min(1.0, 1.0 - d))
                except Exception:
                    pass
            item["metadata"] = meta
        out.append(item)
    return out

def create_error_response(error_message: str, error_code: str = "WEAVIATE_ERROR", details: Any = None) -> Dict[str, Any]:
    resp = {
        "success": False,
        "error": error_message,
        "error_code": error_code,
    }
    if details is not None:
        resp["details"] = details
    return resp

def create_success_response(data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "message": message,
    }

def safe_json_parse(value: Any, default: Any = None) -> Any:
    """
    Accepts dict/list and returns as-is. If string, tries json.loads.
    Returns `default` on failure.
    """
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON: %r", value)
            return default
    return default

def _parse_number(val: str) -> Optional[Union[int, float]]:
    try:
        if isinstance(val, (int, float)):
            return val
        s = str(val).strip()
        if s.lower() in ("nan", "inf", "-inf"):
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        return float(s)
    except Exception:
        return None

def extract_properties_from_text(text: str) -> Dict[str, Any]:
    """
    Parses lines like:
      name: Alice
      active: true
      count: 12
      score: -0.45
    """
    props: Dict[str, Any] = {}
    if not text:
        return props
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if not key:
            continue
        low = val.lower()
        if low in ("true", "false"):
            props[key] = (low == "true")
            continue
        num = _parse_number(val)
        if num is not None:
            props[key] = num
        else:
            props[key] = val
    return props

def _value_field_for_python(value: Any) -> str:
    """
    Choose correct Weaviate filter value key based on Python type.
    """
    if isinstance(value, bool):
        return "valueBoolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "valueInt"
    if isinstance(value, float):
        return "valueNumber"
    # Fallback to text
    return "valueText"

def build_where_filter(conditions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple Weaviate where-filter.
    Input:
      {"status": "published", "views": 10, "featured": True}
    Output:
      {"operator": "And", "operands": [
         {"path":["status"], "operator":"Equal", "valueText":"published"},
         {"path":["views"],  "operator":"Equal", "valueInt":10},
         {"path":["featured"],"operator":"Equal","valueBoolean":true}
      ]}
    """
    if not conditions:
        return {}

    def _single_cond(k: str, v: Any) -> Dict[str, Any]:
        val_key = _value_field_for_python(v)
        return {"path": [k], "operator": "Equal", val_key: v}

    items = list(conditions.items())
    if len(items) == 1:
        k, v = items[0]
        return _single_cond(k, v)

    return {
        "operator": "And",
        "operands": [_single_cond(k, v) for k, v in items],
    }

def csv_or_list_to_list(value: Any) -> Optional[List[str]]:
    """
    Accept CSV string or list -> normalized list[str] (or None).
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]
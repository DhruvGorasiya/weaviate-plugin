from typing import Any, Dict, List, Union
import re
import json

VALID_DATA_TYPES = {
    "text", "int", "number", "boolean", "date",
    "uuid", "geoCoordinates", "blob"
}

def validate_weaviate_url(url: str) -> bool:
    pattern = r'^https?://[a-zA-Z0-9.-]+(:\d+)?(/.*)?$'
    return bool(url and re.match(pattern, url))

def validate_api_key(api_key: str) -> bool:
    return bool(api_key and api_key.strip())

def validate_collection_name(name: str) -> bool:
    # v4 allows lowercase and underscores
    pattern = r'^[A-Za-z_][A-Za-z0-9_]*$'
    return bool(name and re.match(pattern, name))

def validate_properties(properties: List[Dict[str, Any]]) -> bool:
    if not isinstance(properties, list) or not properties:
        return False
    for prop in properties:
        if not isinstance(prop, dict):
            return False
        n, t = prop.get("name"), prop.get("data_type")
        if not (isinstance(n, str) and n and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', n)):
            return False
        if not (isinstance(t, str) and t in VALID_DATA_TYPES):
            return False
    return True

def validate_vector(vector: List[Union[int, float]], expected_dim: int = None) -> bool:
    if not isinstance(vector, list) or not vector:
        return False
    if not all(isinstance(x, (int, float)) for x in vector):
        return False
    if expected_dim and len(vector) != expected_dim:
        return False
    return True

def validate_where_filter(where_filter: Dict[str, Any]) -> bool:
    if not isinstance(where_filter, dict):
        return False
    try:
        json.dumps(where_filter)
    except (TypeError, ValueError):
        return False
    # minimal structural check
    if "operator" in where_filter and where_filter["operator"] in {"And", "Or", "Not"}:
        return "operands" in where_filter
    if "path" in where_filter and "operator" in where_filter:
        return True
    return False

def validate_limit(limit: Union[int, str], max_limit: int = 1000) -> bool:
    try:
        val = int(limit)
        return 1 <= val <= max_limit
    except (TypeError, ValueError):
        return False

def validate_alpha(alpha: Union[float, int, str]) -> bool:
    try:
        val = float(alpha)
        return 0.0 <= val <= 1.0
    except (TypeError, ValueError):
        return False
from typing import Any, Dict, List, Optional
import re
import json

def validate_weaviate_url(url: str) -> bool:
    pattern = r'^https?://[a-zA-Z0-9.-]+(:\d+)?(/.*)?$'
    return bool(re.match(pattern, url))

def validate_api_key(api_key: str) -> bool:
    return len(api_key.strip()) > 0

def validate_collection_name(name: str) -> bool:
    pattern = r'^[A-Z][a-zA-Z0-9]*$'
    return bool(re.match(pattern, name))

def validate_properties(properties: List[Dict[str, Any]]) -> bool:
    if not isinstance(properties, list) or len(properties) == 0:
        return False
    
    for prop in properties:
        if not isinstance(prop, dict):
            return False
        if 'name' not in prop or 'data_type' not in prop:
            return False
        if not isinstance(prop['name'], str) or not isinstance(prop['data_type'], str):
            return False
    
    return True

def validate_vector(vector: List[float]) -> bool:
    if not isinstance(vector, list):
        return False
    if len(vector) == 0:
        return False
    return all(isinstance(x, (int, float)) for x in vector)

def validate_where_filter(where_filter: Dict[str, Any]) -> bool:
    try:
        json.dumps(where_filter)
        return True
    except (TypeError, ValueError):
        return False

def validate_limit(limit: int) -> bool:
    return isinstance(limit, int) and 1 <= limit <= 1000

def validate_alpha(alpha: float) -> bool:
    return isinstance(alpha, (int, float)) and 0.0 <= alpha <= 1.0

from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

def format_search_results(results: List[Dict[str, Any]], include_metadata: bool = True) -> List[Dict[str, Any]]:
    formatted_results = []
    for result in results:
        formatted_result = {
            'uuid': result.get('uuid'),
            'properties': result.get('properties', {}),
        }
        if include_metadata and 'metadata' in result:
            formatted_result['metadata'] = result['metadata']
        formatted_results.append(formatted_result)
    return formatted_results

def create_error_response(error_message: str, error_code: str = "WEAVIATE_ERROR") -> Dict[str, Any]:
    return {
        'success': False,
        'error': error_message,
        'error_code': error_code
    }

def create_success_response(data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
    return {
        'success': True,
        'data': data,
        'message': message
    }

def safe_json_parse(json_string: str, default: Any = None) -> Any:
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {json_string}")
        return default

def extract_properties_from_text(text: str) -> Dict[str, Any]:
    properties = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if value.lower() in ['true', 'false']:
                properties[key] = value.lower() == 'true'
            elif value.isdigit():
                properties[key] = int(value)
            elif value.replace('.', '').isdigit():
                properties[key] = float(value)
            else:
                properties[key] = value
    
    return properties

def build_where_filter(conditions: Dict[str, Any]) -> Dict[str, Any]:
    if not conditions:
        return {}
    
    if len(conditions) == 1:
        key, value = next(iter(conditions.items()))
        return {
            "path": [key],
            "operator": "Equal",
            "valueText": str(value)
        }
    
    return {
        "operator": "And",
        "operands": [
            {
                "path": [key],
                "operator": "Equal", 
                "valueText": str(value)
            }
            for key, value in conditions.items()
        ]
    }

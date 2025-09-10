from collections.abc import Generator
from typing import Any, List, Optional
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

def _to_list(value) -> Optional[List[str]]:
    """Accept CSV string or list, return a clean list of strings or None."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]

class KeywordSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = (tool_parameters.get('collection_name') or '').strip()
            query = (tool_parameters.get('query') or '').strip()
            limit_raw = tool_parameters.get('limit', 10)
            where_filter_raw = tool_parameters.get('where_filter')
            return_properties_in = tool_parameters.get('return_properties')
            search_properties_in = tool_parameters.get('search_properties')

            # Basic validation
            if not collection_name:
                yield self.create_json_message(create_error_response("Collection name is required"))
                return
            if not query:
                yield self.create_json_message(create_error_response("Query is required"))
                return

            # Parse limit
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_json_message(create_error_response("Limit must be an integer between 1 and 1000"))
                return
            if not validate_limit(limit):
                yield self.create_json_message(create_error_response("Limit must be between 1 and 1000"))
                return

            # Where filter: accept JSON string or dict
            where_filter = None
            if isinstance(where_filter_raw, str):
                s = where_filter_raw.strip()
                if s:
                    where_filter = safe_json_parse(s)
                    if where_filter is None or not validate_where_filter(where_filter):
                        yield self.create_json_message(create_error_response("Invalid where filter. Provide valid JSON"))
                        return
            elif isinstance(where_filter_raw, dict):
                where_filter = where_filter_raw
                if not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response("Invalid where filter JSON"))
                    return

            # Properties parsing (CSV or list)
            return_properties = _to_list(return_properties_in)
            search_properties = _to_list(search_properties_in)

            # Connect
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds['url'],
                api_key=creds.get('api_key'),
                timeout=60
            )

            try:
                results = client.text_search(
                    class_name=collection_name,
                    query=query,
                    limit=limit,
                    where_filter=where_filter,
                    return_properties=return_properties,
                    search_properties=search_properties
                )

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': [],
                            'count': 0,
                            'collection': collection_name,
                            'query': query,
                            'search_type': 'keyword'
                        },
                        message="No results found"
                    ))
                    return

                yield self.create_json_message(create_success_response(
                    data={
                        'results': results,
                        'count': len(results),
                        'collection': collection_name,
                        'query': query,
                        'search_type': 'keyword'
                    },
                    message=f"Found {len(results)} documents matching keywords"
                ))

            except Exception as e:
                logger.exception("Keyword search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))
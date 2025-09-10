from collections.abc import Generator
from typing import Any
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class KeywordSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = tool_parameters.get('collection_name', '').strip()
            query = tool_parameters.get('query', '').strip()
            limit = tool_parameters.get('limit', 10)
            where_filter_str = tool_parameters.get('where_filter', '').strip()
            return_properties_str = tool_parameters.get('return_properties', '').strip()
            search_properties_str = tool_parameters.get('search_properties', '').strip()
            
            if not collection_name:
                yield self.create_text_message("Error: Collection name is required")
                return
            
            if not query:
                yield self.create_text_message("Error: Query is required")
                return
            
            if not validate_limit(limit):
                yield self.create_text_message("Error: Limit must be between 1 and 1000")
                return
            
            where_filter = None
            if where_filter_str:
                where_filter = safe_json_parse(where_filter_str)
                if where_filter is None or not validate_where_filter(where_filter):
                    yield self.create_text_message("Error: Invalid where filter format. Use valid JSON")
                    return
            
            return_properties = None
            if return_properties_str:
                return_properties = [prop.strip() for prop in return_properties_str.split(',') if prop.strip()]
            
            search_properties = None
            if search_properties_str:
                search_properties = [prop.strip() for prop in search_properties_str.split(',') if prop.strip()]
            
            credentials = self.runtime.credentials
            client = WeaviateClient(
                url=credentials['url'],
                api_key=credentials.get('api_key'),
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
                    yield self.create_text_message("No results found")
                    return
                
                response = create_success_response(
                    data={
                        'results': results,
                        'count': len(results),
                        'collection': collection_name,
                        'query': query,
                        'search_type': 'keyword'
                    },
                    message=f"Found {len(results)} documents matching keywords"
                )
                
                yield self.create_json_message(response)
                
            except Exception as e:
                logger.error(f"Keyword search error: {str(e)}")
                yield self.create_text_message(f"Search failed: {str(e)}")
            finally:
                client.disconnect()
                
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield self.create_text_message(f"Tool execution failed: {str(e)}")

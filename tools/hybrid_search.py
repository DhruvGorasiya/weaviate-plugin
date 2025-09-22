from collections.abc import Generator
from typing import Any
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_limit, validate_where_filter, validate_alpha
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class HybridSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = (tool_parameters.get('collection_name') or '').strip()
            query = (tool_parameters.get('query') or '').strip()
            where_filter_str = (tool_parameters.get('where_filter') or '').strip()
            return_properties_str = (tool_parameters.get('return_properties') or '').strip()

            if not collection_name:
                yield self.create_text_message("Error: Collection name is required")
                return

            # Check that at least one of query or query_vector is provided
            qv_raw = tool_parameters.get('query_vector', '')
            if isinstance(qv_raw, str):
                qv_raw = qv_raw.strip()
            
            if not query and not qv_raw:
                yield self.create_text_message("Error: Either query text or query vector is required")
                return

            # alpha - safe cast from string
            alpha_raw = tool_parameters.get('alpha', 0.7)
            try:
                alpha = float(alpha_raw)
            except (TypeError, ValueError):
                yield self.create_text_message("Error: Alpha must be a number between 0.0 and 1.0")
                return
            if not validate_alpha(alpha):
                yield self.create_text_message("Error: Alpha must be between 0.0 and 1.0")
                return

            # limit - safe cast from string
            limit_raw = tool_parameters.get('limit', 10)
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_text_message("Error: Limit must be an integer between 1 and 1000")
                return
            if not validate_limit(limit):
                yield self.create_text_message("Error: Limit must be between 1 and 1000")
                return

            # query_vector - accept JSON array OR CSV string
            query_vector = None
            if qv_raw:
                if qv_raw.startswith('['):
                    query_vector = safe_json_parse(qv_raw)
                else:
                    try:
                        query_vector = [float(x.strip()) for x in qv_raw.split(',') if x.strip()]
                    except ValueError:
                        yield self.create_text_message("Error: Invalid query vector. Use JSON array or comma-separated numbers")
                        return
                if not validate_vector(query_vector):
                    yield self.create_text_message("Error: Query vector must be a non-empty list of numbers")
                    return

            where_filter = None
            if where_filter_str:
                where_filter = safe_json_parse(where_filter_str)
                if where_filter is None or not validate_where_filter(where_filter):
                    yield self.create_text_message("Error: Invalid where filter format. Use valid JSON")
                    return

            return_properties = None
            if return_properties_str:
                return_properties = [p.strip() for p in return_properties_str.split(',') if p.strip()]

            # credentials
            creds = self.runtime.credentials
            client = WeaviateClient(url=creds['url'], api_key=creds.get('api_key'), timeout=60)

            try:
                results = client.hybrid_search(
                    class_name=collection_name,
                    query=query,
                    query_vector=query_vector,
                    alpha=alpha,
                    limit=limit,
                    where_filter=where_filter,
                    return_properties=return_properties
                )

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={'results': [], 'count': 0, 'collection': collection_name, 'alpha': alpha, 'query': query},
                        message="No results found"
                    ))
                    return

                yield self.create_json_message(create_success_response(
                    data={'results': results, 'count': len(results), 'collection': collection_name, 'alpha': alpha, 'query': query},
                    message=f"Found {len(results)} results using hybrid search"
                ))

            except Exception as e:
                logger.exception("Hybrid search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))

            finally:
                client.disconnect()

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))

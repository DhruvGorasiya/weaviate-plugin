from collections.abc import Generator
from typing import Any
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class VectorSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = (tool_parameters.get('collection_name') or '').strip()
            qv_raw = (tool_parameters.get('query_vector') or '').strip()
            limit_raw = tool_parameters.get('limit', 10)
            where_filter_str = (tool_parameters.get('where_filter') or '').strip()
            return_properties_str = (tool_parameters.get('return_properties') or '').strip()

            if not collection_name:
                yield self.create_json_message(create_error_response("Collection name is required"))
                return
            if not qv_raw:
                yield self.create_json_message(create_error_response("Query vector is required"))
                return

            # Parse limit safely
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_json_message(create_error_response("Limit must be an integer between 1 and 1000"))
                return
            if not validate_limit(limit):
                yield self.create_json_message(create_error_response("Limit must be between 1 and 1000"))
                return

            # Accept JSON array OR CSV string for vectors
            if qv_raw.startswith('['):
                query_vector = safe_json_parse(qv_raw)
            else:
                try:
                    query_vector = [float(x.strip()) for x in qv_raw.split(',') if x.strip()]
                except ValueError:
                    yield self.create_json_message(create_error_response(
                        "Invalid query vector. Use a JSON array or comma-separated numbers"
                    ))
                    return

            if not validate_vector(query_vector):
                yield self.create_json_message(create_error_response(
                    "Query vector must be a non-empty list of numbers"
                ))
                return

            where_filter = None
            if where_filter_str:
                where_filter = safe_json_parse(where_filter_str)
                if where_filter is None or not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response(
                        "Invalid where filter format. Provide valid JSON"
                    ))
                    return

            return_properties = None
            if return_properties_str:
                return_properties = [p.strip() for p in return_properties_str.split(',') if p.strip()]

            # Connect
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds['url'],
                api_key=creds.get('api_key'),
                timeout=60
            )

            try:
                results = client.vector_search(
                    class_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    where_filter=where_filter,
                    return_properties=return_properties
                )

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={'results': [], 'count': 0, 'collection': collection_name},
                        message="No results found"
                    ))
                    return

                yield self.create_json_message(create_success_response(
                    data={'results': results, 'count': len(results), 'collection': collection_name},
                    message=f"Found {len(results)} similar vectors"
                ))

            except Exception as e:
                logger.exception("Vector search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))
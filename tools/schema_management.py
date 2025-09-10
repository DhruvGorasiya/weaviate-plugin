from collections.abc import Generator
from typing import Any
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_collection_name, validate_properties
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class SchemaManagementTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            operation = tool_parameters.get('operation', '').strip().lower()
            collection_name = tool_parameters.get('collection_name', '').strip()
            properties_str = tool_parameters.get('properties', '').strip()
            vectorizer = tool_parameters.get('vectorizer', '').strip()
            
            if operation != 'list_collections' and not collection_name:
                yield self.create_text_message("Error: Collection name is required for this operation")
                return
            
            if collection_name and not validate_collection_name(collection_name):
                yield self.create_text_message("Error: Invalid collection name format. Use PascalCase (e.g., 'MyCollection')")
                return
            
            credentials = self.runtime.credentials
            client = WeaviateClient(
                url=credentials['url'],
                api_key=credentials.get('api_key'),
                timeout=60
            )
            
            try:
                if operation == 'list_collections':
                    collections = client.list_collections()
                    response = create_success_response(
                        data={'collections': collections},
                        message=f"Found {len(collections)} collections"
                    )
                    yield self.create_json_message(response)
                    return
                
                if operation == 'create_collection':
                    if not properties_str:
                        yield self.create_text_message("Error: Properties are required for collection creation")
                        return
                    
                    properties = safe_json_parse(properties_str)
                    if properties is None or not validate_properties(properties):
                        yield self.create_text_message("Error: Invalid properties format. Use valid JSON array of property objects")
                        return
                    
                    vectorizer_config = vectorizer if vectorizer else None
                    
                    success = client.create_collection(
                        class_name=collection_name,
                        properties=properties,
                        vectorizer=vectorizer_config
                    )
                    
                    if success:
                        response = create_success_response(
                            data={'collection_name': collection_name},
                            message=f"Collection '{collection_name}' created successfully"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message(f"Error: Failed to create collection '{collection_name}'")
                
                elif operation == 'delete_collection':
                    success = client.delete_collection(collection_name)
                    if success:
                        response = create_success_response(
                            data={'collection_name': collection_name},
                            message=f"Collection '{collection_name}' deleted successfully"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message(f"Error: Failed to delete collection '{collection_name}'")
                
                elif operation == 'get_schema':
                    schema = client.get_collection_schema(collection_name)
                    if schema:
                        response = create_success_response(
                            data={'schema': schema},
                            message=f"Schema retrieved for collection '{collection_name}'"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message(f"Error: Failed to get schema for collection '{collection_name}'")
                
                elif operation == 'get_stats':
                    stats = client.get_collection_stats(collection_name)
                    if stats:
                        response = create_success_response(
                            data={'stats': stats},
                            message=f"Statistics retrieved for collection '{collection_name}'"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message(f"Error: Failed to get stats for collection '{collection_name}'")
                
                else:
                    yield self.create_text_message(f"Error: Unknown operation '{operation}'")
                
            except Exception as e:
                logger.error(f"Schema management error: {str(e)}")
                yield self.create_text_message(f"Operation failed: {str(e)}")
            finally:
                client.disconnect()
                
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield self.create_text_message(f"Tool execution failed: {str(e)}")

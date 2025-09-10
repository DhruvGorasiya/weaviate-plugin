from collections.abc import Generator
from typing import Any
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class DataManagementTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            operation = tool_parameters.get('operation', '').strip().lower()
            collection_name = tool_parameters.get('collection_name', '').strip()
            object_data_str = tool_parameters.get('object_data', '').strip()
            object_uuid = tool_parameters.get('object_uuid', '').strip()
            return_properties_str = tool_parameters.get('return_properties', '').strip()
            
            if not collection_name and operation != 'list_collections':
                yield self.create_text_message("Error: Collection name is required")
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
                
                if operation == 'insert':
                    if not object_data_str:
                        yield self.create_text_message("Error: Object data is required for insert operation")
                        return
                    
                    object_data = safe_json_parse(object_data_str)
                    if object_data is None:
                        yield self.create_text_message("Error: Invalid object data format. Use valid JSON")
                        return
                    
                    if isinstance(object_data, dict):
                        object_data = [object_data]
                    elif not isinstance(object_data, list):
                        yield self.create_text_message("Error: Object data must be a JSON object or array of objects")
                        return
                    
                    uuids = client.insert_objects(collection_name, object_data)
                    if uuids:
                        response = create_success_response(
                            data={'inserted_uuids': uuids, 'count': len(uuids)},
                            message=f"Successfully inserted {len(uuids)} objects"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message("Error: Failed to insert objects")
                
                elif operation == 'update':
                    if not object_uuid:
                        yield self.create_text_message("Error: Object UUID is required for update operation")
                        return
                    
                    if not object_data_str:
                        yield self.create_text_message("Error: Object data is required for update operation")
                        return
                    
                    object_data = safe_json_parse(object_data_str)
                    if object_data is None or not isinstance(object_data, dict):
                        yield self.create_text_message("Error: Invalid object data format. Use valid JSON object")
                        return
                    
                    success = client.update_object(collection_name, object_uuid, object_data)
                    if success:
                        response = create_success_response(
                            data={'uuid': object_uuid},
                            message="Object updated successfully"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message("Error: Failed to update object")
                
                elif operation == 'delete':
                    if not object_uuid:
                        yield self.create_text_message("Error: Object UUID is required for delete operation")
                        return
                    
                    success = client.delete_object(collection_name, object_uuid)
                    if success:
                        response = create_success_response(
                            data={'uuid': object_uuid},
                            message="Object deleted successfully"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message("Error: Failed to delete object")
                
                elif operation == 'get':
                    if not object_uuid:
                        yield self.create_text_message("Error: Object UUID is required for get operation")
                        return
                    
                    return_properties = None
                    if return_properties_str:
                        return_properties = [prop.strip() for prop in return_properties_str.split(',') if prop.strip()]
                    
                    result = client.get_object(collection_name, object_uuid, return_properties)
                    if result:
                        response = create_success_response(
                            data={'object': result},
                            message="Object retrieved successfully"
                        )
                        yield self.create_json_message(response)
                    else:
                        yield self.create_text_message("Object not found")
                
                else:
                    yield self.create_text_message(f"Error: Unknown operation '{operation}'")
                
            except Exception as e:
                logger.error(f"Data management error: {str(e)}")
                yield self.create_text_message(f"Operation failed: {str(e)}")
            finally:
                client.disconnect()
                
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield self.create_text_message(f"Tool execution failed: {str(e)}")

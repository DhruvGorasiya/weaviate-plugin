from collections.abc import Generator
from typing import Any
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

_ALLOWED_OPS = {"list_collections", "insert", "update", "delete", "get"}

class DataManagementTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            # normalize inputs
            op_raw = tool_parameters.get("operation", "")
            operation = (op_raw or "").strip().lower()

            collection_name = (tool_parameters.get("collection_name") or "").strip()
            object_data_str = (tool_parameters.get("object_data") or "").strip()
            object_uuid = (tool_parameters.get("object_uuid") or "").strip()
            return_properties_str = (tool_parameters.get("return_properties") or "").strip()

            if operation not in _ALLOWED_OPS:
                yield self.create_json_message(create_error_response(
                    f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}"
                ))
                return

            if operation != "list_collections" and not collection_name:
                yield self.create_json_message(create_error_response(
                    "Collection name is required"
                ))
                return

            # credentials / client
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds["url"],
                api_key=creds.get("api_key"),
                timeout=60,
            )

            try:
                # ---- list_collections ----
                if operation == "list_collections":
                    collections = client.list_collections()
                    yield self.create_json_message(create_success_response(
                        data={"collections": collections, "count": len(collections)},
                        message=f"Found {len(collections)} collections"
                    ))
                    return

                # parse optional projection
                return_properties = None
                if return_properties_str:
                    return_properties = [p.strip() for p in return_properties_str.split(",") if p.strip()]

                # ---- insert ----
                if operation == "insert":
                    if not object_data_str:
                        yield self.create_json_message(create_error_response(
                            "Object data is required for insert operation"
                        ))
                        return

                    object_data = safe_json_parse(object_data_str)
                    if object_data is None:
                        yield self.create_json_message(create_error_response(
                            "Invalid object data format. Provide valid JSON (object or array of objects)"
                        ))
                        return

                    # allow single dict or list of dicts
                    if isinstance(object_data, dict):
                        payload = [object_data]
                    elif isinstance(object_data, list):
                        payload = object_data
                    else:
                        yield self.create_json_message(create_error_response(
                            "Object data must be a JSON object or an array of objects"
                        ))
                        return

                    uuids = client.insert_objects(collection_name, payload)
                    if uuids:
                        yield self.create_json_message(create_success_response(
                            data={"inserted_uuids": uuids, "count": len(uuids), "collection": collection_name},
                            message=f"Successfully inserted {len(uuids)} objects"
                        ))
                    else:
                        yield self.create_json_message(create_error_response("Failed to insert objects"))
                    return

                # ---- update ----
                if operation == "update":
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID is required for update operation"
                        ))
                        return
                    if not object_data_str:
                        yield self.create_json_message(create_error_response(
                            "Object data is required for update operation"
                        ))
                        return

                    object_data = safe_json_parse(object_data_str)
                    if not isinstance(object_data, dict):
                        yield self.create_json_message(create_error_response(
                            "Invalid object data format. Provide a JSON object"
                        ))
                        return

                    success = client.update_object(collection_name, object_uuid, object_data)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"uuid": object_uuid, "collection": collection_name},
                            message="Object updated successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response("Failed to update object"))
                    return

                # ---- delete ----
                if operation == "delete":
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID is required for delete operation"
                        ))
                        return

                    success = client.delete_object(collection_name, object_uuid)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"uuid": object_uuid, "collection": collection_name},
                            message="Object deleted successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response("Failed to delete object"))
                    return

                # ---- get ----
                if operation == "get":
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID is required for get operation"
                        ))
                        return

                    result = client.get_object(collection_name, object_uuid, return_properties)
                    if result:
                        yield self.create_json_message(create_success_response(
                            data={"object": result, "collection": collection_name},
                            message="Object retrieved successfully"
                        ))
                    else:
                        # still JSON, for consistent shape
                        yield self.create_json_message(create_success_response(
                            data={"object": None, "collection": collection_name},
                            message="Object not found"
                        ))
                    return

                # fallback (shouldn’t happen)
                yield self.create_json_message(create_error_response(
                    f"Unsupported operation '{operation}'"
                ))

            except Exception as e:
                logger.exception("Data management error")
                yield self.create_json_message(create_error_response(f"Operation failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))
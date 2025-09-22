from collections.abc import Generator
from typing import Any, List, Dict, Optional
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

_ALLOWED_OPS = {"list_collections", "insert", "update", "delete", "get", "list_objects"}

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

                # ---- list_objects ----
                if operation == "list_objects":
                    where_filter = None
                    if tool_parameters.get("where_filter"):
                        where_filter = safe_json_parse(tool_parameters.get("where_filter"))
                        if not isinstance(where_filter, dict):
                            yield self.create_json_message(create_error_response(
                                "Invalid where_filter format. Provide valid JSON object"
                            ))
                            return
                    
                    limit = tool_parameters.get("limit", 100)
                    tenant = tool_parameters.get("tenant")
                    include_vector = tool_parameters.get("include_vector", False)
                    return_additional = None
                    if tool_parameters.get("return_additional"):
                        return_additional = [f.strip() for f in tool_parameters.get("return_additional").split(",") if f.strip()]
                    
                    objects = client.list_objects(
                        class_name=collection_name,
                        where_filter=where_filter,
                        limit=limit,
                        tenant=tenant,
                        return_properties=return_properties,
                        include_vector=include_vector,
                        return_additional=return_additional
                    )
                    
                    yield self.create_json_message(create_success_response(
                        data={"objects": objects, "count": len(objects), "collection": collection_name},
                        message=f"Retrieved {len(objects)} objects from collection '{collection_name}'"
                    ))
                    return

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

                    mode = tool_parameters.get("mode", "single")
                    tenant = tool_parameters.get("tenant")
                    batch_size = tool_parameters.get("batch_size", 100)
                    
                    try:
                        if mode == "batch":
                            result = client.insert_objects_batch(
                                class_name=collection_name,
                                objects=payload,
                                tenant=tenant,
                                batch_size=batch_size
                            )
                            
                            yield self.create_json_message(create_success_response(
                                data={
                                    "inserted_count": result["inserted_count"],
                                    "failed_count": result["failed_count"],
                                    "errors": result["errors"],
                                    "collection": collection_name
                                },
                                message=f"Batch insert completed: {result['inserted_count']} inserted, {result['failed_count']} failed"
                            ))
                        else:
                            # Single mode (existing logic)
                            uuids = client.insert_objects(collection_name, payload)
                            if uuids:
                                yield self.create_json_message(create_success_response(
                                    data={"inserted_uuids": uuids, "count": len(uuids), "collection": collection_name},
                                    message=f"Successfully inserted {len(uuids)} objects"
                                ))
                            else:
                                yield self.create_json_message(create_error_response("Failed to insert objects - no UUIDs returned"))
                    except Exception as insert_error:
                        yield self.create_json_message(create_error_response(
                            f"Insert failed: {str(insert_error)}"
                        ))
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

                    update_mode = tool_parameters.get("update_mode", "merge")
                    tenant = tool_parameters.get("tenant")
                    
                    try:
                        if update_mode == "replace":
                            success = client.replace_object(
                                class_name=collection_name,
                                uuid=object_uuid,
                                properties=object_data,
                                tenant=tenant
                            )
                        elif update_mode == "vector_only":
                            vector = object_data.get("vector")
                            if not vector:
                                yield self.create_json_message(create_error_response(
                                    "Vector data is required for vector_only update mode"
                                ))
                                return
                            success = client.update_vector(
                                class_name=collection_name,
                                uuid=object_uuid,
                                vector=vector,
                                tenant=tenant
                            )
                        else:  # merge mode (default)
                            success = client.update_object(collection_name, object_uuid, object_data)
                        
                        if success:
                            yield self.create_json_message(create_success_response(
                                data={"uuid": object_uuid, "collection": collection_name, "mode": update_mode},
                                message=f"Object updated successfully using {update_mode} mode"
                            ))
                        else:
                            yield self.create_json_message(create_error_response(f"Failed to update object using {update_mode} mode"))
                    except Exception as update_error:
                        yield self.create_json_message(create_error_response(
                            f"Update failed: {str(update_error)}"
                        ))
                    return

                # ---- delete ----
                if operation == "delete":
                    tenant = tool_parameters.get("tenant")
                    dry_run = tool_parameters.get("dry_run", False)
                    
                    # Check if we have a where filter instead of UUID
                    where_filter_str = tool_parameters.get("where_filter")
                    if where_filter_str:
                        where_filter = safe_json_parse(where_filter_str)
                        if not isinstance(where_filter, dict):
                            yield self.create_json_message(create_error_response(
                                "Invalid where_filter format. Provide valid JSON object"
                            ))
                            return
                        
                        result = client.delete_by_filter(
                            class_name=collection_name,
                            where_filter=where_filter,
                            tenant=tenant,
                            dry_run=dry_run
                        )
                        
                        if dry_run:
                            yield self.create_json_message(create_success_response(
                                data=result,
                                message=f"Dry run: would delete {result['would_delete_count']} objects"
                            ))
                        else:
                            yield self.create_json_message(create_success_response(
                                data=result,
                                message=f"Deleted {result['deleted_count']} objects, {result['failed_count']} failed"
                            ))
                        return
                    
                    # Original UUID-based delete
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID or where_filter is required for delete operation"
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

                    tenant = tool_parameters.get("tenant")
                    include_vector = tool_parameters.get("include_vector", False)
                    return_additional = None
                    if tool_parameters.get("return_additional"):
                        return_additional = [f.strip() for f in tool_parameters.get("return_additional").split(",") if f.strip()]

                    result = client.get_object(
                        class_name=collection_name, 
                        uuid=object_uuid, 
                        return_properties=return_properties
                    )
                    
                    if result:
                        yield self.create_json_message(create_success_response(
                            data={"object": result, "collection": collection_name},
                            message="Object retrieved successfully"
                        ))
                    else:
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
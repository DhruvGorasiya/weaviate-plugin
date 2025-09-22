from collections.abc import Generator
from typing import Any, List, Dict, Optional
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_collection_name, validate_properties
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

_ALLOWED_OPS = {
    "list_collections",
    "create_collection",
    "delete_collection",
    "get_schema",
    "get_stats",
    "exists",
    "add_property",
    "update_config",
}

# Optional: restrict vectorizers we recognize; None/self_provided is default
_ALLOWED_VECTORIZERS = {"self_provided", "text2vec-openai", "text2vec-transformers"}

class SchemaManagementTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            op_raw = tool_parameters.get("operation", "")
            operation = (op_raw or "").strip().lower()

            collection_name = (tool_parameters.get("collection_name") or "").strip()
            properties_raw = tool_parameters.get("properties")
            vectorizer_raw = (tool_parameters.get("vectorizer") or "").strip()

            if operation not in _ALLOWED_OPS:
                yield self.create_json_message(create_error_response(
                    f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}"
                ))
                return

            if operation != "list_collections" and not collection_name:
                yield self.create_json_message(create_error_response(
                    "Collection name is required for this operation"
                ))
                return

            if collection_name and not validate_collection_name(collection_name):
                yield self.create_json_message(create_error_response(
                    "Invalid collection name. Use letters, digits, and underscores, starting with a letter or underscore (e.g., my_collection, UserProfiles)"
                ))
                return

            # Normalize vectorizer
            vectorizer = None
            if vectorizer_raw:
                v = vectorizer_raw.strip()
                # Map common aliases
                if v in {"none", "self", "self_provided"}:
                    vectorizer = None  # default to self-provided in client
                elif v in _ALLOWED_VECTORIZERS:
                    vectorizer = v
                else:
                    yield self.create_json_message(create_error_response(
                        f"Unsupported vectorizer '{v}'. Allowed: {sorted(_ALLOWED_VECTORIZERS)}"
                    ))
                    return

            # Connect
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds["url"],
                api_key=creds.get("api_key"),
                timeout=60,
            )

            try:
                # ---- list_collections ----
                if operation == "list_collections":
                    cols = client.list_collections()
                    yield self.create_json_message(create_success_response(
                        data={"collections": cols, "count": len(cols)},
                        message=f"Found {len(cols)} collections"
                    ))
                    return

                # ---- create_collection ----
                if operation == "create_collection":
                    if not properties_raw:
                        yield self.create_json_message(create_error_response(
                            "Properties are required for collection creation"
                        ))
                        return

                    props = safe_json_parse(properties_raw)
                    # Allow a single object or an array
                    if isinstance(props, dict):
                        props = [props]

                    if not validate_properties(props):
                        yield self.create_json_message(create_error_response(
                            "Invalid properties. Provide a JSON array of property objects with 'name' and 'data_type'"
                        ))
                        return

                    # Get optional parameters
                    description = (tool_parameters.get("description") or "").strip() or None
                    multi_tenancy = tool_parameters.get("multi_tenancy")

                    created = client.create_collection(
                        class_name=collection_name,
                        properties=props,
                        vectorizer=vectorizer,
                        vector_index_config=None,  # Can be exposed later if needed
                        description=description,
                        multi_tenancy=bool(multi_tenancy) if multi_tenancy is not None else None,
                    )
                    if created:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name},
                            message=f"Collection '{collection_name}' created successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to create collection '{collection_name}'"
                        ))
                    return

                # ---- delete_collection ----
                if operation == "delete_collection":
                    ok = client.delete_collection(collection_name)
                    if ok:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name},
                            message=f"Collection '{collection_name}' deleted successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to delete collection '{collection_name}'"
                        ))
                    return

                # ---- get_schema ----
                if operation == "get_schema":
                    schema = client.get_collection_schema(collection_name)
                    if schema:
                        yield self.create_json_message(create_success_response(
                            data={"schema": schema, "collection_name": collection_name},
                            message=f"Schema retrieved for '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to get schema for '{collection_name}'"
                        ))
                    return

                # ---- get_stats ----
                if operation == "get_stats":
                    stats = client.get_collection_stats(collection_name)
                    if stats:
                        yield self.create_json_message(create_success_response(
                            data={"stats": stats, "collection_name": collection_name},
                            message=f"Statistics retrieved for '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to get stats for '{collection_name}'"
                        ))
                    return

                # ---- exists ----
                if operation == "exists":
                    exists = client.collection_exists(collection_name)
                    yield self.create_json_message(create_success_response(
                        data={"exists": exists, "collection_name": collection_name},
                        message=f"Collection '{collection_name}' exists: {exists}"
                    ))
                    return

                # ---- add_property ----
                if operation == "add_property":
                    prop_str = (tool_parameters.get("property") or "").strip()
                    if not prop_str:
                        yield self.create_json_message(create_error_response(
                            "Property definition is required for add_property operation"
                        ))
                        return
                    
                    prop = safe_json_parse(prop_str)
                    if not isinstance(prop, dict) or "name" not in prop or "data_type" not in prop:
                        yield self.create_json_message(create_error_response(
                            "Invalid property format. Provide JSON with 'name' and 'data_type' fields"
                        ))
                        return
                    
                    success = client.add_property(collection_name, prop)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "property": prop["name"]},
                            message=f"Property '{prop['name']}' added to collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to add property '{prop['name']}' to collection '{collection_name}'"
                        ))
                    return

                # ---- update_config ----
                if operation == "update_config":
                    cfg_str = (tool_parameters.get("config") or "").strip()
                    if not cfg_str:
                        yield self.create_json_message(create_error_response(
                            "Config updates are required for update_config operation"
                        ))
                        return
                    
                    cfg = safe_json_parse(cfg_str)
                    if not isinstance(cfg, dict):
                        yield self.create_json_message(create_error_response(
                            "Invalid config format. Provide JSON object with configuration updates"
                        ))
                        return
                    
                    success = client.update_collection_config(collection_name, cfg)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "applied_config": cfg},
                            message=f"Configuration updated for collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to update configuration for collection '{collection_name}'"
                        ))
                    return

                # Fallback (shouldn't hit due to _ALLOWED_OPS)
                yield self.create_json_message(create_error_response(
                    f"Unsupported operation '{operation}'"
                ))

            except Exception as e:
                logger.exception("Schema management error")
                yield self.create_json_message(create_error_response(f"Operation failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))

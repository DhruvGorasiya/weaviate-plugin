from collections.abc import Generator
from typing import Any, List
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

_ALLOWED_OPS = {"list_tenants", "add_tenants", "delete_tenants"}

class TenantManagementTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            op_raw = tool_parameters.get("operation", "")
            operation = (op_raw or "").strip().lower()

            collection_name = (tool_parameters.get("collection_name") or "").strip()
            tenants_str = (tool_parameters.get("tenants") or "").strip()

            if operation not in _ALLOWED_OPS:
                yield self.create_json_message(create_error_response(
                    f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}"
                ))
                return

            if not collection_name:
                yield self.create_json_message(create_error_response(
                    "Collection name is required for tenant operations"
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
                # ---- list_tenants ----
                if operation == "list_tenants":
                    tenants = client.list_tenants(collection_name)
                    yield self.create_json_message(create_success_response(
                        data={"tenants": tenants, "count": len(tenants), "collection_name": collection_name},
                        message=f"Found {len(tenants)} tenants in collection '{collection_name}'"
                    ))
                    return

                # ---- add_tenants ----
                if operation == "add_tenants":
                    if not tenants_str:
                        yield self.create_json_message(create_error_response(
                            "Tenant list is required for add_tenants operation"
                        ))
                        return

                    tenants = safe_json_parse(tenants_str)
                    if not isinstance(tenants, list):
                        yield self.create_json_message(create_error_response(
                            "Invalid tenants format. Provide JSON array of tenant names"
                        ))
                        return

                    success = client.add_tenants(collection_name, tenants)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "added_tenants": tenants},
                            message=f"Added {len(tenants)} tenants to collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to add tenants to collection '{collection_name}'"
                        ))
                    return

                # ---- delete_tenants ----
                if operation == "delete_tenants":
                    if not tenants_str:
                        yield self.create_json_message(create_error_response(
                            "Tenant list is required for delete_tenants operation"
                        ))
                        return

                    tenants = safe_json_parse(tenants_str)
                    if not isinstance(tenants, list):
                        yield self.create_json_message(create_error_response(
                            "Invalid tenants format. Provide JSON array of tenant names"
                        ))
                        return

                    success = client.delete_tenants(collection_name, tenants)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "deleted_tenants": tenants},
                            message=f"Deleted {len(tenants)} tenants from collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to delete tenants from collection '{collection_name}'"
                        ))
                    return

            except Exception as e:
                logger.exception("Tenant management error")
                yield self.create_json_message(create_error_response(f"Operation failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))

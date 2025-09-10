from collections.abc import Generator
from typing import Any, List, Optional
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse
import openai
import anthropic

logger = logging.getLogger(__name__)

def _to_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]

def _parse_query_vector(raw) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except Exception:
            return None
    s = str(raw).strip()
    if not s:
        return None
    if s.startswith("["):
        arr = safe_json_parse(s)
        if isinstance(arr, list):
            try:
                return [float(x) for x in arr]
            except Exception:
                return None
        return None
    # CSV fallback
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return None

class GenerativeSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = (tool_parameters.get('collection_name') or '').strip()
            query = (tool_parameters.get('query') or '').strip()
            query_vector_raw = tool_parameters.get('query_vector')
            limit_raw = tool_parameters.get('limit', 5)

            llm_provider = (tool_parameters.get('llm_provider') or 'openai').strip().lower()
            llm_model = (tool_parameters.get('llm_model') or 'gpt-3.5-turbo').strip()
            llm_api_key = (tool_parameters.get('llm_api_key') or '').strip()

            where_filter_raw = tool_parameters.get('where_filter')
            return_properties_in = tool_parameters.get('return_properties')

            # ---- Validate basics
            if not collection_name:
                yield self.create_json_message(create_error_response("Collection name is required"))
                return
            if not query:
                yield self.create_json_message(create_error_response("Query is required"))
                return

            # limit: accept string or number; enforce 1–20 for generation latency
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_json_message(create_error_response("Limit must be an integer between 1 and 20"))
                return
            if not (1 <= limit <= 20):
                yield self.create_json_message(create_error_response("Limit must be between 1 and 20"))
                return
            # (Optional) also call your validate_limit if you want 1..1000 guard
            if not validate_limit(limit):
                yield self.create_json_message(create_error_response("Limit must be between 1 and 1000"))
                return

            # query_vector: accept JSON array or CSV
            query_vector = _parse_query_vector(query_vector_raw)
            if query_vector is not None and not validate_vector(query_vector):
                yield self.create_json_message(create_error_response("query_vector must be a non-empty list of numbers"))
                return

            # where filter: accept JSON string or dict
            where_filter = None
            if isinstance(where_filter_raw, str):
                s = where_filter_raw.strip()
                if s:
                    where_filter = safe_json_parse(s)
                    if where_filter is None or not validate_where_filter(where_filter):
                        yield self.create_json_message(create_error_response("Invalid where filter JSON"))
                        return
            elif isinstance(where_filter_raw, dict):
                where_filter = where_filter_raw
                if not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response("Invalid where filter JSON"))
                    return

            # return properties: CSV or list
            return_properties = _to_list(return_properties_in)

            # ---- Connect to Weaviate
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds['url'],
                api_key=creds.get('api_key'),
                timeout=60
            )

            try:
                # Retrieval
                if query_vector:
                    results = client.hybrid_search(
                        class_name=collection_name,
                        query=query,
                        query_vector=query_vector,
                        alpha=0.7,
                        limit=limit,
                        where_filter=where_filter,
                        return_properties=return_properties
                    )
                    search_type = "hybrid"
                else:
                    results = client.text_search(
                        class_name=collection_name,
                        query=query,
                        limit=limit,
                        where_filter=where_filter,
                        return_properties=return_properties
                    )
                    search_type = "keyword"

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': [],
                            'count': 0,
                            'collection': collection_name,
                            'query': query,
                            'search_type': search_type,
                            'answer': None,
                            'context': []
                        },
                        message="No relevant documents found"
                    ))
                    return

                # Build context from retrieved docs
                def props_to_text(doc: dict) -> str:
                    props = doc.get('properties') or {}
                    if isinstance(props, dict):
                        if 'text' in props and isinstance(props['text'], str):
                            return props['text']
                        # fallback: compact all properties
                        return "\n".join(f"{k}: {v}" for k, v in props.items() if isinstance(v, (str, int, float)))
                    return str(props)

                context_passages = [props_to_text(d) for d in results if d]
                context_passages = [c for c in context_passages if c.strip()]
                context_text = "\n\n".join(context_passages)

                if not context_passages:
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': results,
                            'count': len(results),
                            'collection': collection_name,
                            'query': query,
                            'search_type': search_type,
                            'answer': None,
                            'context': []
                        },
                        message="Retrieved documents had no usable text content"
                    ))
                    return

                # If no LLM key, return context only
                if not llm_api_key:
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': results,
                            'count': len(results),
                            'collection': collection_name,
                            'query': query,
                            'search_type': search_type,
                            'answer': None,
                            'context': context_passages,
                            'note': "No llm_api_key provided; returning retrieved context only."
                        },
                        message=f"Retrieved {len(results)} documents; no generation performed"
                    ))
                    return

                # ---- Optional generation
                try:
                    answer = self._generate_response_llm(
                        query=query,
                        context=context_text,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        llm_api_key=llm_api_key
                    )
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': results,
                            'count': len(results),
                            'collection': collection_name,
                            'query': query,
                            'search_type': search_type,
                            'answer': answer,
                            'context': context_passages
                        },
                        message=f"Retrieved {len(results)} documents and generated an answer"
                    ))
                except Exception as gen_err:
                    logger.exception("LLM generation error")
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': results,
                            'count': len(results),
                            'collection': collection_name,
                            'query': query,
                            'search_type': search_type,
                            'answer': None,
                            'context': context_passages,
                            'note': f"Generation failed: {gen_err}"
                        },
                        message=f"Retrieved {len(results)} documents; generation failed"
                    ))

            except Exception as e:
                logger.exception("Generative search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))
            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))

    def _generate_response_llm(self, query: str, context: str, llm_provider: str,
                               llm_model: str, llm_api_key: str) -> str:
        """
        Minimal LLM wrapper. Imports are deferred so the tool can run without
        openai/anthropic installed when generation is skipped.
        """
        prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"
        )

        if llm_provider == "openai":
            # Requires openai>=1.0.0
            import openai as _openai
            client = _openai.OpenAI(api_key=llm_api_key)
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=700,
            )
            return (resp.choices[0].message.content or "").strip()

        if llm_provider == "anthropic":
            # Requires anthropic>=0.30.0
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=llm_api_key)
            resp = client.messages.create(
                model=llm_model,
                max_tokens=700,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            # content is a list of blocks
            return "".join(block.get("text", "") for block in (resp.content or [])).strip()

        # Fallback: no provider match
        return ""
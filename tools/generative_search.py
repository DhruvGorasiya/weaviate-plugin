from collections.abc import Generator
from typing import Any
import json
import logging
import openai
import anthropic

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class GenerativeSearchTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = tool_parameters.get('collection_name', '').strip()
            query = tool_parameters.get('query', '').strip()
            query_vector_str = tool_parameters.get('query_vector', '').strip()
            limit = tool_parameters.get('limit', 5)
            llm_provider = tool_parameters.get('llm_provider', 'openai').strip().lower()
            llm_model = tool_parameters.get('llm_model', 'gpt-3.5-turbo').strip()
            llm_api_key = tool_parameters.get('llm_api_key', '').strip()
            where_filter_str = tool_parameters.get('where_filter', '').strip()
            return_properties_str = tool_parameters.get('return_properties', '').strip()
            
            if not collection_name:
                yield self.create_text_message("Error: Collection name is required")
                return
            
            if not query:
                yield self.create_text_message("Error: Query is required")
                return
            
            if not validate_limit(limit) or limit > 20:
                yield self.create_text_message("Error: Limit must be between 1 and 20")
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
            
            credentials = self.runtime.credentials
            client = WeaviateClient(
                url=credentials['url'],
                api_key=credentials.get('api_key'),
                timeout=60
            )
            
            try:
                # Perform search
                if query_vector_str:
                    try:
                        query_vector = [float(x.strip()) for x in query_vector_str.split(',')]
                        if not validate_vector(query_vector):
                            yield self.create_text_message("Error: Invalid query vector format")
                            return
                        
                        results = client.hybrid_search(
                            class_name=collection_name,
                            query=query,
                            query_vector=query_vector,
                            alpha=0.7,
                            limit=limit,
                            where_filter=where_filter,
                            return_properties=return_properties
                        )
                    except ValueError:
                        yield self.create_text_message("Error: Invalid query vector format. Use comma-separated numbers")
                        return
                else:
                    # Use text search only
                    results = client.text_search(
                        class_name=collection_name,
                        query=query,
                        limit=limit,
                        where_filter=where_filter,
                        return_properties=return_properties
                    )
                
                if not results:
                    yield self.create_text_message("No relevant documents found for the query")
                    return
                
                # Prepare context from retrieved documents
                context_docs = []
                for result in results:
                    doc_text = ""
                    if return_properties:
                        for prop in return_properties:
                            if prop in result.get('properties', {}):
                                doc_text += f"{prop}: {result['properties'][prop]}\n"
                    else:
                        # Use all properties
                        for key, value in result.get('properties', {}).items():
                            doc_text += f"{key}: {value}\n"
                    
                    if doc_text.strip():
                        context_docs.append(doc_text.strip())
                
                if not context_docs:
                    yield self.create_text_message("No usable content found in retrieved documents")
                    return
                
                # Generate response using LLM
                context = "\n\n".join(context_docs)
                response_text = self._generate_response(
                    query=query,
                    context=context,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key
                )
                
                response = create_success_response(
                    data={
                        'generated_response': response_text,
                        'context_documents': len(context_docs),
                        'query': query,
                        'collection': collection_name
                    },
                    message="Generated response using retrieved context"
                )
                
                yield self.create_json_message(response)
                
            except Exception as e:
                logger.error(f"Generative search error: {str(e)}")
                yield self.create_text_message(f"Search failed: {str(e)}")
            finally:
                client.disconnect()
                
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield self.create_text_message(f"Tool execution failed: {str(e)}")
    
    def _generate_response(self, query: str, context: str, llm_provider: str, 
                          llm_model: str, llm_api_key: str) -> str:
        try:
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            if llm_provider == "openai":
                client = openai.OpenAI(api_key=llm_api_key)
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            
            elif llm_provider == "anthropic":
                client = anthropic.Anthropic(api_key=llm_api_key)
                response = client.messages.create(
                    model=llm_model,
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            
            else:
                # Fallback to simple response
                return f"Based on the retrieved context, here's what I found related to '{query}':\n\n{context[:500]}..."
                
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return f"Error generating response: {str(e)}"

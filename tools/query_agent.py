from collections.abc import Generator
from typing import Any, Dict, List
import json
import logging
import openai
import anthropic
import re

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_limit
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class QueryAgentTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            query = tool_parameters.get('query', '').strip()
            collection_name = tool_parameters.get('collection_name', '').strip()
            max_results = tool_parameters.get('max_results', 10)
            llm_provider = tool_parameters.get('llm_provider', 'openai').strip().lower()
            llm_model = tool_parameters.get('llm_model', 'gpt-3.5-turbo').strip()
            llm_api_key = tool_parameters.get('llm_api_key', '').strip()
            
            if not query:
                yield self.create_text_message("Error: Query is required")
                return
            
            if not validate_limit(max_results) or max_results > 100:
                yield self.create_text_message("Error: Max results must be between 1 and 100")
                return
            
            credentials = self.runtime.credentials
            client = WeaviateClient(
                url=credentials['url'],
                api_key=credentials.get('api_key'),
                timeout=60
            )
            
            try:
                # Interpret the query using LLM
                interpretation = self._interpret_query(
                    query=query,
                    collection_name=collection_name,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key
                )
                
                if not interpretation:
                    yield self.create_text_message("Error: Could not interpret the query")
                    return
                
                # Execute the interpreted operation
                result = self._execute_operation(
                    client=client,
                    interpretation=interpretation,
                    max_results=max_results
                )
                
                if result is None:
                    yield self.create_text_message("Error: Could not execute the operation")
                    return
                
                # Generate a natural language response
                response_text = self._generate_response(
                    query=query,
                    result=result,
                    interpretation=interpretation,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key
                )
                
                response = create_success_response(
                    data={
                        'response': response_text,
                        'operation': interpretation.get('operation'),
                        'collection': interpretation.get('collection_name'),
                        'result_data': result
                    },
                    message="Query processed successfully"
                )
                
                yield self.create_json_message(response)
                
            except Exception as e:
                logger.error(f"Query agent error: {str(e)}")
                yield self.create_text_message(f"Query processing failed: {str(e)}")
            finally:
                client.disconnect()
                
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield self.create_text_message(f"Tool execution failed: {str(e)}")
    
    def _interpret_query(self, query: str, collection_name: str, llm_provider: str, 
                        llm_model: str, llm_api_key: str) -> Dict[str, Any]:
        try:
            prompt = f"""You are a Weaviate query agent. Analyze the following natural language query and determine what operation to perform.

Available operations:
1. search - Search for documents (vector, keyword, or hybrid)
2. list_collections - List all collections
3. get_schema - Get collection schema
4. get_stats - Get collection statistics
5. insert - Insert new documents
6. update - Update existing documents
7. delete - Delete documents
8. get - Get specific document by ID

Query: "{query}"
Collection: "{collection_name if collection_name else 'not specified'}"

Return a JSON object with:
- operation: the operation type
- collection_name: the collection to work with (infer if not specified)
- search_type: "vector", "keyword", or "hybrid" (for search operations)
- search_query: the search terms (for search operations)
- filters: any filters to apply
- properties: specific properties to return
- document_data: data for insert/update operations
- document_id: ID for get/update/delete operations

Example response:
{{"operation": "search", "collection_name": "Documents", "search_type": "hybrid", "search_query": "artificial intelligence", "filters": null, "properties": ["title", "content"]}}"""
            
            if llm_provider == "openai":
                client = openai.OpenAI(api_key=llm_api_key)
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                interpretation_text = response.choices[0].message.content.strip()
            elif llm_provider == "anthropic":
                client = anthropic.Anthropic(api_key=llm_api_key)
                response = client.messages.create(
                    model=llm_model,
                    max_tokens=500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                interpretation_text = response.content[0].text.strip()
            else:
                # Fallback interpretation
                return self._fallback_interpretation(query, collection_name)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', interpretation_text, re.DOTALL)
            if json_match:
                return safe_json_parse(json_match.group(), {})
            else:
                return self._fallback_interpretation(query, collection_name)
                
        except Exception as e:
            logger.error(f"Query interpretation error: {str(e)}")
            return self._fallback_interpretation(query, collection_name)
    
    def _fallback_interpretation(self, query: str, collection_name: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['search', 'find', 'look for', 'query']):
            return {
                'operation': 'search',
                'collection_name': collection_name or 'Documents',
                'search_type': 'hybrid',
                'search_query': query,
                'filters': None,
                'properties': None
            }
        elif any(word in query_lower for word in ['list', 'show', 'get all']):
            return {
                'operation': 'list_collections',
                'collection_name': None,
                'search_type': None,
                'search_query': None,
                'filters': None,
                'properties': None
            }
        else:
            return {
                'operation': 'search',
                'collection_name': collection_name or 'Documents',
                'search_type': 'hybrid',
                'search_query': query,
                'filters': None,
                'properties': None
            }
    
    def _execute_operation(self, client: WeaviateClient, interpretation: Dict[str, Any], 
                          max_results: int) -> Any:
        try:
            operation = interpretation.get('operation')
            collection_name = interpretation.get('collection_name')
            
            if operation == 'search':
                search_type = interpretation.get('search_type', 'hybrid')
                search_query = interpretation.get('search_query', '')
                filters = interpretation.get('filters')
                properties = interpretation.get('properties')
                
                if search_type == 'vector':
                    # This would need a vector, so fallback to hybrid
                    return client.hybrid_search(
                        class_name=collection_name,
                        query=search_query,
                        query_vector=[0.0] * 1536,  # Dummy vector
                        alpha=0.5,
                        limit=max_results,
                        where_filter=filters,
                        return_properties=properties
                    )
                elif search_type == 'keyword':
                    return client.text_search(
                        class_name=collection_name,
                        query=search_query,
                        limit=max_results,
                        where_filter=filters,
                        return_properties=properties
                    )
                else:  # hybrid
                    return client.hybrid_search(
                        class_name=collection_name,
                        query=search_query,
                        query_vector=[0.0] * 1536,  # Dummy vector
                        alpha=0.7,
                        limit=max_results,
                        where_filter=filters,
                        return_properties=properties
                    )
            
            elif operation == 'list_collections':
                return {'collections': client.list_collections()}
            
            elif operation == 'get_schema':
                schema = client.get_collection_schema(collection_name)
                return {'schema': schema} if schema else None
            
            elif operation == 'get_stats':
                stats = client.get_collection_stats(collection_name)
                return {'stats': stats} if stats else None
            
            else:
                return {'message': f'Operation {operation} not yet implemented in query agent'}
                
        except Exception as e:
            logger.error(f"Operation execution error: {str(e)}")
            return None
    
    def _generate_response(self, query: str, result: Any, interpretation: Dict[str, Any],
                          llm_provider: str, llm_model: str, llm_api_key: str) -> str:
        try:
            operation = interpretation.get('operation')
            
            if operation == 'search' and isinstance(result, list):
                if not result:
                    return f"I searched for '{query}' but didn't find any matching documents."
                
                response = f"I found {len(result)} documents matching '{query}':\n\n"
                for i, doc in enumerate(result[:5], 1):  # Show first 5
                    properties = doc.get('properties', {})
                    title = properties.get('title', properties.get('name', f'Document {i}'))
                    response += f"{i}. {title}\n"
                    if 'content' in properties:
                        content = properties['content'][:200] + "..." if len(properties['content']) > 200 else properties['content']
                        response += f"   {content}\n\n"
                
                if len(result) > 5:
                    response += f"... and {len(result) - 5} more documents."
                
                return response
            
            elif operation == 'list_collections':
                collections = result.get('collections', [])
                if not collections:
                    return "No collections found in your Weaviate instance."
                return f"I found {len(collections)} collections: {', '.join(collections)}"
            
            elif operation == 'get_schema':
                schema = result.get('schema')
                if not schema:
                    return f"Could not retrieve schema for collection '{interpretation.get('collection_name')}'."
                return f"Retrieved schema for collection '{interpretation.get('collection_name')}'."
            
            elif operation == 'get_stats':
                stats = result.get('stats')
                if not stats:
                    return f"Could not retrieve statistics for collection '{interpretation.get('collection_name')}'."
                count = stats.get('total_count', 0)
                return f"Collection '{interpretation.get('collection_name')}' contains {count} documents."
            
            else:
                return f"Processed your query: '{query}' using operation '{operation}'."
                
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"Processed your query: '{query}' but encountered an error generating the response."

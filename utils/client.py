import weaviate
from typing import Any, Dict, List, Optional, Union
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Property, DataType, Configure
# from weaviate.classes.types import UUID
import logging

logger = logging.getLogger(__name__)

class WeaviateClient:
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: int = 60):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
        
    def connect(self) -> weaviate.WeaviateClient:
        if self._client is None:
            auth_config = weaviate.AuthApiKey(api_key=self.api_key) if self.api_key else None
            self._client = weaviate.connect_to_local(
                url=self.url,
                auth_credentials=auth_config,
                timeout_config=(5, self.timeout)
            )
        return self._client
    
    def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None
    
    def create_collection(self, class_name: str, properties: List[Dict[str, Any]], 
                         vectorizer: Optional[str] = None, 
                         vector_index_config: Optional[Dict[str, Any]] = None) -> bool:
        try:
            client = self.connect()
            collection_properties = []
            
            for prop in properties:
                collection_properties.append(
                    Property(
                        name=prop['name'],
                        data_type=DataType(prop['data_type']),
                        description=prop.get('description', ''),
                        vectorizer_config=prop.get('vectorizer_config')
                    )
                )
            
            collection_config = {
                'properties': collection_properties,
                'vectorizer_config': vectorizer,
                'vector_index_config': vector_index_config
            }
            
            client.collections.create(class_name, **collection_config)
            return True
        except Exception as e:
            logger.error(f"Error creating collection {class_name}: {str(e)}")
            return False
    
    def delete_collection(self, class_name: str) -> bool:
        try:
            client = self.connect()
            client.collections.delete(class_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {class_name}: {str(e)}")
            return False
    
    def get_collection_schema(self, class_name: str) -> Optional[Dict[str, Any]]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            return collection.config.get()
        except Exception as e:
            logger.error(f"Error getting schema for {class_name}: {str(e)}")
            return None
    
    def insert_objects(self, class_name: str, objects: List[Dict[str, Any]]) -> List[str]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            result = collection.data.insert_many(objects)
            return [str(uuid) for uuid in result.uuids]
        except Exception as e:
            logger.error(f"Error inserting objects to {class_name}: {str(e)}")
            return []
    
    def update_object(self, class_name: str, uuid: str, properties: Dict[str, Any]) -> bool:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            collection.data.update(uuid, properties)
            return True
        except Exception as e:
            logger.error(f"Error updating object {uuid} in {class_name}: {str(e)}")
            return False
    
    def delete_object(self, class_name: str, uuid: str) -> bool:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            collection.data.delete_by_id(uuid)
            return True
        except Exception as e:
            logger.error(f"Error deleting object {uuid} from {class_name}: {str(e)}")
            return False
    
    def vector_search(self, class_name: str, query_vector: List[float], 
                     limit: int = 10, where_filter: Optional[Dict[str, Any]] = None,
                     return_properties: Optional[List[str]] = None,
                     return_metadata: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            
            query_builder = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit
            )
            
            if where_filter:
                query_builder = query_builder.where(where_filter)
            
            if return_properties:
                query_builder = query_builder.return_metadata(return_metadata or [])
                query_builder = query_builder.return_properties(return_properties)
            
            result = query_builder.do()
            
            return [
                {
                    'uuid': str(obj.uuid),
                    'properties': obj.properties,
                    'metadata': obj.metadata
                }
                for obj in result.objects
            ]
        except Exception as e:
            logger.error(f"Error performing vector search in {class_name}: {str(e)}")
            return []
    
    def hybrid_search(self, class_name: str, query: str, query_vector: List[float],
                     alpha: float = 0.7, limit: int = 10,
                     where_filter: Optional[Dict[str, Any]] = None,
                     return_properties: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            
            query_builder = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit
            )
            
            if where_filter:
                query_builder = query_builder.where(where_filter)
            
            if return_properties:
                query_builder = query_builder.return_properties(return_properties)
            
            result = query_builder.do()
            
            return [
                {
                    'uuid': str(obj.uuid),
                    'properties': obj.properties,
                    'metadata': obj.metadata
                }
                for obj in result.objects
            ]
        except Exception as e:
            logger.error(f"Error performing hybrid search in {class_name}: {str(e)}")
            return []
    
    def get_object(self, class_name: str, uuid: str, 
                  return_properties: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            
            query_builder = collection.query.get_by_id(uuid)
            if return_properties:
                query_builder = query_builder.return_properties(return_properties)
            
            result = query_builder.do()
            if result.objects:
                obj = result.objects[0]
                return {
                    'uuid': str(obj.uuid),
                    'properties': obj.properties,
                    'metadata': obj.metadata
                }
            return None
        except Exception as e:
            logger.error(f"Error getting object {uuid} from {class_name}: {str(e)}")
            return None
    
    def list_collections(self) -> List[str]:
        try:
            client = self.connect()
            collections = client.collections.list_all()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def get_collection_stats(self, class_name: str) -> Optional[Dict[str, Any]]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            aggregate = collection.aggregate.over_all(total_count=True).do()
            return {
                'total_count': aggregate.total_count,
                'class_name': class_name
            }
        except Exception as e:
            logger.error(f"Error getting stats for {class_name}: {str(e)}")
            return None
    
    def text_search(self, class_name: str, query: str, 
                   limit: int = 10, where_filter: Optional[Dict[str, Any]] = None,
                   return_properties: Optional[List[str]] = None,
                   search_properties: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            client = self.connect()
            collection = client.collections.get(class_name)
            
            query_builder = collection.query.bm25(
                query=query,
                limit=limit
            )
            
            if where_filter:
                query_builder = query_builder.where(where_filter)
            
            if return_properties:
                query_builder = query_builder.return_properties(return_properties)
            
            if search_properties:
                query_builder = query_builder.return_metadata(['score'])
            
            result = query_builder.do()
            
            return [
                {
                    'uuid': str(obj.uuid),
                    'properties': obj.properties,
                    'metadata': obj.metadata
                }
                for obj in result.objects
            ]
        except Exception as e:
            logger.error(f"Error performing text search in {class_name}: {str(e)}")
            return []

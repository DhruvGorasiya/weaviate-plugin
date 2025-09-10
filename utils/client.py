import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.config import Property, DataType, Tokenization, Configure
from weaviate.classes.data import DataObject
from weaviate.exceptions import UnexpectedStatusCodeError

logger = logging.getLogger(__name__)

def _parse_endpoint(url: str):
    """
    Accepts full http(s)://host[:port] and returns parts for connect_to_custom.
    """
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        # allow raw host
        host = url.replace("https://", "").replace("http://", "")
        return host, 443, True, host, 443, True
    http_secure = (p.scheme == "https")
    http_port = p.port or (443 if http_secure else 80)
    host = p.hostname or p.netloc

    grpc_secure = http_secure
    grpc_port = p.port or (443 if grpc_secure else 50051)
    return host, http_port, http_secure, host, grpc_port, grpc_secure

class WeaviateClient:
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: int = 60):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[weaviate.WeaviateClient] = None

    def connect(self) -> weaviate.WeaviateClient:
        if self._client is not None:
            return self._client

        http_host, http_port, http_secure, grpc_host, grpc_port, grpc_secure = _parse_endpoint(self.url)
        auth = Auth.api_key(self.api_key) if self.api_key else None

        self._client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
            auth_credentials=auth,
            timeout_config=(10, self.timeout),
        )
        if not self._client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        return self._client

    def disconnect(self):
        try:
            if self._client:
                self._client.close()
        except Exception:
            logger.debug("Client close failed quietly", exc_info=True)
        finally:
            self._client = None

    # ---------- Collections / Schema ----------

    def list_collections(self) -> List[str]:
        try:
            client = self.connect()
            infos = client.collections.list_all()
            # list_all() may return list[str] or richer objects; normalize to names
            names = []
            for it in infos:
                name = getattr(it, "name", None) or str(it)
                names.append(name)
            return names
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def get_collection_schema(self, class_name: str) -> Optional[Dict[str, Any]]:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            cfg = col.config.get()
            # cfg is a structured object; convert to dict if needed
            out = {
                "name": getattr(cfg, "name", class_name),
                "properties": [vars(p) for p in (getattr(cfg, "properties", []) or [])],
                "vectorizers": getattr(cfg, "vectorizers", None),
            }
            return out
        except Exception as e:
            logger.error(f"Error getting schema for {class_name}: {e}")
            return None

    def create_collection(
        self,
        class_name: str,
        properties: List[Dict[str, Any]],
        vectorizer: Optional[str] = None,
        vector_index_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Note: For Dify, default to self-provided vectors unless user explicitly sets a vectorizer.
        """
        try:
            client = self.connect()
            props: List[Property] = []
            for p in properties:
                dt_raw = (p.get("data_type") or "text").upper()
                dt = getattr(DataType, dt_raw, DataType.TEXT)
                token = p.get("tokenization")
                tok = getattr(Tokenization, token.upper(), None) if token else None
                props.append(Property(name=p["name"], data_type=dt, tokenization=tok))

            if vectorizer and vectorizer.lower() == "text2vec-openai":
                vec_cfg = Configure.Vectorizer.text2vec_openai()
            elif vectorizer and vectorizer.lower() == "text2vec-transformers":
                vec_cfg = Configure.Vectorizer.text2vec_transformers()
            else:
                vec_cfg = Configure.Vectors.self_provided()

            client.collections.create(
                name=class_name,
                properties=props,
                vector_config=vec_cfg,
            )
            # vector_index_config can be added when needed via col.config.update
            return True
        except Exception as e:
            logger.error(f"Error creating collection {class_name}: {e}")
            return False

    def delete_collection(self, class_name: str) -> bool:
        try:
            client = self.connect()
            if client.collections.exists(class_name):
                client.collections.delete(class_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {class_name}: {e}")
            return False

    def get_collection_stats(self, class_name: str) -> Optional[Dict[str, Any]]:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            agg = col.aggregate.over_all(total_count=True)
            res = agg.do()
            total = getattr(res, "total_count", None)
            return {"class_name": class_name, "total_count": total}
        except Exception as e:
            logger.error(f"Error getting stats for {class_name}: {e}")
            return None

    # ---------- Data Objects ----------

    def insert_objects(self, class_name: str, objects: List[Dict[str, Any]]) -> List[str]:
        """
        Accepts list of dicts, each like:
          {"id": <optional_uuid>, "properties": {...}, "vector": [..] (optional)}
        Returns list of UUIDs inserted.
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            uuids_out: List[str] = []
            with col.data.batch.dynamic() as batch:
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    uid = obj.get("id")
                    props = obj.get("properties") or obj  # allow raw property dicts
                    vec = obj.get("vector")
                    vec_payload = {"default": vec} if isinstance(vec, list) else None
                    batch.add_object(properties=props, uuid=uid, vector=vec_payload)
                    if uid:
                        uuids_out.append(str(uid))
            return uuids_out
        except Exception as e:
            logger.error(f"Error inserting objects to {class_name}: {e}")
            return []

    def update_object(self, class_name: str, uuid: str, properties: Dict[str, Any]) -> bool:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            col.data.update(uuid=uuid, properties=properties)
            return True
        except UnexpectedStatusCodeError as e:
            if getattr(e, "status_code", None) == 404:
                return False
            logger.error(f"Update error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating object {uuid} in {class_name}: {e}")
            return False

    def delete_object(self, class_name: str, uuid: str) -> bool:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            col.data.delete_by_id(uuid)
            return True
        except UnexpectedStatusCodeError as e:
            if getattr(e, "status_code", None) == 404:
                return False
            logger.error(f"Delete error: {e}")
            return False
    def get_object(self, class_name: str, uuid: str, return_properties: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            obj = col.data.object.get_by_id(uuid, return_properties=return_properties)
            if obj is None:
                return None
            return {
                "uuid": str(getattr(obj, "uuid", uuid)),
                "properties": getattr(obj, "properties", None),
                "metadata": getattr(obj, "metadata", None),
            }
        except Exception as e:
            logger.error(f"Error getting object {uuid} from {class_name}: {e}")
            return None

    # ---------- Queries ----------

    def _build_where(self, where_filter: Optional[Dict[str, Any]]):
        """
        Accepts dicts that match Weaviate Filter JSON structure or simple {field, operator, value}.
        Your tools already validate the shape; assume pass-through.
        """
        return where_filter or None

    def vector_search(
        self,
        class_name: str,
        query_vector: List[float],
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            res = col.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=self._build_where(where_filter),
                return_properties=return_properties,
                return_metadata=MetadataQuery(distance=True),
                include_vector=False,
                target_vector="default",
            )
            return [
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", None),
                    "metadata": {"distance": getattr(getattr(o, "metadata", None), "distance", None)},
                }
                for o in (getattr(res, "objects", []) or [])
            ]
        except Exception as e:
            logger.error(f"Error performing vector search in {class_name}: {e}")
            return []

    def hybrid_search(
        self,
        class_name: str,
        query: str,
        query_vector: List[float],
        alpha: float = 0.7,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            res = col.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=self._build_where(where_filter),
                return_properties=return_properties,
                return_metadata=MetadataQuery(score=True),
            )
            return [
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", None),
                    "metadata": {"score": getattr(getattr(o, "metadata", None), "score", None)},
                }
                for o in (getattr(res, "objects", []) or [])
            ]
        except Exception as e:
            logger.error(f"Error performing hybrid search in {class_name}: {e}")
            return []

    def text_search(
        self,
        class_name: str,
        query: str,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
        search_properties: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            res = col.query.bm25(
                query=query,
                limit=limit,
                filters=self._build_where(where_filter),
                return_properties=return_properties,
                query_properties=search_properties,  # restrict BM25 fields if provided
                include_vector=False,
            )
            return [
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", None),
                    "metadata": {},  # BM25 may not include score by default; add if you want with MetadataQuery
                }
                for o in (getattr(res, "objects", []) or [])
            ]
        except Exception as e:
            logger.error(f"Error performing text search in {class_name}: {e}")
            return []
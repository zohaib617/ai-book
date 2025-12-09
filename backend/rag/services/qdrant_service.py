from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
from backend.rag.models.chunk import DocumentChunk
from config import config
import uuid
from openai import OpenAI

class QdrantService:
    """
    Service for managing vector storage with Qdrant
    """
    def __init__(self):
        # Initialize Qdrant client
        if config.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(url=config.QDRANT_URL)

        # Initialize OpenAI client for embeddings
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for embeddings")

        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Collection name for the robotics book content
        self.collection_name = "robotics_book_content"

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """
        Initialize the Qdrant collection for storing content chunks
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Size of OpenAI embeddings
                    distance=models.Distance.COSINE
                )
            )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def store_chunk(self, chunk: DocumentChunk) -> str:
        """
        Store a content chunk in Qdrant
        """
        # Generate embedding for the content
        embedding = self._get_embedding(chunk.content)

        # Generate a unique ID for the chunk
        chunk_id = str(uuid.uuid4())

        # Prepare payload with metadata
        payload = {
            "content": chunk.content,
            "source_module": chunk.source_module,
            "source_chapter": chunk.source_chapter,
            "token_count": chunk.token_count,
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None
        }

        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )

        return chunk_id

    def store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Store multiple content chunks in Qdrant
        """
        chunk_ids = []
        points = []

        for chunk in chunks:
            # Generate embedding for the content
            embedding = self._get_embedding(chunk.content)

            # Generate a unique ID for the chunk
            chunk_id = str(uuid.uuid4())

            # Prepare payload with metadata
            payload = {
                "content": chunk.content,
                "source_module": chunk.source_module,
                "source_chapter": chunk.source_chapter,
                "token_count": chunk.token_count,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None
            }

            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )
            )

            chunk_ids.append(chunk_id)

        # Store all points in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return chunk_ids

    def search_content(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant content based on query
        """
        # Generate embedding for the query
        query_embedding = self._get_embedding(query)

        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "text": result.payload["content"],
                "module": result.payload["source_module"],
                "chapter": result.payload["source_chapter"],
                "similarity_score": result.score
            })

        return results

    def delete_chunk(self, chunk_id: str):
        """
        Delete a content chunk from Qdrant
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[chunk_id]
            )
        )

    def get_all_chunks(self, source_module: Optional[str] = None, source_chapter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all chunks, optionally filtered by source
        """
        # Prepare filter if needed
        if source_module or source_chapter:
            must_conditions = []
            if source_module:
                must_conditions.append(
                    models.FieldCondition(
                        key="source_module",
                        match=models.MatchValue(value=source_module)
                    )
                )
            if source_chapter:
                must_conditions.append(
                    models.FieldCondition(
                        key="source_chapter",
                        match=models.MatchValue(value=source_chapter)
                    )
                )

            filter_condition = models.Filter(must=must_conditions)
        else:
            filter_condition = None

        # Retrieve points from Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=10000  # Adjust as needed
        )

        # Format results
        formatted_results = []
        for point in results[0]:  # results is a tuple (points, next_page_offset)
            formatted_results.append({
                "id": point.id,
                "text": point.payload["content"],
                "module": point.payload["source_module"],
                "chapter": point.payload["source_chapter"],
                "token_count": point.payload["token_count"],
                "created_at": point.payload["created_at"]
            })

        return formatted_results

# Create a global instance
qdrant_service = QdrantService()
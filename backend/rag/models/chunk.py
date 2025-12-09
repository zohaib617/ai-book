from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentChunk(BaseModel):
    """
    Model representing a chunk of content for RAG system
    """
    id: Optional[str] = None
    content: str
    source_module: str
    source_chapter: str
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: dict = {}
    created_at: datetime = None

    class Config:
        # Allow additional fields if needed
        extra = "allow"
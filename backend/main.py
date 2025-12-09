from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import asyncio
from agents.openai_agents import openai_agent, RAGResponse
from rag.services.qdrant_service import qdrant_service
from rag.services.chunking_service import chunking_service
from config import config
from routes.chat import router as chat_router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Physical AI & Humanoid Robotics Book RAG API",
    description="API for querying robotics book content using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Models
class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7

class Source(BaseModel):
    module: str
    chapter: str
    text: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    citations: List[str]

class ContentChunk(BaseModel):
    content: str
    module: str
    chapter: str
    chunk_size: Optional[int] = 750

class IndexResponse(BaseModel):
    message: str
    indexed_chunks: int

class ValidateResponse(BaseModel):
    status: str
    vector_db_connected: bool
    content_indexed: bool
    last_indexed: Optional[str]

# Placeholder for RAG functionality
async def retrieve_content(question: str) -> List[dict]:
    """
    Retrieve relevant content chunks from the vector database based on the question
    """
    try:
        # Query the Qdrant vector database for similar content
        results = qdrant_service.search_content(question, top_k=config.TOP_K)
        return results
    except Exception as e:
        # If Qdrant is not available, return empty list
        print(f"Error retrieving content from Qdrant: {str(e)}")
        return []

@app.post("/api/query", response_model=QueryResponse)
async def query_content(request: QueryRequest):
    """
    Submit a question about humanoid robotics and get a response based on the book content
    """
    try:
        # Retrieve relevant content chunks
        context_chunks = await retrieve_content(request.question)

        # Generate response using OpenAI agent
        rag_response: RAGResponse = openai_agent.generate_response(request.question, context_chunks)

        # Format the response
        sources = [
            Source(
                module=chunk["module"],
                chapter=chunk["chapter"],
                text=chunk["text"],
                similarity_score=chunk["similarity_score"]
            )
            for chunk in context_chunks
        ]

        return QueryResponse(
            answer=rag_response.answer,
            sources=sources,
            citations=rag_response.citations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

from rag.models.chunk import DocumentChunk as ChunkModel
from datetime import datetime

@app.post("/api/index", response_model=IndexResponse)
async def index_content(content: ContentChunk):
    """
    Add or update the book content in the vector database for RAG
    """
    try:
        # Use the chunking service to split content into appropriate sizes
        chunks = chunking_service.chunk_content(
            content.content,
            content.module,
            content.chapter
        )

        # Store chunks in Qdrant
        chunk_ids = qdrant_service.store_chunks(chunks)

        return IndexResponse(
            message="Content successfully indexed",
            indexed_chunks=len(chunk_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing content: {str(e)}")

@app.get("/api/validate", response_model=ValidateResponse)
async def validate_system():
    """
    Check if the RAG system is properly configured and working
    """
    try:
        # Test Qdrant connection by trying to get collection info
        qdrant_service.client.get_collection(qdrant_service.collection_name)
        vector_db_connected = True
    except:
        vector_db_connected = False

    # Check if any content is indexed
    try:
        all_chunks = qdrant_service.get_all_chunks()
        content_indexed = len(all_chunks) > 0
        last_indexed = all_chunks[0]["created_at"] if all_chunks and "created_at" in all_chunks[0] else None
    except:
        content_indexed = False
        last_indexed = None

    status = "valid" if vector_db_connected else "invalid"

    return ValidateResponse(
        status=status,
        vector_db_connected=vector_db_connected,
        content_indexed=content_indexed,
        last_indexed=last_indexed
    )

# Include chat routes
app.include_router(chat_router, prefix="/api", tags=["chat"])

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
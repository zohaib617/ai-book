from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from ..services.rag_service import RAGService
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Verify that required environment variables are set
if not settings.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable is not set")

# Initialize the RAG service
rag_service = RAGService()

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = "physical-ai-humanoid-robotics"
    selected_text: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """
    Chat endpoint that uses RAG to answer questions about the Physical AI & Humanoid Robotics book content
    """
    try:
        # Use the RAG service to get a response based on the book content
        response = await rag_service.get_response(
            query=chat_request.message,
            context_filter=chat_request.context,
            selected_text=chat_request.selected_text
        )

        return ChatResponse(
            response=response.answer,
            sources=response.sources
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing chat request")
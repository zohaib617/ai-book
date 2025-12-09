import asyncio
from typing import List, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from ..config import settings

@dataclass
class RAGResponse:
    answer: str
    sources: List[str]
    context_used: str


class RAGService:
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=True
        )

        # Initialize OpenAI client using API key from environment variable
        self.openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self.collection_name = settings.QDRANT_COLLECTION_NAME

    async def get_response(self, query: str, context_filter: str = "physical-ai-humanoid-robotics", selected_text: Optional[str] = None) -> RAGResponse:
        """
        Get a response using RAG (Retrieval Augmented Generation) from the book content
        """
        try:
            # Search for relevant content in the vector database
            search_results = await self.search_content(query, context_filter)

            # Prepare context from retrieved documents
            context = self._prepare_context(search_results, selected_text)

            # Generate response using OpenAI
            answer = await self._generate_answer(query, context)

            # Extract source information
            sources = [result.get("source", "Unknown") for result in search_results if "source" in result]

            return RAGResponse(
                answer=answer,
                sources=sources,
                context_used=context
            )
        except Exception as e:
            # In case of error, return a default response
            return RAGResponse(
                answer=f"I'm sorry, I encountered an error processing your request about: {query}. Please make sure the backend services are properly configured.",
                sources=[],
                context_used=""
            )

    async def search_content(self, query: str, context_filter: str) -> List[dict]:
        """
        Search the Qdrant vector database for relevant content
        """
        try:
            # Convert query to embedding
            query_embedding = await self._get_embedding(query)

            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=None,  # Could add filters based on context_filter here
                limit=5  # Return top 5 results
            )

            # Extract content from search results
            results = []
            for hit in search_result:
                result = {
                    "content": hit.payload.get("content", ""),
                    "source": hit.payload.get("source", ""),
                    "module": hit.payload.get("module", ""),
                    "score": hit.score
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Error searching content: {e}")
            return []

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for the given text using OpenAI
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a dummy embedding in case of error
            return [0.0] * 1536  # Standard size for text-embedding-ada-002

    def _prepare_context(self, search_results: List[dict], selected_text: Optional[str] = None) -> str:
        """
        Prepare context from search results and optionally selected text
        """
        context_parts = []

        # If specific text was selected by the user, prioritize it
        if selected_text:
            context_parts.append(f"USER SELECTED TEXT: {selected_text}")

        # Add content from search results
        for result in search_results:
            content = result.get("content", "")
            source = result.get("source", "")
            module = result.get("module", "")

            if content:
                context_parts.append(f"SOURCE: {source} | MODULE: {module}\nCONTENT: {content}")

        return "\n\n".join(context_parts)

    async def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using OpenAI based on the context
        """
        try:
            system_message = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics educational book.
            Answer questions based only on the provided context from the book content.
            If the context doesn't contain the information needed to answer the question,
            clearly state that the information is not available in the provided context.

            Context from book modules:
            {context}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                max_tokens=500,
                temperature=0.3
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I'm sorry, I couldn't generate a response for: {query}"
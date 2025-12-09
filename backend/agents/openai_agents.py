from openai import OpenAI
import os
from typing import List, Dict, Any
from pydantic import BaseModel
from config import config

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[str]

class OpenAIAgent:
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = "gpt-3.5-turbo"  # Can be configured as needed

    def generate_response(self, question: str, context_chunks: List[Dict[str, Any]]) -> RAGResponse:
        """
        Generate a response based on the question and retrieved context chunks
        """
        # Format the context for the prompt
        context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])

        # Prepare the prompt
        prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics educational book.
        Answer the following question based only on the provided context from the book.
        If the context doesn't contain enough information to answer the question, say so.

        Context:
        {context_text}

        Question: {question}

        Please provide a clear, accurate answer based on the context.
        Include relevant citations to the modules/chapters where the information was found.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an educational assistant for a robotics book. Answer questions based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )

            # Extract the answer
            answer = response.choices[0].message.content

            # Extract sources (for now, just return the input context chunks as sources)
            sources = []
            for chunk in context_chunks:
                sources.append({
                    "module": chunk.get("module", "unknown"),
                    "chapter": chunk.get("chapter", "unknown"),
                    "text": chunk.get("text", "")[:200] + "...",  # Truncate for brevity
                    "similarity_score": chunk.get("similarity_score", 0.0)
                })

            # For citations, we'll return a placeholder - in a real system, this would come from the context
            citations = ["Citation would be extracted from context in full implementation"]

            return RAGResponse(
                answer=answer,
                sources=sources,
                citations=citations
            )

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

# Create a global instance
openai_agent = OpenAIAgent()
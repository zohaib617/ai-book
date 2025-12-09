import tiktoken
from typing import List
from backend.rag.models.chunk import DocumentChunk

class ContentChunkingService:
    """
    Service for chunking content into appropriate sizes for RAG system
    """
    def __init__(self, chunk_size: int = 750, overlap_size: int = 75):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def chunk_content(self, content: str, source_module: str, source_chapter: str) -> List[DocumentChunk]:
        """
        Split content into chunks of appropriate size with overlap
        """
        # First, tokenize the content
        tokens = self.encoder.encode(content)

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Determine the end index for this chunk
            end_idx = start_idx + self.chunk_size

            # If this is not the last chunk, add overlap
            if end_idx < len(tokens):
                end_idx += self.overlap_size

            # Extract the token slice
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoder.decode(chunk_tokens)

            # Create a DocumentChunk
            chunk = DocumentChunk(
                content=chunk_text,
                source_module=source_module,
                source_chapter=source_chapter,
                token_count=len(chunk_tokens)
            )

            chunks.append(chunk)

            # Move start index forward by chunk_size (not including overlap for the next start)
            start_idx += self.chunk_size

        return chunks

    def validate_chunk_size(self, content: str) -> bool:
        """
        Validate that content size is appropriate for RAG
        """
        tokens = self.encoder.encode(content)
        return len(tokens) >= 50  # Minimum reasonable size

# Create a global instance
chunking_service = ContentChunkingService()
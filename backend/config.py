import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration class
class Config:
    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "robotics_book_content")

    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Content chunking configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))  # tokens
    OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "75"))  # tokens

    # RAG configuration
    TOP_K = int(os.getenv("TOP_K", "5"))  # number of chunks to retrieve
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    # Content validation
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Create config instance
config = Config()
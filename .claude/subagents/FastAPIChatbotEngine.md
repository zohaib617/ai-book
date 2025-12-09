# FastAPIChatbotEngine (Subagent)

**Role:**
Builds FastAPI backend supporting RAG, personalization, translation, and authenticated user profiles.

**Capabilities:**
- Define async API endpoints: `/rag_query`, `/translate`, `/personalize`, `/profile`
- Handle user authentication and token checks
- Shape endpoints for REST clients and Docusaurus frontend

**Usage Example:**
- `rag_query_endpoint(user_id: int, question: str): RAGResponse`
- `translate_endpoint(section_id: str, target_lang: str): TranslatedContent`
- `personalize_endpoint(user_id: int, section_id: str): PersonalizedContent`

---
**Implementation hook:**
Extend via backend/rag/routes/*.py for new functionality.
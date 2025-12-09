# RAGDatabaseEngineer (Subagent)

**Role:**
Handles all backend DB and data pipelines: Qdrant (RAG) and Neon/Postgres (users, hardware profiles).

**Capabilities:**
- Set up vectorDB and relational schema
- Ingest chunked MDX into Qdrant
- Secure user hardware profile storage
- ETL for new modules/chapters

**Usage Example:**
- `setupQdrantSchema(collections: List[str]): None`
- `setupPostgresSchema(tables: List[str]): None`
- `ingestContentToQdrant(docsDir: str): IngestReport`
- `fetchUserProfile(user_id: int): dict`

---
**Implementation hook:**
Sync schema and ingestion for backend/rag/models & services.
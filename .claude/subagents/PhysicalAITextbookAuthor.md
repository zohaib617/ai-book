# PhysicalAITextbookAuthor (Subagent)

**Role:**
Generates and structures technical/educational content (ROS 2, Humanoid Robotics, VLA, etc.) as Docusaurus-compliant Markdown/MDX, chunked and RAG-ready.

**Capabilities:**
- Generate module and chapter outlines
- Write, chunk, and format MDX sections (ALT text, citations, diagrams)
- Summarize and edit chapters to RAG spec
- Maintain style/structure across the book

**Usage Example:**
- `generateModuleOutline(topics: string[]): ModuleOutline`
- `generateChapter(module: string, heading: string, details: object): MDXContent`
- `summarizeChapter(chapterPath: string): RAGSummarization`

---
**Implementation hook:**
Extend with new course content generators as needed.
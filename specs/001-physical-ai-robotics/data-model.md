# Data Model: Physical AI & Humanoid Robotics Book

## Content Entities

### Module
- **Name**: String (e.g., "ROS 2 Fundamentals", "Digital Twin", "AI-Robot Brain", "VLA")
- **Description**: String - Overview of the module's focus and objectives
- **Chapters**: Array of Chapter entities
- **Learning Objectives**: Array of String - What students will understand after completing the module
- **Prerequisites**: Array of String - Knowledge required before starting this module
- **Word Count**: Integer - Total word count (1,500-3,000 range)

### Chapter
- **Title**: String - The chapter name
- **Module**: Reference to parent Module
- **Position**: Integer - Order within the module (1-3)
- **Content**: String - The main chapter content in MD/MDX format
- **Learning Goals**: Array of String - Specific objectives for this chapter
- **RAG Summary**: String - Concise summary for RAG retrieval (500-900 tokens)
- **Diagrams**: Array of Diagram entities
- **Code Examples**: Array of CodeExample entities
- **Exercises**: Array of Exercise entities

### Diagram
- **Title**: String - Description of the diagram
- **Alt Text**: String - Accessible description for screen readers
- **File Path**: String - Relative path to the image file
- **Caption**: String - Explanatory text for the diagram
- **Module**: Reference to parent Module

### CodeExample
- **Title**: String - Brief description of the code purpose
- **Language**: String - Programming language (Python, etc.)
- **Code**: String - The actual code snippet
- **Explanation**: String - Detailed explanation of the code
- **Chapter**: Reference to parent Chapter
- **Validation Status**: Enum (verified, needs_review, invalid)

### Exercise
- **Title**: String - Brief description of the exercise
- **Type**: Enum (conceptual, practical, coding)
- **Difficulty**: Enum (beginner, intermediate, advanced)
- **Question**: String - The exercise prompt
- **Expected Outcome**: String - What the student should learn/produce
- **Chapter**: Reference to parent Chapter

### Citation
- **Type**: Enum (academic_paper, official_documentation, book, website)
- **Authors**: Array of String - Authors or organization
- **Title**: String - Title of the cited work
- **Publication**: String - Journal, conference, or publisher
- **Date**: String - Publication date
- **URL**: String - Link to the source (if applicable)
- **Used In**: Array of Chapter references

## RAG System Entities

### DocumentChunk
- **Content**: String - The text chunk (500-900 tokens)
- **Source Module**: String - Module identifier
- **Source Chapter**: String - Chapter identifier
- **Token Count**: Integer - Number of tokens in the chunk
- **Embedding**: Array of Float - Vector representation for similarity search
- **Metadata**: Object - Additional information for retrieval

### UserQuery
- **Question**: String - The user's question
- **Timestamp**: DateTime - When the query was made
- **Matched Chunks**: Array of DocumentChunk references
- **Response**: String - The AI-generated response
- **Source Citations**: Array of Citation references

## Validation Rules

### Module Validation
- Word count must be between 1,500 and 3,000
- Must have 2-3 chapters
- Learning objectives must be specific and measurable
- Prerequisites must reference other modules or general knowledge

### Chapter Validation
- Content must end with a RAG summary
- All diagrams must have ALT text
- Code examples must be verified for correctness
- Exercises must align with learning goals

### Content Validation
- No hallucinated claims (all facts must be cited)
- Technical accuracy verified against official documentation
- Citations must follow IEEE/APA format
- All external links must be valid

## State Transitions

### Content Creation Flow
1. **Draft** → Content is being written
2. **Reviewed** → Technical review completed
3. **Validated** → Accuracy verification passed
4. **Published** → Ready for RAG indexing

### RAG Processing Flow
1. **Queued** → Content chunk awaiting processing
2. **Processed** → Embedding generated
3. **Indexed** → Available for retrieval
4. **Active** → Serving queries
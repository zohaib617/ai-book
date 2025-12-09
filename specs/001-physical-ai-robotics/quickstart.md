# Quickstart: Physical AI & Humanoid Robotics Book

## Prerequisites

- Node.js 18+ (for Docusaurus)
- Python 3.9+ (for backend services)
- Git
- Access to OpenAI API (for RAG functionality)

## Setting Up the Documentation Site

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Local Development Server**
   ```bash
   npm run start
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

4. **Deploy to GitHub Pages**
   ```bash
   npm run deploy
   ```

## Setting Up the RAG Backend

1. **Create Python Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Set Up Environment Variables**
   ```bash
   # Create .env file in backend directory
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

4. **Start Backend Server**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

## Adding New Content

1. **Create New Module Directory**
   ```bash
   mkdir docs/modules/module-n-name
   ```

2. **Add Chapter Files**
   ```bash
   touch docs/modules/module-n-name/chapter-n-title.md
   ```

3. **Follow Content Template**
   ```md
   ---
   title: Chapter Title
   sidebar_position: 1
   ---

   # Chapter Title

   [Main content here]

   ## Summary

   [RAG-friendly summary of this chapter - 500-900 tokens]
   ```

4. **Add Diagrams with ALT Text**
   - Place images in `docs/static/img/`
   - Include descriptive ALT text in Markdown
   - Ensure diagrams support accessibility

## Content Creation Guidelines

### Writing Style
- Use clear, technical language appropriate for CS/AI/robotics students
- Explain concepts with progressive complexity
- Include practical examples and use cases
- Maintain consistent terminology across modules

### Technical Accuracy
- Verify all technical claims against official documentation
- Cite sources using IEEE/APA format
- Include 40% academic or official documentation sources
- Test all code examples in appropriate environments

### RAG Optimization
- Structure content in 500-900 token chunks
- End each chapter with a concise RAG summary
- Use semantic headings for better retrieval
- Include relevant keywords for search

### Accessibility
- Provide ALT text for all diagrams and images
- Use semantic HTML elements in MDX
- Maintain proper heading hierarchy
- Ensure color contrast meets WCAG standards

## RAG System Integration

1. **Index New Content**
   ```bash
   cd backend
   python -m rag.index_content
   ```

2. **Test RAG Queries**
   ```bash
   # Query endpoint: POST /api/query
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "your question here"}'
   ```

3. **Validate Content Retrieval**
   - Ensure queries return relevant results
   - Verify citations point to correct sections
   - Check response accuracy against book content

## Deployment

### Documentation Site
- The site automatically deploys to GitHub Pages when pushed to the main branch
- Configure in `docusaurus.config.js` under the `deploymentBranch` setting

### RAG Backend
- Deploy to a cloud provider (AWS, GCP, Azure, or Vercel/Render)
- Set up environment variables with API keys
- Configure the RAG system to index content from the deployed site

## Troubleshooting

### Common Issues
- **Build fails**: Check for syntax errors in Markdown files
- **RAG not returning results**: Verify content has been indexed
- **API errors**: Check environment variables and API key validity
- **Diagram not showing**: Verify image path and file format

### Validation Commands
```bash
# Validate content structure
npm run validate:content

# Test RAG functionality
npm run test:rag

# Check citations
npm run check:citations
```
# AI/Spec-Driven Book on Physical AI & Humanoid Robotics Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-08

## Active Technologies

- Docusaurus: Static site generator for technical documentation
- Markdown/MDX: Content format for documentation
- TypeScript: For Docusaurus customization
- FastAPI: Python web framework for backend API
- Qdrant: Vector database for RAG system
- OpenAI Agents: AI agents for conversation flow
- Node.js: Runtime for Docusaurus
- Python 3.9+: Runtime for backend services
- ROS 2: Robot Operating System for robotics concepts
- Gazebo: Robot simulation environment
- NVIDIA Isaac: AI framework for robotics
- Whisper: Speech recognition for voice-to-action

## Project Structure

```text
# Educational Book with RAG Backend
docs/
├── modules/
│   ├── module-1-ros2/
│   │   ├── chapter-1-nodes-topics-services.md
│   │   ├── chapter-2-rclpy-bridge.md
│   │   └── chapter-3-urdf-basics.md
│   ├── module-2-digital-twin/
│   │   ├── chapter-1-gazebo-physics.md
│   │   ├── chapter-2-sensor-simulation.md
│   │   └── chapter-3-unity-rendering.md
│   ├── module-3-ai-brain/
│   │   ├── chapter-1-isaac-sim.md
│   │   ├── chapter-2-isaac-navigation.md
│   │   └── chapter-3-nav2-bipedal.md
│   └── module-4-vla/
│       ├── chapter-1-whisper-vla.md
│       ├── chapter-2-cognitive-planning.md
│       └── chapter-3-capstone.md
├── src/
│   ├── components/
│   ├── pages/
│   └── css/
├── static/
│   └── img/             # Diagrams with ALT-text
└── docusaurus.config.js # Docusaurus configuration

backend/
├── main.py              # FastAPI entry point
├── rag/
│   ├── models/
│   ├── services/
│   └── routes/
├── agents/
│   └── openai_agents.py
├── requirements.txt
└── config.py

package.json             # Docusaurus dependencies
tsconfig.json            # TypeScript configuration
```

## Commands

### Docusaurus Commands
```bash
npm run start          # Start local development server
npm run build          # Build for production
npm run deploy         # Deploy to GitHub Pages
npm run serve          # Serve built site locally
```

### Backend Commands
```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# Start backend server
cd backend
uvicorn main:app --reload
```

### Content Management
```bash
# Create new module directory
mkdir docs/modules/module-n-name

# Add new chapter
touch docs/modules/module-n-name/chapter-n-title.md
```

### RAG System
```bash
# Index new content
cd backend
python -m rag.index_content

# Test RAG queries
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

## Code Style

### Markdown/MDX Style
- Use proper heading hierarchy (#, ##, ###)
- Include ALT text for all images
- Use proper code block syntax with language specification
- Follow IEEE/APA citation format
- Include RAG summaries at the end of each chapter

### Python Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Use meaningful variable names
- Follow FastAPI best practices

### TypeScript Style
- Use interfaces for object structures
- Follow React best practices for components
- Use proper error handling
- Follow Docusaurus theming conventions

## Recent Changes

- Physical AI & Humanoid Robotics Book: Created 4-module educational content covering ROS 2, Digital Twin simulation, AI perception/navigation, and Vision-Language-Action integration
- RAG System Implementation: Added FastAPI backend with Qdrant vector database for content retrieval
- Docusaurus Documentation Site: Set up documentation structure with module/chapter organization

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
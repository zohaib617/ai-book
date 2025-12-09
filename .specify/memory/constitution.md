<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.0.0 (initial constitution)
- Added sections: Core Principles (6), Technical Standards, Development Workflow, Governance
- Templates requiring updates: ✅ all templates updated to align with new principles
- Follow-up TODOs: None
-->
# AI/Spec-Driven Book on Physical AI & Humanoid Robotics Constitution

## Core Principles

### Technical Accuracy First
All content must be technically accurate using verified sources (ROS 2, Gazebo, Isaac, VLA). No hallucinations allowed; all claims must be sourced from official documentation, academic papers, or verified implementations.

### Student-Centered Clarity
Content must be structured clearly for CS/AI/robotics students. Code examples should be complete, runnable, and well-commented. Concepts should be explained with progressive complexity from basic to advanced.

### RAG-Ready Content (NON-NEGOTIABLE)
All content must be structured for RAG applications: clean chunking, 500-900 token segments, no hallucinations. Every chapter ends with a concise RAG summary suitable for retrieval.

### Verified Code Correctness
All code examples must be correct and runnable (Python, ROS 2, Nav2, rclpy). Code must be tested in appropriate environments before inclusion. Code examples should include error handling and edge cases where relevant.

### Academic Citation Standards
All content must follow IEEE/APA citation style. 40% of sources must be academic or official documentation. All diagrams must include ALT text for accessibility.

### Grounded Chatbot Responses
The RAG chatbot must answer only from selected book text. No external knowledge or hallucinations allowed. Responses must cite specific book sections as sources.

## Technical Standards
- 8-12 chapters, 1,200-2,000 words each
- Format: Docusaurus MD/MDX, deployable to GitHub Pages
- Backend: FastAPI + OpenAI Agents + Neon Postgres + Qdrant
- RAG chunks: 500-900 tokens
- Technology stack: ROS 2 (Nodes, Topics, URDF, controllers), Gazebo/Unity (physics, sensors, digital twin), NVIDIA Isaac (VSLAM, navigation, synthetic data), VLA (Whisper commands → ROS 2 actions)

## Development Workflow
- All changes must be spec-driven using Spec-Kit Plus
- Every feature requires a specification document before implementation
- Code reviews must verify compliance with all constitution principles
- All code examples must be tested in appropriate simulation environments (Gazebo, Isaac)
- Content must be validated against real ROS 2 implementations
- Chapters must include practical exercises and code examples

## Governance
This constitution supersedes all other practices for this project. All contributions must comply with these principles. Amendments require documentation, approval from project maintainers, and migration plan for existing content. All PRs and reviews must verify compliance with technical accuracy, citation standards, and RAG-readiness requirements.

**Version**: 1.0.0 | **Ratified**: 2025-12-08 | **Last Amended**: 2025-12-08
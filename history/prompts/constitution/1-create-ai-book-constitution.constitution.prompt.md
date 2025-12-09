---
id: 1
title: Create AI Book Constitution
stage: constitution
date: 2025-12-08
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,ai-book,physical-ai,robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```
Project: AI/Spec-Driven Book on Physical AI & Humanoid Robotics
(Using Docusaurus, Spec-Kit Plus, Claude Code, OpenAI Agents, RAG)

Core principles:
- Technical accuracy using verified sources (ROS 2, Gazebo, Isaac, VLA)
- Clear, structured writing for CS/AI/robotics students
- RAG-ready content: clean chunking, no hallucinations
- Code correctness (Python, ROS 2, Nav2, rclpy)

Key standards:
- Citation style: IEEE/APA
- Sources: 40% academic or official documentation
- Every chapter ends with a concise RAG summary
- All diagrams include ALT text
- Chatbot must answer only from selected book text

Constraints:
- 8–12 chapters, 1,200–2,000 words each
- Format: Docusaurus MD/MDX, deployable to GitHub Pages
- Backend: FastAPI + OpenAI Agents + Neon Postgres + Qdrant
- RAG chunks: 500–900 tokens

Scope:
- ROS 2: Nodes, Topics, URDF, controllers
- Gazebo/Unity: physics, sensors, digital twin
- NVIDIA Isaac: VSLAM, navigation, synthetic data
- VLA: Whisper commands → ROS 2 actions
- Capstone: Autonomous humanoid robot pipeline

Success criteria:
- Builds without errors
- RAG chatbot answers stay fully grounded
- All claims sourced
- All code runnable
```

## Response snapshot

Created a comprehensive constitution for the AI/Spec-Driven Book on Physical AI & Humanoid Robotics project with six core principles:

1. Technical Accuracy First - Ensuring all content is technically accurate using verified sources
2. Student-Centered Clarity - Structuring content clearly for CS/AI/robotics students
3. RAG-Ready Content (NON-NEGOTIABLE) - Structuring content for RAG applications with clean chunking
4. Verified Code Correctness - Ensuring all code examples are correct and runnable
5. Academic Citation Standards - Following IEEE/APA citation style with 40% academic sources
6. Grounded Chatbot Responses - Requiring chatbot to answer only from selected book text

Also included Technical Standards, Development Workflow, and Governance sections as required.

## Outcome

- ✅ Impact: Created initial project constitution with 6 core principles aligned to AI/robotics book requirements
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with project-specific constitution
- 🔁 Next prompts: Plan feature specifications based on this constitution
- 🧠 Reflection: Constitution captures all key requirements from the user input and establishes governance structure

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - all constitution principles aligned with project requirements
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Create initial feature specifications based on this constitution
# Physical AI & Humanoid Robotics Book

Welcome to the Physical AI & Humanoid Robotics educational book repository. This comprehensive resource covers the complete pipeline from ROS 2 fundamentals to Vision-Language-Action integration for humanoid robots.

## Table of Contents

1. [Module 1: ROS 2 Fundamentals](docs/modules/module-1-ros2/intro.md)
   - ROS 2 Nodes, Topics, Services
   - rclpy: Python Agent to ROS 2 bridge
   - URDF basics for humanoid robots

2. [Module 2: Digital Twin (Gazebo & Unity)](docs/modules/module-2-digital-twin/intro.md)
   - Gazebo physics: gravity, collisions
   - Sensor simulation (LiDAR, Depth, IMU)
   - Unity: high-fidelity rendering & interaction

3. [Module 3: AI-Robot Brain (NVIDIA Isaac™)](docs/modules/module-3-ai-brain/intro.md)
   - Isaac Sim photorealistic simulation + synthetic data
   - Isaac ROS pipelines: VSLAM, navigation
   - Nav2 for bipedal humanoid movement

4. [Module 4: Vision-Language-Action (VLA)](docs/modules/module-4-vla/intro.md)
   - Voice-to-Action (Whisper)
   - Cognitive Planning (LLMs → ROS 2 actions)
   - Capstone: Autonomous Humanoid pipeline

## Features

- **Comprehensive Coverage**: End-to-end robotics pipeline from middleware to AI integration
- **Educational Focus**: Designed for CS/AI/robotics students with progressive complexity
- **Technical Accuracy**: All content verified against official documentation and real implementations
- **RAG-Ready**: Content structured for Retrieval-Augmented Generation applications
- **Accessible**: Proper citations (IEEE/APA), ALT-text for diagrams, and structured content

## Getting Started

### Prerequisites

- Node.js 18+ for Docusaurus
- Python 3.9+ for backend services
- Access to OpenAI API for RAG functionality (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd physical-ai-humanoid-robotics-book
   ```

2. Install Docusaurus dependencies:
   ```bash
   npm install
   ```

3. Install Python backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Running the Documentation

1. Start the Docusaurus development server:
   ```bash
   npm run start
   ```

2. Access the documentation at `http://localhost:3000`

### Running the RAG Backend

1. Set up environment variables:
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## Architecture

The system consists of:

- **Frontend**: Docusaurus-based documentation site with modular content
- **Backend**: FastAPI-based RAG system with Qdrant vector database
- **AI Integration**: OpenAI agents for conversation and content retrieval
- **Simulation**: Integration with ROS 2, Isaac Sim, and Gazebo for robotics simulation

## Content Structure

```
docs/
├── modules/
│   ├── module-1-ros2/          # ROS 2 fundamentals
│   ├── module-2-digital-twin/  # Simulation and digital twin
│   ├── module-3-ai-brain/      # Perception and navigation
│   └── module-4-vla/           # Vision-Language-Action
└── static/
    └── img/                    # Diagrams with ALT-text
```

## RAG System

The book content is optimized for RAG (Retrieval-Augmented Generation) applications:

- Content chunked into 500-900 token segments
- Proper metadata for retrieval
- Citation tracking for grounding responses
- Technical accuracy verification

## Contributing

We welcome contributions to improve the content, fix errors, or add new sections. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-topic`)
3. Make your changes
4. Ensure content meets technical accuracy standards
5. Add proper citations and ALT-text for diagrams
6. Submit a pull request

## License

This educational content is provided under the terms specified in the LICENSE file.

## Acknowledgments

This book was created using the Spec-Driven Development methodology with Claude Code, Docusaurus, and other open-source tools. Special thanks to the ROS, Isaac Sim, and broader robotics communities for their excellent documentation and tools.

## Support

For questions about the content, please open an issue in the repository. For technical issues with the deployment, refer to the documentation or open an issue.
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'modules/module-1-ros2/intro',
        'modules/module-1-ros2/chapter-1-nodes-topics-services',
        'modules/module-1-ros2/chapter-2-rclpy-bridge',
        'modules/module-1-ros2/chapter-3-urdf-basics',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      items: [
        'modules/module-2-digital-twin/intro',
        'modules/module-2-digital-twin/chapter-1-gazebo-physics',
        'modules/module-2-digital-twin/chapter-2-sensor-simulation',
        'modules/module-2-digital-twin/chapter-3-unity-rendering',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain',
      items: [
        'modules/module-3-ai-brain/intro',
        'modules/module-3-ai-brain/chapter-1-isaac-sim',
        'modules/module-3-ai-brain/chapter-2-isaac-navigation',
        'modules/module-3-ai-brain/chapter-3-nav2-bipedal',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'modules/module-4-vla/intro',
        'modules/module-4-vla/chapter-1-whisper-vla',
        'modules/module-4-vla/chapter-2-cognitive-planning',
        'modules/module-4-vla/chapter-3-capstone',
        'modules/module-4-vla/conclusion',
      ],
    },
  ],
};

export default sidebars;
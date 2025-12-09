---
title: "Chapter 3 - URDF Basics for Humanoid Robots"
sidebar_position: 4
---

# Chapter 3: URDF Basics for Humanoid Robots

## Introduction

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. For humanoid robots, URDF is essential for defining the physical structure, kinematic relationships, and visual properties. This chapter covers the fundamentals of URDF with a focus on humanoid robot applications.

## Learning Goals

After completing this chapter, you will:
- Understand the structure and components of URDF files
- Define robot links and joints for humanoid structures
- Specify visual and collision properties for robot parts
- Create kinematic chains appropriate for humanoid robots
- Understand the relationship between URDF and robot simulation

## 1. Understanding URDF Structure

### URDF Overview

URDF (Unified Robot Description Format) is an XML format that describes a robot's physical properties:
- **Links**: Rigid parts of the robot (e.g., torso, limbs, head)
- **Joints**: Connections between links that allow motion
- **Materials**: Visual appearance properties
- **Transmissions**: How actuators connect to joints
- **Gazebo plugins**: Simulation-specific properties

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links definition -->
  <link name="base_link">
    <!-- Link properties -->
  </link>

  <!-- Joint definition -->
  <joint name="joint_name" type="joint_type">
    <!-- Joint properties -->
  </joint>
</robot>
```

## 2. Defining Links

### Link Components

A link represents a rigid body in the robot. It contains:

#### Visual Properties
```xml
<link name="torso">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.1 0.3"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1"/>
    </material>
  </visual>
</link>
```

#### Collision Properties
```xml
<link name="torso">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.1 0.3"/>
    </geometry>
  </collision>
</link>
```

#### Inertial Properties
```xml
<link name="torso">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

### Geometry Types

URDF supports several geometry types:
- `<box size="x y z"/>` - Rectangular prism
- `<cylinder radius="r" length="l"/>` - Cylindrical shape
- `<sphere radius="r"/>` - Spherical shape
- `<mesh filename="path_to_mesh.stl"/>` - Complex shapes from mesh files

## 3. Defining Joints

### Joint Types

URDF supports several joint types:

#### Fixed Joint
```xml
<joint name="fixed_joint" type="fixed">
  <parent link="base_link"/>
  <child link="sensor_link"/>
</joint>
```

#### Revolute Joint
```xml
<joint name="hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="thigh"/>
  <origin xyz="0 0 -0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

#### Continuous Joint
```xml
<joint name="continuous_joint" type="continuous">
  <parent link="base_link"/>
  <child link="rotating_part"/>
  <axis xyz="0 0 1"/>
</joint>
```

#### Prismatic Joint
```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="base_link"/>
  <child link="slider"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
</joint>
```

## 4. Humanoid Robot Structure

### Typical Humanoid Kinematic Chain

A basic humanoid robot typically has this structure:

```
base_link (pelvis)
├── torso
│   ├── head
│   ├── upper_left_arm
│   │   ├── lower_left_arm
│   │   └── left_hand
│   ├── upper_right_arm
│   │   ├── lower_right_arm
│   │   └── right_hand
│   ├── left_thigh
│   │   ├── left_calf
│   │   └── left_foot
│   └── right_thigh
│       ├── right_calf
│       └── right_foot
```

### Complete Humanoid Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Pelvis Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Torso to Base Joint -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>

  <!-- Head Joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>
</robot>
```

## 5. Materials and Colors

### Defining Materials

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="green">
  <color rgba="0 1 0 1"/>
</material>

<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<material name="black">
  <color rgba="0 0 0 1"/>
</material>

<material name="white">
  <color rgba="1 1 1 1"/>
</material>
```

### Using Textures

```xml
<material name="texture_mat">
  <color rgba="1 1 1 1"/>
  <texture filename="package://robot_description/meshes/texture.png"/>
</material>
```

## 6. Transmissions

### Joint Transmission

```xml
<transmission name="transmission_hip">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="hip_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="hip_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## 7. Gazebo Integration

### Gazebo-Specific Properties

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>1.0</kd>
</gazebo>
```

### Adding Gazebo Plugins

```xml
<gazebo>
  <plugin name="robot_state_publisher" filename="libgazebo_ros_robot_state_publisher.so">
    <tf_prefix>robot</tf_prefix>
  </plugin>
</gazebo>
```

## 8. Best Practices for Humanoid URDF

### Naming Conventions
- Use descriptive names: `left_hip_joint`, `right_upper_arm_link`
- Follow consistent patterns across the robot
- Use underscores to separate words

### Structure Organization
- Organize files in a hierarchical structure
- Use XACRO for complex robots to reduce redundancy
- Separate different parts into different files for modularity

### Physical Accuracy
- Ensure inertial properties are realistic
- Use appropriate collision geometries
- Consider mass distribution for stability

### Simulation Considerations
- Balance visual detail with performance
- Use simpler collision geometries than visual geometries
- Include proper joint limits and safety constraints

## 9. Summary

URDF is fundamental to describing humanoid robots in ROS. It defines the robot's structure through links and joints, specifies visual and collision properties, and enables simulation and control. A well-designed URDF is crucial for successful robot simulation, visualization, and control in ROS-based systems.

## RAG Summary

URDF (Unified Robot Description Format) is an XML format describing robot models with links (rigid bodies) and joints (connections). For humanoid robots, URDF defines the kinematic chain from base to extremities. Key components include visual/collision properties, joint types (fixed, revolute, continuous, prismatic), materials, and inertial properties. Best practices include descriptive naming, realistic physical properties, and proper organization for simulation.

## Exercises

1. Create a simplified URDF for a bipedal robot with 6 DOF per leg
2. Design a URDF structure for a humanoid robot arm with 7 DOF
3. Explain the differences between visual and collision properties in URDF
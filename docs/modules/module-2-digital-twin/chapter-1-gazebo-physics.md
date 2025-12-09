---
title: "Chapter 1 - Gazebo Physics: Gravity and Collisions"
sidebar_position: 2
---

# Chapter 1: Gazebo Physics: Gravity and Collisions

## Introduction

Gazebo is a powerful 3D simulation environment that provides accurate physics simulation for robotics applications. Understanding how physics works in Gazebo is crucial for creating realistic robot simulations. This chapter covers the fundamentals of physics simulation in Gazebo, focusing on gravity and collision detection.

## Learning Goals

After completing this chapter, you will:
- Understand how physics engines work in simulation environments
- Configure gravity settings for different environments
- Implement collision detection and response
- Understand the importance of physics simulation in robotics development
- Create physically accurate models for robot simulation

## 1. Physics Engine Fundamentals

### What is a Physics Engine?

A **physics engine** is a software component that simulates physical systems by applying laws of physics to virtual objects. In robotics simulation, physics engines provide:

- **Collision detection**: Identifying when objects intersect or touch
- **Collision response**: Calculating how objects react to collisions
- **Rigid body dynamics**: Simulating motion of solid objects
- **Constraints**: Modeling joints and connections between objects

### Gazebo's Physics Architecture

Gazebo supports multiple physics engines as plugins:
- **ODE (Open Dynamics Engine)**: Default engine, good balance of speed and accuracy
- **Bullet**: Popular for games, good for collision detection
- **DART**: Advanced dynamics research platform
- **Simbody**: Biomechanics-focused engine

### Physics Simulation Loop

```
1. Update object positions based on velocity
2. Detect collisions between objects
3. Calculate collision response forces
4. Apply forces to objects
5. Update velocities and accelerations
6. Repeat at fixed time intervals
```

## 2. Gravity in Gazebo

### Default Gravity Settings

Gazebo simulates Earth's gravity by default with a value of 9.8 m/s² in the negative Z direction:

```xml
<world>
  <gravity>0 0 -9.8</gravity>
  <!-- Other world settings -->
</world>
```

### Customizing Gravity

You can modify gravity for different environments:

```xml
<!-- Moon gravity (~1.6 m/s²) -->
<gravity>0 0 -1.6</gravity>

<!-- Mars gravity (~3.7 m/s²) -->
<gravity>0 0 -3.7</gravity>

<!-- Zero gravity (space simulation) -->
<gravity>0 0 0</gravity>

<!-- Custom direction (for tilted worlds) -->
<gravity>-1.0 0 -9.0</gravity>
```

### Gravity Considerations for Humanoid Robots

When simulating humanoid robots, consider:

- **Balance and stability**: Proper gravity ensures realistic balance challenges
- **Walking dynamics**: Gravity affects how robots interact with surfaces
- **Energy consumption**: Gravity impacts motor effort for maintaining posture
- **Fall detection**: Realistic gravity helps test recovery behaviors

## 3. Collision Detection

### Collision Shapes

Gazebo supports various collision shapes:

#### Primitive Shapes
- **Box**: `box size="x y z"`
- **Sphere**: `sphere radius="r"`
- **Cylinder**: `cylinder radius="r" length="l"`
- **Capsule**: For rounded shapes (not all engines support this)

#### Mesh Collisions
- **Triangle meshes**: For complex shapes using STL or other formats
- **Convex hulls**: Simplified collision meshes for better performance

### Collision Properties

```xml
<link name="collision_link">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>  <!-- Static friction coefficient -->
          <mu2>1.0</mu2>  <!-- Secondary friction coefficient -->
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1000000</kp>  <!-- Contact stiffness -->
          <kd>100</kd>      <!-- Contact damping -->
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Friction Models

#### Static Friction
- Determines when objects start sliding
- Coefficient typically between 0 (ice) and 1 (rubber on concrete)
- For humanoid robots, typical values: 0.5-0.9 for feet

#### Dynamic Friction
- Determines resistance during sliding
- Usually slightly lower than static friction

## 4. Collision Detection Strategies

### Simple vs. Complex Collisions

#### Simple Collisions (Recommended for Real-time Simulation)
- Use primitive shapes (boxes, spheres, cylinders)
- Sacrifices visual accuracy for performance
- Good for control algorithms and basic interaction

#### Complex Collisions
- Use detailed mesh shapes
- Higher accuracy but slower performance
- Suitable for precise manipulation tasks

### Multi-Collision Links

For complex shapes, you can define multiple collision elements:

```xml
<link name="complex_link">
  <collision name="collision_part_1">
    <geometry>
      <box size="0.1 0.1 0.2"/>
    </geometry>
  </collision>
  <collision name="collision_part_2">
    <geometry>
      <sphere radius="0.05"/>
    </geometry>
    <origin xyz="0.05 0 0.15"/>
  </collision>
</link>
```

## 5. Physics Performance Considerations

### Time Step Settings

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Physics update rate (1ms) -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time simulation speed -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Hz -->
</physics>
```

### Performance vs. Accuracy Trade-offs

| Setting | Performance | Accuracy | Use Case |
|---------|-------------|----------|----------|
| Large time step (0.01s) | Fast | Lower | Rough simulation |
| Small time step (0.001s) | Slower | Higher | Precise control |
| Simple collision shapes | Fast | Adequate | Real-time control |
| Complex collision meshes | Slower | Higher | Precise interaction |

### Optimization Strategies

- Use simplified collision geometry for fast-moving parts
- Increase friction coefficients instead of perfect collisions for stability
- Tune contact parameters (kp, kd) for desired behavior
- Use appropriate update rates for your application

## 6. Practical Examples for Humanoid Robots

### Foot Contact Simulation

```xml
<link name="foot_link">
  <collision name="foot_collision">
    <geometry>
      <box size="0.15 0.08 0.02"/>  <!-- Foot-sized box -->
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>   <!-- High friction for stable standing -->
          <mu2>0.8</mu2>
        </ode>
      </friction>
      <contact>
        <ode>
          <kp>10000000</kp>  <!-- High stiffness for solid contact -->
          <kd>1000</kd>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Balance and Stability Testing

Physics simulation is essential for:
- Testing walking controllers
- Evaluating fall recovery algorithms
- Assessing stability margins
- Validating center of mass calculations

## 7. Common Issues and Troubleshooting

### Objects Falling Through the Ground
- Check collision geometry alignment
- Verify surface contact parameters
- Ensure proper mass and inertia values

### Unstable Simulation
- Reduce time step size
- Adjust contact stiffness (kp) and damping (kd)
- Check mass distribution and center of mass

### Penetration Between Objects
- Increase contact stiffness (kp)
- Use more appropriate friction coefficients
- Verify collision geometry accuracy

## 8. Summary

Physics simulation in Gazebo is fundamental to realistic robot simulation. Properly configured gravity and collision detection ensure that robots behave realistically in the simulated environment. For humanoid robots, accurate physics simulation is essential for developing and testing balance, locomotion, and interaction algorithms before deploying to real hardware.

## RAG Summary

Gazebo provides physics simulation using engines like ODE, Bullet, or DART. Gravity is configured in world files (default 9.8 m/s²). Collision detection uses primitive shapes or meshes with friction, bounce, and contact properties. Performance vs. accuracy trade-offs involve time step size and collision complexity. For humanoid robots, proper physics simulation is crucial for balance and locomotion development.

## Exercises

1. Create a simple world with custom gravity settings
2. Design a collision model for a robot hand with appropriate friction
3. Explain the impact of different time step sizes on simulation stability
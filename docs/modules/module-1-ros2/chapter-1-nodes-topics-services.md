---
title: "Chapter 1 - ROS 2 Nodes, Topics, Services"
sidebar_position: 2
---

# Chapter 1: ROS 2 Nodes, Topics, Services

## Introduction

The Robot Operating System 2 (ROS 2) serves as the "nervous system" for robots, enabling communication between different components. In this chapter, you'll learn about the fundamental communication patterns in ROS 2: Nodes, Topics, and Services.

## Learning Goals

After completing this chapter, you will:
- Understand the role of Nodes as computational units in ROS 2
- Explain how Topics enable asynchronous many-to-many communication
- Describe how Services facilitate synchronous request-response communication
- Identify appropriate use cases for each communication pattern

## 1. Understanding ROS 2 Nodes

### What is a Node?

A **Node** is the fundamental building block of a ROS 2 program. It's a process that performs computation and typically represents a single function within a robot system. For example, you might have separate nodes for:
- Sensor data processing
- Motor control
- Path planning
- Localization
- Perception

### Creating a Node

In ROS 2, nodes are created using client libraries like `rclpy` (Python) or `rclcpp` (C++). Here's a basic example:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # Node initialization code here
```

### Node Characteristics

- **Lightweight**: Nodes are designed to be small and focused on specific tasks
- **Distributed**: Nodes can run on different machines and still communicate
- **Composable**: Multiple nodes work together to achieve complex robot behaviors
- **Managed**: Nodes can be started, stopped, and monitored by ROS 2 tools

## 2. Communication via Topics

### What are Topics?

**Topics** enable asynchronous, many-to-many communication in ROS 2. Multiple nodes can publish messages to the same topic, and multiple nodes can subscribe to receive those messages. This creates a decoupled communication pattern where publishers and subscribers don't need to know about each other.

### Publisher-Subscriber Pattern

```
Publisher Node A ──┐
                   ├── Topic ──► Subscriber Node X
Publisher Node B ──┘             Subscriber Node Y
                                 Subscriber Node Z
```

### Message Types

Each topic has a specific message type that defines the structure of data being transmitted. Common message types include:
- `std_msgs`: Basic data types (integers, floats, strings)
- `sensor_msgs`: Sensor data (LIDAR, cameras, IMU)
- `geometry_msgs`: Geometric data (poses, points, vectors)
- `nav_msgs`: Navigation-specific messages

### Example: Publishing to a Topic

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
```

## 3. Communication via Services

### What are Services?

**Services** enable synchronous, request-response communication in ROS 2. Unlike topics, services establish a direct connection between a client and a server. The client sends a request and waits for a response, making services appropriate for actions that require confirmation or computation.

### Client-Service Pattern

```
Client Node ──► Service Request ──► Service Server
                  ◄── Response ────
```

### When to Use Services vs Topics

| Use Services When... | Use Topics When... |
|---------------------|-------------------|
| You need a response to a request | You want to broadcast information |
| You need confirmation of completion | You want decoupled communication |
| The operation is synchronous | The operation is asynchronous |
| You have a one-to-one interaction | You have many-to-many communication |

### Example: Creating a Service

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response
```

## 4. Practical Considerations

### Communication Pattern Selection

When designing your robot system, consider:

1. **Real-time requirements**: Topics are better for continuous data streams
2. **Reliability needs**: Services guarantee delivery and response
3. **System complexity**: Topics allow for more flexible, decoupled architectures
4. **Data frequency**: High-frequency data typically uses topics

### Best Practices

- Keep nodes focused on single responsibilities
- Use descriptive names for topics and services
- Choose appropriate Quality of Service (QoS) settings
- Handle connection and disconnection events gracefully

## 5. Summary

ROS 2 Nodes, Topics, and Services form the backbone of robot communication systems. Nodes provide computational units, Topics enable asynchronous broadcasting, and Services facilitate synchronous request-response interactions. Understanding when and how to use each pattern is crucial for building robust and maintainable robot systems.

## RAG Summary

ROS 2 uses three main communication patterns: Nodes (computational units), Topics (asynchronous many-to-many communication), and Services (synchronous request-response). Nodes are lightweight processes that perform specific functions. Topics use a publisher-subscriber pattern for broadcasting data streams. Services use a client-server pattern for request-response interactions. Choose Topics for continuous data and decoupled systems, Services for operations requiring confirmation or synchronous behavior.

## Exercises

1. Design a simple robot system with 3 nodes using appropriate communication patterns
2. Identify which communication pattern would be best for sensor data publishing
3. Explain when you would use a Service instead of a Topic for a robot command
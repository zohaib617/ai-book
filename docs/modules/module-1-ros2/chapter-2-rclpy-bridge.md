---
title: "Chapter 2 - rclpy: Python Agent to ROS 2 Bridge"
sidebar_position: 3
---

# Chapter 2: rclpy: Python Agent to ROS 2 Bridge

## Introduction

`rclpy` is the Python client library for ROS 2 that provides a Python API for interacting with the ROS 2 middleware. It serves as the bridge between Python applications and the ROS 2 ecosystem, enabling Python developers to create nodes, publish and subscribe to topics, provide and call services, and manage parameters.

## Learning Goals

After completing this chapter, you will:
- Understand the role of rclpy in the ROS 2 ecosystem
- Create ROS 2 nodes using Python
- Implement publishers and subscribers with rclpy
- Build service clients and servers using Python
- Handle ROS 2 parameters in Python applications

## 1. Understanding rclpy Architecture

### Client Library Concept

`rclpy` is a **client library** that provides Python bindings to the ROS 2 middleware. It sits between your Python code and the ROS 2 client library (rcl), which in turn communicates with the underlying middleware (like DDS).

```
Your Python Code
        │
        ▼
    rclpy (Python bindings)
        │
        ▼
    rcl (ROS Client Library)
        │
        ▼
    Middleware (DDS Implementation)
```

### Key Components of rclpy

- **Node**: The basic execution unit in ROS 2
- **Publisher**: Sends messages to topics
- **Subscriber**: Receives messages from topics
- **Service Server**: Provides service functionality
- **Service Client**: Calls services
- **Parameter Client**: Manages node parameters
- **Timer**: Executes callbacks at specified intervals

## 2. Creating Your First rclpy Node

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)

    # Create an instance of your node
    my_node = MyNode()

    # Start processing callbacks
    rclpy.spin(my_node)

    # Destroy the node explicitly
    my_node.destroy_node()
    rclpy.shutdown()

class MyNode(Node):
    def __init__(self):
        # Call the parent class constructor and give it a name
        super().__init__('my_node_name')
        # Node initialization code goes here
```

### Node Lifecycle

When you create a node with rclpy:

1. **Initialization**: The node is created and registered with the ROS graph
2. **Execution**: The node runs and processes callbacks
3. **Shutdown**: The node is properly destroyed and unregistered

## 3. Implementing Publishers and Subscribers

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')

        # Create a publisher
        self.publisher = self.create_publisher(String, 'topic', 10)

        # Create a timer to periodically publish messages
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for message content
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        # Create a subscription
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)  # QoS history depth
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
```

## 4. Working with Services

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')

        # Create a service
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    minimal_service.destroy_node()
    rclpy.shutdown()
```

### Creating a Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')

        # Create a client
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()

    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result: {response.sum}')

    minimal_client.destroy_node()
    rclpy.shutdown()
```

## 5. Managing Parameters

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('threshold', 0.5)

        # Get parameter values
        my_param = self.get_parameter('my_parameter').value
        threshold = self.get_parameter('threshold').value

        self.get_logger().info(f'My parameter: {my_param}')
        self.get_logger().info(f'Threshold: {threshold}')

        # Set a parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(f'Set parameter: {param.name} = {param.value}')
        return SetParametersResult(successful=True)

from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_parameters

def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()
    rclpy.spin(parameter_node)
    parameter_node.destroy_node()
    rclpy.shutdown()
```

## 6. Advanced rclpy Concepts

### Quality of Service (QoS)

QoS settings control how messages are delivered between publishers and subscribers:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,  # or BEST_EFFORT
    history=HistoryPolicy.KEEP_LAST  # or KEEP_ALL
)

# Use the QoS profile when creating publisher/subscriber
publisher = self.create_publisher(String, 'topic', qos_profile)
```

### Timers and Callback Groups

```python
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Create callback group for concurrent execution
cb_group = MutuallyExclusiveCallbackGroup()

# Create timers with different callback groups
timer1 = self.create_timer(1.0, callback1, callback_group=cb_group)
timer2 = self.create_timer(0.5, callback2, callback_group=cb_group)
```

## 7. Best Practices

### Error Handling

```python
import rclpy
from rclpy.exceptions import ParameterNotDeclaredException

def safe_parameter_access(self, param_name, default_value):
    try:
        if not self.has_parameter(param_name):
            self.declare_parameter(param_name, default_value)
        return self.get_parameter(param_name).value
    except ParameterNotDeclaredException:
        self.get_logger().error(f'Parameter {param_name} not declared')
        return default_value
```

### Resource Management

- Always call `destroy_node()` to properly clean up resources
- Use try-finally blocks or context managers when appropriate
- Be mindful of memory usage in long-running nodes

## 8. Summary

`rclpy` provides the essential Python interface to ROS 2, enabling Python developers to create fully functional ROS 2 nodes. With rclpy, you can implement publishers, subscribers, services, and parameter management. Understanding the architecture and best practices of rclpy is crucial for building robust Python-based ROS 2 applications.

## RAG Summary

rclpy is the Python client library for ROS 2 that bridges Python applications with the ROS 2 middleware. It provides classes for Nodes, Publishers, Subscribers, Services, and Parameters. Key concepts include node lifecycle, publisher/subscriber implementation, service client/server patterns, and parameter management. Quality of Service (QoS) settings control message delivery characteristics. Best practices include proper resource management and error handling.

## Exercises

1. Create a Python node that publishes sensor data and handles parameters
2. Implement a service client that calls a distance calculation service
3. Design a parameter configuration system for a robot controller node
---
title: "Chapter 2 - LLM Cognitive Planning: From LLMs to ROS 2 Actions"
sidebar_position: 3
---

# Chapter 2: LLM Cognitive Planning: From LLMs to ROS 2 Actions

## Introduction

Large Language Models (LLMs) have revolutionized the way we approach cognitive planning in robotics. By leveraging the reasoning capabilities of LLMs, robots can interpret high-level natural language commands and translate them into executable action sequences. This chapter explores how to integrate LLMs with ROS 2 to create sophisticated cognitive planning systems that bridge human intent with robot action.

## Learning Goals

After completing this chapter, you will:
- Understand the architecture of LLM-based cognitive planning systems
- Implement LLM integration with ROS 2 action execution
- Design prompt engineering strategies for robotics applications
- Create multi-step planning pipelines with LLMs
- Implement safety and validation mechanisms for LLM-generated plans

## 1. LLM Cognitive Planning Architecture

### Overview of Cognitive Planning

Cognitive planning in robotics involves translating high-level goals into executable action sequences. With LLMs, this process becomes more flexible and adaptive:

```
Natural Language Goal → LLM Interpretation → Task Decomposition → Action Sequencing → ROS 2 Execution
```

### Key Components

#### LLM Interface Layer
- **Input**: Natural language goals or commands
- **Output**: Structured action plans or sequences
- **Technology**: OpenAI GPT, Anthropic Claude, or open-source alternatives
- **Requirements**: Context awareness, safety constraints, executable output

#### Task Decomposition Engine
- **Input**: High-level goals from LLM
- **Output**: Subtasks and dependencies
- **Technology**: Rule-based systems, LLM reasoning, or hybrid approaches
- **Requirements**: Task dependency management, resource allocation

#### Action Mapping System
- **Input**: Subtasks and parameters
- **Output**: ROS 2 action calls and parameters
- **Technology**: ROS 2 action clients, service calls, publishers
- **Requirements**: Real-time execution, error handling, feedback integration

#### Execution Monitor
- **Input**: Action execution status
- **Output**: Success/failure feedback, replanning triggers
- **Technology**: ROS 2 action monitoring, state machines
- **Requirements**: Real-time monitoring, adaptive replanning

## 2. LLM Integration with ROS 2

### Basic LLM Interface

```python
import openai
import json
from typing import Dict, List, Any
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus

class LLMRobotInterface:
    def __init__(self, api_key: str):
        # Initialize OpenAI client
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo"  # or "gpt-4" for more complex reasoning

        # ROS 2 interface
        self.node = None
        self.action_clients = {}

    def plan_from_command(self, command: str, robot_context: Dict) -> Dict:
        """
        Generate a plan from a natural language command using LLM
        """
        # Create system prompt with robot context
        system_prompt = self._create_system_prompt(robot_context)

        # Create user prompt with the command
        user_prompt = f"Generate a step-by-step plan to execute: '{command}'. " \
                     f"Return the plan as a JSON list of actions with parameters."

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent outputs
                max_tokens=500
            )

            plan_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            plan = self._extract_json_from_response(plan_text)
            return plan

        except Exception as e:
            return {
                "error": f"LLM planning failed: {str(e)}",
                "plan": []
            }

    def _create_system_prompt(self, robot_context: Dict) -> str:
        """Create system prompt with robot capabilities and context"""
        capabilities = robot_context.get('capabilities', [])
        environment = robot_context.get('environment', {})

        prompt = f"""
        You are a cognitive planning assistant for a robot. Your role is to convert natural language commands into executable action sequences.

        Robot capabilities: {', '.join(capabilities)}

        Environment: {json.dumps(environment)}

        Available actions:
        - navigate_to(location): Move robot to specified location
        - pick_object(object_name): Pick up an object
        - place_object(object_name, location): Place object at location
        - detect_object(object_name): Look for an object
        - say(text): Make robot speak
        - wait(duration): Wait for specified time

        Rules:
        1. Return plans as JSON array of action objects
        2. Each action object has 'action' and 'parameters' fields
        3. Only use available actions
        4. Consider safety and feasibility
        5. Include necessary preconditions and checks
        """
        return prompt

    def _extract_json_from_response(self, response_text: str) -> List[Dict]:
        """Extract JSON plan from LLM response"""
        try:
            # Look for JSON between markers or at the end
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif response_text.strip().startswith("["):
                # Try to extract JSON directly
                import re
                json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_text = response_text.strip()
            else:
                # Try to parse the whole response as JSON
                json_text = response_text.strip()

            return json.loads(json_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract action sequence from text
            return self._parse_actions_from_text(response_text)

    def _parse_actions_from_text(self, text: str) -> List[Dict]:
        """Fallback: parse actions from plain text"""
        # This is a simple fallback - in practice, you'd want more sophisticated parsing
        actions = []

        # Look for action patterns in the text
        if "navigate to" in text.lower():
            # Extract location and create navigation action
            actions.append({
                "action": "navigate_to",
                "parameters": {"location": "unknown_location"}
            })

        return actions
```

### ROS 2 Action Client Integration

```python
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize LLM interface
        self.llm_interface = LLMRobotInterface(api_key="your-api-key")

        # Create action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers and subscribers
        self.plan_request_sub = self.create_subscription(
            String, 'llm_plan_request', self.plan_request_callback, 10)
        self.plan_execution_pub = self.create_publisher(
            String, 'plan_execution_status', 10)

        # Store current plan and execution state
        self.current_plan = []
        self.plan_index = 0
        self.is_executing = False

    def plan_request_callback(self, msg):
        """Handle incoming plan requests"""
        command = msg.data

        # Get robot context
        robot_context = self._get_robot_context()

        # Generate plan using LLM
        plan = self.llm_interface.plan_from_command(command, robot_context)

        if "error" not in plan:
            self.get_logger().info(f"Generated plan: {plan}")
            self.execute_plan(plan)
        else:
            self.get_logger().error(f"Plan generation failed: {plan['error']}")
            self.publish_execution_status(f"Plan failed: {plan['error']}")

    def _get_robot_context(self) -> Dict:
        """Get current robot context for LLM planning"""
        return {
            "capabilities": [
                "navigation", "object_manipulation",
                "speech", "object_detection"
            ],
            "environment": {
                "locations": ["kitchen", "living_room", "bedroom"],
                "objects": ["cup", "book", "bottle"],
                "current_pose": {"x": 0.0, "y": 0.0, "theta": 0.0}
            }
        }

    async def execute_plan(self, plan: List[Dict]):
        """Execute the plan generated by LLM"""
        if self.is_executing:
            self.get_logger().warn("Plan execution already in progress")
            return

        self.current_plan = plan
        self.plan_index = 0
        self.is_executing = True

        self.get_logger().info(f"Starting execution of plan with {len(plan)} steps")

        while self.plan_index < len(self.current_plan) and self.is_executing:
            action = self.current_plan[self.plan_index]
            success = await self.execute_single_action(action)

            if success:
                self.plan_index += 1
                self.get_logger().info(f"Completed action {self.plan_index}: {action['action']}")
            else:
                self.get_logger().error(f"Failed to execute action: {action}")
                self.is_executing = False
                self.publish_execution_status(f"Plan execution failed at step {self.plan_index}")
                return

        self.is_executing = False
        self.publish_execution_status("Plan completed successfully")

    async def execute_single_action(self, action: Dict) -> bool:
        """Execute a single action from the plan"""
        action_name = action.get('action')
        parameters = action.get('parameters', {})

        try:
            if action_name == 'navigate_to':
                return await self.execute_navigation_action(parameters)
            elif action_name == 'pick_object':
                return await self.execute_pick_action(parameters)
            elif action_name == 'place_object':
                return await self.execute_place_action(parameters)
            elif action_name == 'detect_object':
                return await self.execute_detection_action(parameters)
            elif action_name == 'say':
                return await self.execute_speech_action(parameters)
            elif action_name == 'wait':
                return await self.execute_wait_action(parameters)
            else:
                self.get_logger().error(f"Unknown action: {action_name}")
                return False
        except Exception as e:
            self.get_logger().error(f"Error executing action {action_name}: {str(e)}")
            return False

    async def execute_navigation_action(self, params: Dict) -> bool:
        """Execute navigation action"""
        target_location = params.get('location')

        # In a real system, you'd have a map of named locations
        # For this example, we'll use a simple coordinate mapping
        location_map = {
            'kitchen': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
            'living_room': {'x': -1.0, 'y': 0.0, 'theta': 0.0},
            'bedroom': {'x': 0.0, 'y': 2.0, 'theta': 1.57}
        }

        if target_location in location_map:
            target = location_map[target_location]

            # Create navigation goal
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.pose.position.x = target['x']
            goal_msg.pose.pose.position.y = target['y']
            goal_msg.pose.pose.orientation.z = target['theta']

            # Send navigation goal
            self.nav_client.wait_for_server()
            future = self.nav_client.send_goal_async(goal_msg)

            # Wait for result (with timeout)
            try:
                goal_handle = await future
                if not goal_handle.accepted:
                    self.get_logger().error('Navigation goal rejected')
                    return False

                result_future = goal_handle.get_result_async()
                result = await result_future
                return result.result.status == GoalStatus.STATUS_SUCCEEDED
            except Exception as e:
                self.get_logger().error(f'Navigation failed: {str(e)}')
                return False
        else:
            self.get_logger().error(f'Unknown location: {target_location}')
            return False

    async def execute_pick_action(self, params: Dict) -> bool:
        """Execute pick object action"""
        object_name = params.get('object_name', params.get('object'))

        # This would integrate with manipulation stack
        # For now, just simulate the action
        self.get_logger().info(f"Simulating pick of object: {object_name}")
        return True  # Simulate success

    async def execute_place_action(self, params: Dict) -> bool:
        """Execute place object action"""
        object_name = params.get('object_name', params.get('object'))
        location = params.get('location')

        # This would integrate with manipulation stack
        self.get_logger().info(f"Simulating place of {object_name} at {location}")
        return True  # Simulate success

    async def execute_detection_action(self, params: Dict) -> bool:
        """Execute object detection action"""
        object_name = params.get('object_name', params.get('object'))

        # This would integrate with perception stack
        self.get_logger().info(f"Simulating detection of: {object_name}")
        return True  # Simulate success

    async def execute_speech_action(self, params: Dict) -> bool:
        """Execute speech action"""
        text = params.get('text', params.get('message', ''))

        # This would integrate with TTS system
        self.get_logger().info(f"Robot says: {text}")
        return True

    async def execute_wait_action(self, params: Dict) -> bool:
        """Execute wait action"""
        duration = params.get('duration', 1.0)  # seconds

        # This would use ROS 2 timer
        import time
        time.sleep(duration)  # This is blocking - in real system use async
        return True

    def publish_execution_status(self, status: str):
        """Publish execution status"""
        status_msg = String()
        status_msg.data = status
        self.plan_execution_pub.publish(status_msg)
```

## 3. Prompt Engineering for Robotics

### Effective Prompt Design

```python
class PromptEngineering:
    def __init__(self):
        self.base_context = self._get_base_context()

    def _get_base_context(self) -> str:
        """Get base context for all robotics prompts"""
        return """
        You are a cognitive planning assistant for a robot.
        Always consider:
        1. Safety: Ensure all actions are safe for the robot and environment
        2. Feasibility: Only suggest actions the robot can actually perform
        3. Context: Consider the current state and environment
        4. Sequence: Ensure actions are in logical order
        5. Error handling: Include checks and fallbacks where appropriate
        """

    def create_command_prompt(self, user_command: str, robot_state: Dict) -> str:
        """Create prompt for interpreting user command"""
        prompt = f"""
        {self.base_context}

        Robot State: {json.dumps(robot_state, indent=2)}

        User Command: "{user_command}"

        Please generate a step-by-step plan to execute this command.
        Return your response as a JSON array of action objects.
        Each action object should have:
        - "action": the action name
        - "parameters": a dictionary of parameters
        - "description": brief description of what this step does

        Available actions:
        - navigate_to(location): Move robot to named location
        - pick_object(object_name): Pick up an object
        - place_object(object_name, location): Place object at location
        - detect_object(object_name): Look for an object
        - say(text): Make robot speak
        - wait(duration): Wait for specified seconds
        - check_condition(condition): Check if condition is true

        Example response:
        [
            {{
                "action": "detect_object",
                "parameters": {{"object_name": "red cup"}},
                "description": "Look for the red cup"
            }},
            {{
                "action": "navigate_to",
                "parameters": {{"location": "kitchen"}},
                "description": "Move to kitchen area"
            }}
        ]
        """
        return prompt

    def create_replanning_prompt(self, failed_action: Dict, error: str, current_state: Dict) -> str:
        """Create prompt for replanning when action fails"""
        prompt = f"""
        {self.base_context}

        Previous Action: {json.dumps(failed_action)}
        Error: {error}
        Current Robot State: {json.dumps(current_state)}

        The previous action failed. Please generate a new plan to achieve the original goal,
        taking into account the failure and current state.
        Return your response as a JSON array of action objects.
        """
        return prompt

    def create_multi_step_prompt(self, high_level_goal: str, constraints: Dict) -> str:
        """Create prompt for complex multi-step planning"""
        prompt = f"""
        {self.base_context}

        High-level Goal: "{high_level_goal}"

        Constraints: {json.dumps(constraints)}

        Generate a detailed multi-step plan to achieve this goal.
        Consider:
        1. Prerequisites for each action
        2. Dependencies between actions
        3. Potential obstacles or challenges
        4. Verification steps after major actions

        Return your response as a JSON array of action objects with additional fields:
        - "action": the action name
        - "parameters": a dictionary of parameters
        - "preconditions": list of conditions that must be true before execution
        - "expected_effects": list of changes that should occur after execution
        - "verification": how to verify the action was successful
        """
        return prompt
```

### Safety-Conscious Prompting

```python
class SafetyConsciousPrompting:
    def __init__(self):
        self.safety_rules = [
            "Never suggest actions that could harm humans",
            "Never suggest actions that could damage the robot",
            "Always consider physical constraints of the robot",
            "Verify feasibility before suggesting actions",
            "Include safety checks in plans"
        ]

    def create_safe_command_prompt(self, command: str, safety_context: Dict) -> str:
        """Create a safety-conscious prompt for command execution"""
        safety_prompt = f"""
        Safety Rules:
        {chr(10).join(f"- {rule}" for rule in self.safety_rules)}

        Environment Safety Context:
        - Humans present: {safety_context.get('humans_present', 'unknown')}
        - Fragile objects: {safety_context.get('fragile_objects', [])}
        - Restricted areas: {safety_context.get('restricted_areas', [])}
        - Robot status: {safety_context.get('robot_status', 'normal')}

        Command: "{command}"

        Generate a safe plan that:
        1. Follows all safety rules
        2. Respects environmental constraints
        3. Includes safety verification steps
        4. Has appropriate fallbacks for failures

        Return as JSON array of safe action objects.
        """
        return safety_prompt
```

## 4. Multi-Step Planning and Execution

### Hierarchical Task Network (HTN) Planning

```python
class HTNPlanner:
    def __init__(self):
        self.primitive_actions = {
            'navigate_to',
            'pick_object',
            'place_object',
            'detect_object',
            'say',
            'wait',
            'check_condition'
        }

        self.complex_tasks = {
            'get_coffee': [
                {'action': 'navigate_to', 'parameters': {'location': 'kitchen'}},
                {'action': 'detect_object', 'parameters': {'object_name': 'coffee_cup'}},
                {'action': 'pick_object', 'parameters': {'object_name': 'coffee_cup'}},
                {'action': 'navigate_to', 'parameters': {'location': 'living_room'}},
                {'action': 'place_object', 'parameters': {'object_name': 'coffee_cup', 'location': 'table'}},
                {'action': 'say', 'parameters': {'text': 'I have brought your coffee'}}
            ],
            'greet_person': [
                {'action': 'detect_object', 'parameters': {'object_name': 'person'}},
                {'action': 'say', 'parameters': {'text': 'Hello! How can I help you today?'}}
            ]
        }

    def decompose_task(self, task_name: str, parameters: Dict = None) -> List[Dict]:
        """Decompose a high-level task into primitive actions"""
        if task_name in self.complex_tasks:
            # Clone the template and apply parameters
            task_plan = []
            for step in self.complex_tasks[task_name]:
                new_step = step.copy()
                if parameters:
                    # Apply parameter substitution
                    for key, value in parameters.items():
                        if isinstance(new_step['parameters'], dict):
                            if key in new_step['parameters']:
                                new_step['parameters'][key] = value
                task_plan.append(new_step)
            return task_plan
        else:
            return []

    def integrate_llm_planning(self, high_level_command: str, llm_interface: LLMRobotInterface) -> List[Dict]:
        """Integrate LLM planning with HTN decomposition"""
        # First, try to recognize known high-level tasks
        recognized_task = self._recognize_task(high_level_command)

        if recognized_task:
            # Use predefined decomposition
            return self.decompose_task(recognized_task['name'], recognized_task.get('parameters', {}))
        else:
            # Fall back to LLM planning
            robot_context = {
                'capabilities': list(self.primitive_actions),
                'known_tasks': list(self.complex_tasks.keys())
            }
            return llm_interface.plan_from_command(high_level_command, robot_context)

    def _recognize_task(self, command: str) -> Dict:
        """Recognize known high-level tasks from command"""
        command_lower = command.lower()

        # Simple keyword-based recognition
        if 'coffee' in command_lower:
            return {'name': 'get_coffee', 'parameters': {}}
        elif any(word in command_lower for word in ['hello', 'hi', 'greet']):
            return {'name': 'greet_person', 'parameters': {}}

        return None  # Task not recognized
```

### Plan Execution with Monitoring

```python
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PlanStatus(Enum):
    NOT_STARTED = "not_started"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PlanExecutionState:
    plan: List[Dict]
    current_step: int
    status: PlanStatus
    error: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

class PlanExecutionMonitor:
    def __init__(self, robot_interface: LLMPlanningNode):
        self.robot_interface = robot_interface
        self.current_execution = None
        self.max_execution_time = 300  # 5 minutes

    async def execute_plan_with_monitoring(self, plan: List[Dict]) -> PlanExecutionState:
        """Execute a plan with monitoring and error handling"""
        self.current_execution = PlanExecutionState(
            plan=plan,
            current_step=0,
            status=PlanStatus.EXECUTING,
            start_time=self.robot_interface.get_clock().now().nanoseconds / 1e9
        )

        try:
            while (self.current_execution.current_step < len(plan) and
                   self.current_execution.status == PlanStatus.EXECUTING):

                # Check for timeout
                current_time = self.robot_interface.get_clock().now().nanoseconds / 1e9
                if (current_time - self.current_execution.start_time) > self.max_execution_time:
                    self.current_execution.status = PlanStatus.FAILED
                    self.current_execution.error = "Plan execution timed out"
                    break

                # Execute current step
                step = plan[self.current_execution.current_step]
                success = await self._execute_step_with_monitoring(step)

                if success:
                    self.current_execution.current_step += 1
                else:
                    self.current_execution.status = PlanStatus.FAILED
                    break

            if self.current_execution.status == PlanStatus.EXECUTING:
                self.current_execution.status = PlanStatus.SUCCESS
                self.current_execution.completion_time = (
                    self.robot_interface.get_clock().now().nanoseconds / 1e9
                )

        except Exception as e:
            self.current_execution.status = PlanStatus.FAILED
            self.current_execution.error = f"Execution error: {str(e)}"

        return self.current_execution

    async def _execute_step_with_monitoring(self, step: Dict) -> bool:
        """Execute a single step with monitoring"""
        try:
            # Log step execution
            self.robot_interface.get_logger().info(
                f"Executing step {self.current_execution.current_step + 1}: {step['action']}"
            )

            # Execute the step
            success = await self.robot_interface.execute_single_action(step)

            # Log result
            if success:
                self.robot_interface.get_logger().info(
                    f"Step {self.current_execution.current_step + 1} completed successfully"
                )
            else:
                self.robot_interface.get_logger().error(
                    f"Step {self.current_execution.current_step + 1} failed"
                )

            return success

        except Exception as e:
            self.robot_interface.get_logger().error(
                f"Error in step {self.current_execution.current_step + 1}: {str(e)}"
            )
            return False

    def cancel_execution(self):
        """Cancel current plan execution"""
        if self.current_execution and self.current_execution.status == PlanStatus.EXECUTING:
            self.current_execution.status = PlanStatus.CANCELLED
            self.current_execution.error = "Execution cancelled by user"
```

## 5. Context and Memory Management

### Context-Aware Planning

```python
import time
from typing import Dict, List, Any

class ContextManager:
    def __init__(self):
        self.context = {
            'robot_state': {},
            'environment': {},
            'interaction_history': [],
            'object_locations': {},
            'recent_actions': [],
            'user_preferences': {}
        }
        self.max_history = 50

    def update_robot_state(self, new_state: Dict):
        """Update robot state in context"""
        self.context['robot_state'].update(new_state)

    def update_environment(self, new_env: Dict):
        """Update environment information"""
        self.context['environment'].update(new_env)

    def add_interaction(self, user_input: str, robot_response: str, timestamp: float = None):
        """Add interaction to history"""
        if timestamp is None:
            timestamp = time.time()

        interaction = {
            'user_input': user_input,
            'robot_response': robot_response,
            'timestamp': timestamp
        }

        self.context['interaction_history'].append(interaction)

        # Keep history size manageable
        if len(self.context['interaction_history']) > self.max_history:
            self.context['interaction_history'] = self.context['interaction_history'][-self.max_history:]

    def get_context_for_llm(self) -> Dict:
        """Get context formatted for LLM planning"""
        return {
            'current_pose': self.context['robot_state'].get('pose', {}),
            'battery_level': self.context['robot_state'].get('battery', 100),
            'available_actions': self.context['robot_state'].get('capabilities', []),
            'known_objects': list(self.context['object_locations'].keys()),
            'recent_interactions': self.context['interaction_history'][-5:],  # Last 5 interactions
            'environment_map': self.context['environment'].get('map', {}),
            'time_of_day': self._get_time_of_day()
        }

    def _get_time_of_day(self) -> str:
        """Get current time of day for context"""
        current_hour = time.localtime().tm_hour
        if 5 <= current_hour < 12:
            return 'morning'
        elif 12 <= current_hour < 17:
            return 'afternoon'
        elif 17 <= current_hour < 21:
            return 'evening'
        else:
            return 'night'

    def update_object_location(self, object_name: str, location: str):
        """Update known location of an object"""
        self.context['object_locations'][object_name] = {
            'location': location,
            'timestamp': time.time(),
            'confidence': 1.0
        }

    def get_relevant_context(self, goal: str) -> Dict:
        """Get context relevant to a specific goal"""
        relevant_context = self.get_context_for_llm()

        # Add goal-specific context
        if 'coffee' in goal.lower():
            # Add coffee-related context
            coffee_location = self.context['object_locations'].get('coffee', {}).get('location')
            if coffee_location:
                relevant_context['coffee_location'] = coffee_location

        return relevant_context
```

### Memory-Augmented Planning

```python
class MemoryAugmentedPlanner:
    def __init__(self):
        self.long_term_memory = {}
        self.episodic_memory = []
        self.semantic_memory = {}
        self.context_manager = ContextManager()

    def store_episode(self, episode_data: Dict):
        """Store an episode in episodic memory"""
        episode = {
            'id': len(self.episodic_memory),
            'timestamp': time.time(),
            'situation': episode_data.get('situation'),
            'action_sequence': episode_data.get('actions'),
            'outcome': episode_data.get('outcome'),
            'context': episode_data.get('context')
        }
        self.episodic_memory.append(episode)

    def retrieve_similar_episodes(self, current_situation: str, limit: int = 3) -> List[Dict]:
        """Retrieve similar past episodes"""
        # Simple similarity based on situation keywords
        # In practice, you'd use vector embeddings
        similar_episodes = []

        for episode in self.episodic_memory[-20:]:  # Check last 20 episodes
            if current_situation.lower() in episode['situation'].lower():
                similar_episodes.append(episode)
                if len(similar_episodes) >= limit:
                    break

        return similar_episodes

    def plan_with_memory(self, goal: str, llm_interface: LLMRobotInterface) -> List[Dict]:
        """Plan using both current context and memory"""
        # Get current context
        current_context = self.context_manager.get_relevant_context(goal)

        # Retrieve relevant past episodes
        similar_episodes = self.retrieve_similar_episodes(goal)

        # Create enhanced prompt with memory
        memory_context = {
            'current_context': current_context,
            'similar_episodes': similar_episodes,
            'semantic_knowledge': self._get_semantic_knowledge(goal)
        }

        # Generate plan with memory-augmented context
        plan = llm_interface.plan_from_command(goal, memory_context)

        return plan

    def _get_semantic_knowledge(self, goal: str) -> Dict:
        """Get semantic knowledge relevant to goal"""
        # This would be populated with domain knowledge
        semantic_knowledge = {
            'object_affordances': {
                'cup': ['graspable', 'container', 'movable'],
                'table': ['support', 'surface', 'reachable']
            },
            'spatial_relations': {
                'kitchen': ['cooking', 'food', 'utensils'],
                'bedroom': ['sleep', 'rest', 'clothing']
            }
        }

        return semantic_knowledge
```

## 6. Safety and Validation Mechanisms

### Plan Validation System

```python
class PlanValidator:
    def __init__(self):
        self.safety_constraints = {
            'max_navigation_distance': 20.0,  # meters
            'min_object_size': 0.01,  # meters
            'max_object_weight': 5.0,  # kg
            'forbidden_locations': ['restricted_area', 'danger_zone']
        }

    def validate_plan(self, plan: List[Dict], robot_state: Dict) -> Dict:
        """Validate a plan for safety and feasibility"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'safety_score': 1.0
        }

        for i, action in enumerate(plan):
            action_result = self._validate_action(action, robot_state, i)

            if not action_result['is_valid']:
                validation_result['is_valid'] = False
                validation_result['errors'].extend(action_result['errors'])
            elif action_result['warnings']:
                validation_result['warnings'].extend(action_result['warnings'])

        # Calculate overall safety score
        if validation_result['errors']:
            validation_result['safety_score'] = 0.0
        elif validation_result['warnings']:
            validation_result['safety_score'] = 0.5
        else:
            validation_result['safety_score'] = 1.0

        return validation_result

    def _validate_action(self, action: Dict, robot_state: Dict, step_index: int) -> Dict:
        """Validate a single action"""
        action_name = action.get('action', 'unknown')
        parameters = action.get('parameters', {})

        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        if action_name == 'navigate_to':
            location = parameters.get('location')
            if location in self.safety_constraints['forbidden_locations']:
                result['errors'].append(f"Navigation to {location} is forbidden")
                result['is_valid'] = False

        elif action_name == 'pick_object':
            # Validate object parameters
            obj_name = parameters.get('object_name', parameters.get('object'))
            if obj_name:
                # Check if object is known to be too heavy
                if obj_name in robot_state.get('heavy_objects', []):
                    result['warnings'].append(f"Object {obj_name} may be too heavy")

        elif action_name == 'wait':
            duration = parameters.get('duration', 0)
            if duration > 60:  # More than 1 minute
                result['warnings'].append(f"Long wait duration: {duration}s")

        # Check for action dependencies
        if action_name == 'place_object':
            obj_name = parameters.get('object_name', parameters.get('object'))
            # Would need to check if object was picked first
            if obj_name and not self._object_in_gripper(obj_name, robot_state):
                result['errors'].append(f"Cannot place {obj_name}, not currently holding it")
                result['is_valid'] = False

        return result

    def _object_in_gripper(self, obj_name: str, robot_state: Dict) -> bool:
        """Check if object is currently in robot's gripper"""
        # This would check the robot's current state
        held_objects = robot_state.get('held_objects', [])
        return obj_name in held_objects
```

### Safety Monitor

```python
class SafetyMonitor:
    def __init__(self, robot_interface: LLMPlanningNode):
        self.robot_interface = robot_interface
        self.emergency_stop_active = False
        self.safety_violations = 0
        self.max_violations = 5

    def check_safety_before_action(self, action: Dict) -> bool:
        """Check if an action is safe to execute"""
        if self.emergency_stop_active:
            return False

        # Check for safety violations
        safety_ok = True

        if action['action'] == 'navigate_to':
            # Check if navigation target is safe
            location = action['parameters'].get('location')
            if self._is_location_dangerous(location):
                safety_ok = False
                self._log_safety_violation(f"Dangerous location: {location}")

        elif action['action'] == 'pick_object':
            # Check if object is safe to pick
            obj_name = action['parameters'].get('object_name')
            if self._is_object_dangerous(obj_name):
                safety_ok = False
                self._log_safety_violation(f"Dangerous object: {obj_name}")

        return safety_ok

    def _is_location_dangerous(self, location: str) -> bool:
        """Check if location is dangerous"""
        # Implementation would check location database
        dangerous_locations = ['near_fire', 'construction_zone', 'restricted_area']
        return location in dangerous_locations

    def _is_object_dangerous(self, obj_name: str) -> bool:
        """Check if object is dangerous to handle"""
        # Implementation would check object database
        dangerous_objects = ['knife', 'hot_cup', 'sharp_object']
        return obj_name in dangerous_objects

    def _log_safety_violation(self, violation: str):
        """Log safety violation"""
        self.robot_interface.get_logger().warn(f"Safety violation: {violation}")
        self.safety_violations += 1

        if self.safety_violations >= self.max_violations:
            self.emergency_stop_active = True
            self.robot_interface.get_logger().error("Too many safety violations - emergency stop activated")

    def reset_safety_monitor(self):
        """Reset safety monitor"""
        self.emergency_stop_active = False
        self.safety_violations = 0
```

## 7. Performance Optimization

### Caching and Optimization

```python
from functools import lru_cache
import hashlib

class OptimizedLLMInterface:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.cache = {}
        self.max_cache_size = 100

    @lru_cache(maxsize=50)
    def cached_plan_generation(self, command_hash: str, system_prompt: str, user_prompt: str) -> Dict:
        """Cached version of plan generation"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            plan_text = response.choices[0].message.content.strip()
            return self._extract_json_from_response(plan_text)
        except Exception as e:
            return {"error": str(e), "plan": []}

    def plan_from_command(self, command: str, robot_context: Dict) -> Dict:
        """Generate plan with caching"""
        # Create cache key from command and context
        context_str = json.dumps(robot_context, sort_keys=True)
        cache_key = hashlib.md5(f"{command}_{context_str}".encode()).hexdigest()

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate new plan
        system_prompt = self._create_system_prompt(robot_context)
        user_prompt = f"Generate a step-by-step plan to execute: '{command}'. " \
                     f"Return the plan as a JSON list of actions with parameters."

        plan = self.cached_plan_generation(cache_key, system_prompt, user_prompt)

        # Update cache
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = plan

        return plan
```

## 8. Integration with ROS 2 Ecosystem

### Action Server Integration

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

class CognitivePlanningActionServer:
    def __init__(self, node: LLMPlanningNode):
        self.node = node
        self._action_server = ActionServer(
            node,
            CognitivePlan,  # Custom action type
            'cognitive_plan',
            self._execute_plan_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback
        )

    def _goal_callback(self, goal_request):
        """Accept or reject goal"""
        self.node.get_logger().info('Received cognitive planning goal')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Accept or reject cancel request"""
        self.node.get_logger().info('Received cancel goal request')
        return CancelResponse.ACCEPT

    async def _execute_plan_callback(self, goal_handle):
        """Execute the planning goal"""
        self.node.get_logger().info('Executing cognitive planning goal')

        feedback_msg = CognitivePlan.Feedback()
        result = CognitivePlan.Result()

        try:
            # Get command from goal
            command = goal_handle.request.command
            context = goal_handle.request.context

            # Generate plan using LLM
            plan = self.node.llm_interface.plan_from_command(command, context)

            if "error" not in plan:
                # Execute the plan with monitoring
                monitor = PlanExecutionMonitor(self.node)
                execution_state = await monitor.execute_plan_with_monitoring(plan)

                if execution_state.status == PlanStatus.SUCCESS:
                    result.success = True
                    result.message = "Plan executed successfully"
                    goal_handle.succeed()
                else:
                    result.success = False
                    result.message = f"Plan failed: {execution_state.error}"
                    goal_handle.abort()
            else:
                result.success = False
                result.message = f"Plan generation failed: {plan['error']}"
                goal_handle.abort()

        except Exception as e:
            result.success = False
            result.message = f"Execution error: {str(e)}"
            goal_handle.abort()

        return result
```

## 9. Best Practices

### Design Principles

- **Modularity**: Keep LLM interface separate from ROS 2 execution
- **Safety First**: Always validate plans before execution
- **Context Awareness**: Include relevant context for better planning
- **Error Handling**: Implement robust error handling and recovery
- **Performance**: Use caching and optimization where possible

### Implementation Guidelines

- Use appropriate LLM models for your computational constraints
- Implement proper error handling and fallback mechanisms
- Include safety validation at multiple levels
- Monitor execution and provide feedback
- Log interactions for debugging and improvement

## 10. Summary

LLM cognitive planning bridges natural language understanding with robot action execution. The system involves multiple components: LLM interface for plan generation, context management for situational awareness, safety validation for secure execution, and monitoring for adaptive behavior. Proper integration with ROS 2 enables complex multi-step tasks while maintaining safety and reliability.

## RAG Summary

LLM cognitive planning integrates large language models with ROS 2 to convert natural language commands into executable robot actions. Key components include LLM interface, context management, plan validation, and execution monitoring. The system uses prompt engineering to guide LLM behavior, implements safety validation mechanisms, and integrates with ROS 2 action servers for robust execution. Best practices include modularity, safety-first design, and performance optimization.

## Exercises

1. Implement a cognitive planning system for a mobile manipulator robot
2. Design safety validation mechanisms for LLM-generated plans
3. Create a context-aware prompt engineering system for robotics
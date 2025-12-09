import React, { useState, useRef, useEffect } from 'react';

// Global state for chat window visibility
let globalChatOpen = false;
const chatCallbacks = [];

const RAGChatWidget = ({ fromNavbar = false }) => {
  const [isOpen, setIsOpen] = useState(() => globalChatOpen);
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your AI assistant for Physical AI & Humanoid Robotics. Ask me anything about Modules 1-4!", sender: 'bot', timestamp: new Date() }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Register this instance to sync state across navbar and floating icon
  useEffect(() => {
    const updateState = (newState) => {
      setIsOpen(newState);
      globalChatOpen = newState;
      if (!newState) {
        setUnreadCount(0);
      }
    };

    chatCallbacks.push(updateState);

    return () => {
      const index = chatCallbacks.indexOf(updateState);
      if (index > -1) {
        chatCallbacks.splice(index, 1);
      }
    };
  }, []);

  // Update local state when global state changes
  useEffect(() => {
    if (globalChatOpen !== isOpen) {
      setIsOpen(globalChatOpen);
    }
  }, [globalChatOpen, isOpen]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && !isLoading) {
      inputRef.current?.focus();
    }
  }, [isOpen, isLoading]);

  const toggleChat = () => {
    const newState = !globalChatOpen;
    globalChatOpen = newState;

    // Notify all instances
    chatCallbacks.forEach(callback => callback(newState));
  };

  const openChat = () => {
    if (!globalChatOpen) {
      globalChatOpen = true;
      chatCallbacks.forEach(callback => callback(true));
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Integration note: Replace this mock response with actual API call
      // Example API call:
      // const response = await fetch('/api/chat', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ message: inputValue, context: 'physical-ai-humanoid-robotics' })
      // });
      // const data = await response.json();

      // Mock bot response based on content from the book modules
      let botResponse = "I'm searching through the Physical AI & Humanoid Robotics book content to answer your question...";

      if (inputValue.toLowerCase().includes('ros') || inputValue.toLowerCase().includes('module 1')) {
        botResponse = "Based on Module 1: ROS 2 (Robot Operating System 2) is the middleware for humanoid control. It provides nodes for different functionalities, topics for communication between nodes, and services for request/response interactions. The rclpy library bridges Python agents to ROS 2, and URDF (Unified Robot Description Format) defines robot structure and joints.";
      } else if (inputValue.toLowerCase().includes('simulation') || inputValue.toLowerCase().includes('module 2')) {
        botResponse = "Based on Module 2: Digital twin simulation uses Gazebo for physics simulation including gravity and collision detection, and Unity for high-fidelity rendering and interaction. Sensor simulation includes LiDAR, depth cameras, and IMU sensors to provide realistic sensory input for the humanoid robot.";
      } else if (inputValue.toLowerCase().includes('ai') || inputValue.toLowerCase().includes('module 3')) {
        botResponse = "Based on Module 3: The AI-Robot Brain uses NVIDIA Isaac for perception and navigation. Isaac Sim provides photorealistic simulation and synthetic data. Isaac ROS pipelines handle VSLAM (Visual Simultaneous Localization and Mapping) and navigation. Nav2 is used for bipedal humanoid movement and path planning.";
      } else if (inputValue.toLowerCase().includes('vla') || inputValue.toLowerCase().includes('module 4')) {
        botResponse = "Based on Module 4: Vision-Language-Action (VLA) integrates Whisper for voice-to-action conversion, and cognitive planning uses LLMs to create action plans that connect to ROS 2 actions. The capstone project combines all modules into an autonomous humanoid pipeline.";
      }

      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      const botMessage = {
        id: Date.now() + 1,
        text: botResponse,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);

      if (!globalChatOpen) {
        setUnreadCount(prev => prev + 1);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error processing your request. Please try again.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape' && globalChatOpen) {
      globalChatOpen = false;
      chatCallbacks.forEach(callback => callback(false));
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Render floating icon if not called from navbar
  if (!fromNavbar && !globalChatOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-[999999] chatbot-floating-icon">
        <button
          onClick={openChat}
          className="bg-teal-500 hover:bg-teal-600 text-white p-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-110 focus:outline-none focus:ring-4 focus:ring-teal-300"
          aria-label="Open chat"
          aria-expanded="false"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
        {unreadCount > 0 && (
          <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-6 w-6 flex items-center justify-center animate-pulse">
            {unreadCount}
          </span>
        )}
      </div>
    );
  }

  // Render chat window if open (either from icon or navbar)
  if (globalChatOpen) {
    return (
      <div
        className="fixed bottom-24 right-6 z-[999999] flex flex-col animate-fade-in"
        role="dialog"
        aria-modal="true"
        aria-label="Chat with AI Assistant"
      >
        <div className="bg-white rounded-lg shadow-xl flex flex-col h-[500px] w-full max-w-xs border border-gray-200">
          {/* Header */}
          <div className="bg-teal-500 text-white p-4 rounded-t-lg flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <h3 className="font-semibold">Physical AI Assistant</h3>
            </div>
            <button
              onClick={toggleChat}
              className="text-white hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-white rounded"
              aria-label="Close chat"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    message.sender === 'user'
                      ? 'bg-teal-500 text-white'
                      : 'bg-gray-200 text-gray-800'
                  }`}
                >
                  <p className="text-sm">{message.text}</p>
                  <p className={`text-xs mt-1 ${message.sender === 'user' ? 'text-teal-100' : 'text-gray-500'}`}>
                    {formatTime(message.timestamp)}
                  </p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg max-w-xs">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-200 bg-white">
            <div className="flex space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about Physical AI & Humanoid Robotics..."
                className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                aria-label="Type your message"
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className="bg-teal-500 hover:bg-teal-600 disabled:bg-gray-300 text-white p-2 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-teal-500"
                aria-label="Send message"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  return null;
};

export default RAGChatWidget;

// Export a function to open the chat from navbar
export const openChatFromNavbar = () => {
  globalChatOpen = true;
  chatCallbacks.forEach(callback => callback(true));
};
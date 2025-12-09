import React from 'react';
import { useLocation } from '@docusaurus/router';
import { openChatFromNavbar } from '../components/RAGChatWidget';

// Import the original Navbar component and wrap it
import OriginalNavbar from '@theme-original/Navbar';

export default function Navbar(props) {
  const location = useLocation();

  const handleChatClick = (e) => {
    e.preventDefault();
    openChatFromNavbar();
  };

  return (
    <>
      <OriginalNavbar {...props} />
      {/* Add the Chatbot button to the navbar */}
      <div className="navbar-item">
        <button
          onClick={handleChatClick}
          className="navbar-chat-button text-gray-600 hover:text-teal-500 dark:text-gray-300 dark:hover:text-teal-400 focus:outline-none focus:ring-2 focus:ring-teal-500 rounded p-2"
          aria-label="Open AI Assistant Chat"
          title="AI Assistant"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
        </button>
      </div>

      <style jsx>{`
        .navbar-item {
          display: flex;
          align-items: center;
        }

        .navbar-chat-button {
          margin-left: 1rem;
        }
      `}</style>
    </>
  );
}
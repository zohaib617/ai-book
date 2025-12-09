import React, { useEffect } from 'react';
import RAGChatWidget from './RAGChatWidget';
import { openChatFromNavbar } from './RAGChatWidget';

export default function Root({children}) {
  useEffect(() => {
    // More robust event listener for the chatbot navbar button
    const handleChatbotClick = (e) => {
      // Check if the clicked element has the chatbot button attributes
      if (e.target.textContent === 'Chatbot' ||
          e.target.title === 'Chatbot' ||
          e.target.getAttribute('aria-label') === 'AI Assistant' ||
          e.target.closest('[data-chatbot-button]') ||
          e.target.closest('.chatbot-navbar-button')) {
        e.preventDefault();
        e.stopPropagation();
        openChatFromNavbar();
        return false;
      }
    };

    // Add event listener to the entire document
    document.addEventListener('click', handleChatbotClick, true); // Use capture phase

    // Alternative: Set up a MutationObserver to handle dynamically added navbar items
    const observer = new MutationObserver(() => {
      const chatbotButtons = document.querySelectorAll('.chatbot-navbar-button, [aria-label="AI Assistant"], [href="#"]');
      chatbotButtons.forEach(button => {
        if (!button.dataset.chatbotListener) {
          button.dataset.chatbotListener = 'true';
          button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            openChatFromNavbar();
          });
        }
      });
    });

    // Observe for changes in the navbar
    const navbar = document.querySelector('.navbar');
    if (navbar) {
      observer.observe(navbar, {
        childList: true,
        subtree: true
      });
    }

    // Initial setup for any existing buttons
    const existingButtons = document.querySelectorAll('.chatbot-navbar-button, [aria-label="AI Assistant"], [href="#"]');
    existingButtons.forEach(button => {
      if (!button.dataset.chatbotListener) {
        button.dataset.chatbotListener = 'true';
        button.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          openChatFromNavbar();
        });
      }
    });

    // Cleanup function
    return () => {
      document.removeEventListener('click', handleChatbotClick, true);
      observer.disconnect();
    };
  }, []);

  return (
    <>
      {children}
      <RAGChatWidget />
    </>
  );
}
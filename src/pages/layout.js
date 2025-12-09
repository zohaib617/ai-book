import React from 'react';
import RAGChatWidget from '../components/RAGChatWidget';

export default function Layout({children}) {
  return (
    <>
      {children}
      <RAGChatWidget />
    </>
  );
}
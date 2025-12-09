import React from 'react';
import RAGChatWidget from '@site/src/components/RAGChatWidget';

export default function LayoutWrapper(props) {
  return (
    <>
      {props.children}
      <RAGChatWidget />
    </>
  );
}
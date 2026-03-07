'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  Input, Button, Avatar, Card, Typography, Space, Spin, message as antMessage, List,
} from 'antd';
import {
  SendOutlined, RobotOutlined, UserOutlined, LoadingOutlined,
} from '@ant-design/icons';
import type { InputRef } from 'antd';

const { TextArea } = Input;
const { Text, Paragraph } = Typography;

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  uiComponent?: 'text' | 'list';
  uiData?: any;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hi! How can I help you today?',
      sender: 'bot',
      timestamp: new Date(),
      uiComponent: 'text',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<InputRef>(null);

  useEffect(() => {
    setSessionId(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const parseResponseForUI = (text: string): { uiComponent: Message['uiComponent']; uiData: any } => {
    const lowerText = text.toLowerCase();
    if (lowerText.includes('list') || text.includes('\n- ') || text.includes('\n* ')) {
      const items = text.split('\n').filter(line => line.trim().startsWith('-') || line.trim().startsWith('*'))
        .map(line => line.replace(/^[-*]\s*/, '').trim());
      
      if (items.length > 0) {
        return {
          uiComponent: 'list',
          uiData: { items, text },
        };
      }
    }
    return { uiComponent: 'text', uiData: null };
  };

  const renderMessageContent = (message: Message) => {
    const { uiComponent, uiData, text } = message;
    switch (uiComponent) {
      case 'list':
        return (
          <div>
            <Paragraph className={message.sender === 'user' ? 'text-white' : 'text-black dark:text-white'}>
              {text.split('\n')[0]}
            </Paragraph>
            <div style={{ paddingLeft: '8px' }}>
              {uiData.items.map((item: string, index: number) => (
                <div key={index} style={{ marginBottom: '4px' }}>
                  <Text className={message.sender === 'user' ? 'text-white' : 'text-black dark:text-white'}>
                    • {item}
                  </Text>
                </div>
              ))}
            </div>
          </div>
        );
      case 'text':
      default:
        return (
          <Text
            className={message.sender === 'user' ? 'text-white' : 'text-black dark:text-white'}
            style={{
              display: 'block',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {text}
          </Text>
        );
    }
  };

const handleSend = async () => {
  if (!inputValue.trim() || isLoading) return;

  const userMessageText = inputValue.trim();
  const messageId = `msg_${Date.now()}`;
  
  const userMessage: Message = {
    id: Date.now().toString(),
    text: userMessageText,
    sender: 'user',
    timestamp: new Date(),
    uiComponent: 'text',
  };

  setMessages((prev) => [...prev, userMessage]);
  setInputValue('');
  setIsLoading(true);

  try {
    const response = await fetch('http://localhost:8000/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        threadId: sessionId,
        runId: `run_${Date.now()}`,
        state: {},
        messages: [
          {
            role: 'user',
            id: messageId,
            content: userMessageText,
          }
        ],
        tools: [],
        context: [],
        forwardedProps: {},
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('Server error:', errorData);
      throw new Error('Failed to get response from server');
    }

    const fullText = await response.text();
    let finalText = '';
    const lines = fullText.split('\n');
    
    for (const line of lines) {
      if (line.trim() === '') continue;
      
      if (line.startsWith('data: ')) {
        try {
          const jsonStr = line.slice(6).trim();
          if (jsonStr === '' || jsonStr === '[DONE]') continue;
          
          const data = JSON.parse(jsonStr);
          
          // Extract delta from TEXT_MESSAGE_CONTENT type
          if (data.type === 'TEXT_MESSAGE_CONTENT' && data.delta) {
            finalText += data.delta;
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      }
    }

    if (!finalText || finalText.trim() === '') {
      finalText = 'I received your message but had trouble generating a response. Please try again.';
    }

    const { uiComponent, uiData } = parseResponseForUI(finalText);

    const botMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: finalText,
      sender: 'bot',
      timestamp: new Date(),
      uiComponent,
      uiData,
    };

    setMessages((prev) => [...prev, botMessage]);

  } catch (error) {
    console.error('Error:', error);
    antMessage.error('Failed to connect to the chatbot');

    const errorMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: 'Sorry, I couldn\'t connect to the server. Please make sure the backend is running.',
      sender: 'bot',
      timestamp: new Date(),
      uiComponent: 'text',
    };
    setMessages((prev) => [...prev, errorMessage]);
  } finally {
    setIsLoading(false);
    inputRef.current?.focus();
  }
}; 
  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: '1',
        text: 'Hi! I\'m your AI assistant. How can I help you today?',
        sender: 'bot',
        timestamp: new Date(),
        uiComponent: 'text',
      },
    ]);
    setSessionId(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    antMessage.success('Chat cleared');
  };

  const handleQuickAction = (action: string) => {
    setInputValue(action);
    inputRef.current?.focus();
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col py-8 px-4 bg-white dark:bg-black sm:px-6">
        <div className="flex flex-col items-center gap-6 text-center mb-6 sm:items-start sm:text-left">
        </div>
    
        <div className="flex flex-col flex-1 rounded-lg overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4 bg-zinc-50 dark:bg-zinc-900" style={{ minHeight: '400px', maxHeight: '600px' }}>
            <Space orientation="vertical" style={{ width: '100%' }} size="middle">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  style={{
                    display: 'flex',
                    justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row', gap: '8px', alignItems: 'flex-start' }}>
                    <Avatar
                      size={32}
                      icon={msg.sender === 'user' ? <UserOutlined /> : <RobotOutlined />}
                      style={{
                        background: msg.sender === 'user' ? '#52c41a' : '#1890ff',
                      }}
                    />
                    <div style={{ maxWidth: '550px' }}>
                      <Card
                        size="small"
                        className={msg.sender === 'user' ? 'bg-blue-500' : 'bg-white dark:bg-zinc-800'}
                        style={{
                          borderRadius: '8px',
                          border: msg.sender === 'user' ? 'none' : undefined,
                        }}
                        styles={{ body: { padding: '12px 16px' } }}
                      >
                        {renderMessageContent(msg)}
                        <Text
                          type="secondary"
                          className={msg.sender === 'user' ? 'text-white' : 'text-zinc-500'}
                          style={{
                            fontSize: '11px',
                            opacity: 0.7,
                            display: 'block',
                            marginTop: '8px',
                          }}
                        >
                          {msg.timestamp.toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </Text>
                      </Card>
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                  <Space align="start" size="small">
                    <Avatar
                      size={32}
                      icon={<RobotOutlined />}
                      style={{ background: '#1890ff' }}
                    />
                    <Card
                      size="small"
                      className="bg-white dark:bg-zinc-800"
                      style={{ borderRadius: '8px' }}
                      styles={{ body: { padding: '12px 16px' } }}
                    >
                      <Space>
                        <Spin indicator={<LoadingOutlined spin />} size="small" />
                        <Text className="text-zinc-500 dark:text-zinc-400">Thinking...</Text>
                      </Space>
                    </Card>
                  </Space>
                </div>
              )}

              <div ref={messagesEndRef} />
            </Space>
          </div>

          {/* Input Area */}
          <div className="p-4 bg-white dark:bg-black flex justify-center">
            <Space.Compact style={{ width: '75%'}}>
              <TextArea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message... (Press Enter to send)"
                autoSize={{ minRows: 1, maxRows: 4 }}
                style={{ resize: 'none' }}
                disabled={isLoading}
                className="dark:bg-zinc-900 dark:text-white dark:border-zinc-700"
              />
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                loading={isLoading}
                style={{ height: 'auto' }}
              >
                Send
              </Button>
               <Button 
                onClick={handleClearChat} 
                size="small"
                style={{ height: 'auto' }}
              >
                Clear Chat
              </Button>
            </Space.Compact>
          </div>
        </div>

        {/* Connection Status */}
        <div className="mt-4 text-center">
          <Text className="text-xs text-zinc-400 dark:text-zinc-600">
            Session ID: {sessionId.substring(0, 20)}...
          </Text>
        </div>
      </main>
    </div>
  );
}
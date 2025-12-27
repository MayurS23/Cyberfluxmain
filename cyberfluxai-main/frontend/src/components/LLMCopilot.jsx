import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const LLMCopilot = ({ alerts }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your Security CoPilot powered by GPT-5.1. I can help you analyze threats, explain model outputs, and recommend mitigation strategies. What would you like to know?'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    
    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      // Get recent alert context
      const recentAlerts = alerts.slice(0, 5);
      const context = {
        recent_threats: recentAlerts.length,
        attack_types: [...new Set(recentAlerts.filter(a => a.is_attack).map(a => a.attack_type))]
      };

      const response = await axios.post(`${API}/copilot`, {
        question: userMessage,
        context
      });

      // Add assistant response
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: response.data.response }
      ]);
    } catch (error) {
      console.error('Copilot error:', error);
      toast.error('Failed to get response from CoPilot');
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const suggestedQuestions = [
    'What are the most critical threats right now?',
    'Explain how the ensemble model works',
    'What mitigation steps should I take for DDoS attacks?',
    'How can I improve detection accuracy?'
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6" data-testid="llm-copilot">
      {/* Chat Interface */}
      <Card className="lg:col-span-2 bg-slate-900/50 border-slate-800" data-testid="copilot-chat">
        <CardHeader>
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Security CoPilot</CardTitle>
              <CardDescription className="text-slate-400">AI-powered threat analysis assistant</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Messages */}
            <ScrollArea className="h-[500px] pr-4" ref={scrollRef} data-testid="chat-messages">
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    data-testid={`chat-message-${index}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-4 ${
                        message.role === 'user'
                          ? 'bg-cyan-600 text-white'
                          : 'bg-slate-800 text-slate-100 border border-slate-700'
                      }`}
                    >
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start" data-testid="chat-loading">
                    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                      <Loader2 className="w-5 h-5 animate-spin text-cyan-500" />
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Input */}
            <div className="flex space-x-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about threats, models, or mitigation strategies..."
                className="flex-1 bg-slate-800 border-slate-700 text-slate-100 placeholder:text-slate-500"
                disabled={loading}
                data-testid="copilot-input"
              />
              <Button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="bg-cyan-600 hover:bg-cyan-700 text-white"
                data-testid="copilot-send-btn"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Send'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Suggested Questions */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Suggested Questions</CardTitle>
          <CardDescription className="text-slate-400">Quick insights</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => setInput(question)}
                className="w-full text-left p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-slate-700 text-sm text-slate-300 hover:text-white transition-colors"
                data-testid={`suggested-question-${index}`}
              >
                {question}
              </button>
            ))}
          </div>

          <div className="mt-6 p-4 bg-slate-800/30 rounded-lg border border-slate-700">
            <h4 className="text-sm font-semibold text-slate-100 mb-2">CoPilot Capabilities</h4>
            <ul className="text-xs text-slate-400 space-y-1">
              <li>• Threat analysis and explanation</li>
              <li>• Mitigation recommendations</li>
              <li>• Model output interpretation</li>
              <li>• Security best practices</li>
              <li>• Incident response guidance</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LLMCopilot;
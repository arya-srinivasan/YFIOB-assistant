// This file is intentionally left blank.

import { getBotResponse } from './chatbot-ui.js';

document.addEventListener('DOMContentLoaded', () => {
  const messages = document.getElementById('messages');
  const input = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');

  function appendMessage(text, cls = 'user') {
    const li = document.createElement('li');
    li.className = `message ${cls}`;
    li.textContent = text;
    messages.appendChild(li);
    messages.scrollTop = messages.scrollHeight;
  }

  async function handleSend() {
        const t = input.value.trim();
        if (!t) return;

        appendMessage(t, 'user');
        input.value      = '';
        sendBtn.disabled = true;
        appendMessage('...', 'bot typing');

        try {
            const response = await getBotResponse(t);
            messages.lastChild.remove();
            appendMessage(response, 'bot');
        } catch (err) {
            messages.lastChild.remove();
            appendMessage('Something went wrong. Please try again.', 'bot');
            console.error(err);
        } finally {
            sendBtn.disabled = false;
        }
    }

  sendBtn.addEventListener('click', handleSend);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
  });
});
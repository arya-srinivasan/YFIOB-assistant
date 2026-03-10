// This file is intentionally left blank.

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

  sendBtn.addEventListener('click', () => {
    const t = input.value.trim();
    if (!t) return;
    appendMessage(t, 'user');
    input.value = '';
    setTimeout(() => appendMessage('...bot response (stub)...', 'bot'), 400);
  });
});
const modelSelect = document.getElementById("modelSelect");
const streamToggle = document.getElementById("streamToggle");
const temperatureInput = document.getElementById("temperatureInput");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");

const conversation = [];

function addMessage(role, content) {
  const el = document.createElement("article");
  el.className = `message ${role}`;
  el.textContent = content;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}

function parseTemperature() {
  const value = Number.parseFloat(temperatureInput.value);
  if (Number.isNaN(value)) {
    return 0.7;
  }
  return Math.max(0, Math.min(2, value));
}

async function loadModels() {
  const res = await fetch("/v1/models");
  if (!res.ok) {
    throw new Error("Failed to load models");
  }
  const data = await res.json();
  modelSelect.innerHTML = "";
  data.data.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.id;
    modelSelect.appendChild(option);
  });
}

function setLoading(loading) {
  sendButton.disabled = loading;
  messageInput.disabled = loading;
  modelSelect.disabled = loading;
  streamToggle.disabled = loading;
  temperatureInput.disabled = loading;
}

function extractAssistantText(payload) {
  return payload?.choices?.[0]?.message?.content ?? "";
}

function extractDeltaContent(payload) {
  return payload?.choices?.[0]?.delta?.content ?? "";
}

async function requestNonStreaming(payload, assistantEl) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.detail || "Request failed");
  }
  const text = extractAssistantText(data);
  assistantEl.textContent = text;
  conversation.push({ role: "assistant", content: text });
}

async function requestStreaming(payload, assistantEl) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    let detail = "Streaming failed";
    try {
      const body = await res.json();
      detail = body?.detail || detail;
    } catch (_) {
      detail = "Streaming failed";
    }
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      const lines = event.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) {
          continue;
        }
        const data = line.slice(6).trim();
        if (!data || data === "[DONE]") {
          continue;
        }
        const json = JSON.parse(data);
        const piece = extractDeltaContent(json);
        if (piece) {
          fullText += piece;
          assistantEl.textContent = fullText;
          chatLog.scrollTop = chatLog.scrollHeight;
        }
      }
    }
  }
  conversation.push({ role: "assistant", content: fullText });
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) {
    return;
  }

  conversation.push({ role: "user", content: text });
  addMessage("user", text);
  messageInput.value = "";

  const assistantEl = addMessage("assistant", "");
  const payload = {
    model: modelSelect.value,
    messages: conversation,
    temperature: parseTemperature(),
    stream: streamToggle.checked,
  };

  setLoading(true);
  try {
    if (payload.stream) {
      await requestStreaming(payload, assistantEl);
    } else {
      await requestNonStreaming(payload, assistantEl);
    }
  } catch (error) {
    assistantEl.className = "message system";
    assistantEl.textContent = `Error: ${error.message}`;
  } finally {
    setLoading(false);
    messageInput.focus();
  }
});

async function bootstrap() {
  try {
    await loadModels();
    addMessage("system", "Ready. Pick a model, toggle streaming mode, and start chatting.");
    messageInput.focus();
  } catch (error) {
    addMessage("system", `Startup error: ${error.message}`);
  }
}

bootstrap();

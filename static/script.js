/* PakUni Guide — frontend logic (no external libraries). */
let sessionId = null;
const messagesEl = document.getElementById("messages");
const typingEl = document.getElementById("typing");
const inputEl = document.getElementById("chat-input");
const chipsEl = document.getElementById("chips");

function addMessage(text, who, meta) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderChips(questions) {
  chipsEl.innerHTML = "";
  questions.forEach(q => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = q;
    b.addEventListener("click", () => sendMessage(q));
    chipsEl.appendChild(b);
  });
}

async function sendMessage(text) {
  const message = (text !== undefined ? text : inputEl.value).trim();
  if (!message) return;
  inputEl.value = "";
  addMessage(message, "user");
  typingEl.hidden = false;
  try {
    const res = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId })
    });
    const data = await res.json();
    typingEl.hidden = true;
    if (!data.success) {
      addMessage(data.error || "Something went wrong.", "bot");
      return;
    }
    sessionId = data.session_id;
    const meta = data.intent
      ? `intent: ${data.intent}` + (data.confidence != null ? ` (${data.confidence})` : "")
      : null;
    addMessage(data.reply, "bot", meta);
    if (data.suggestions) renderChips(data.suggestions);
  } catch (e) {
    typingEl.hidden = true;
    addMessage("Could not reach the server. Is the app running?", "bot");
  }
}

document.getElementById("chat-form").addEventListener("submit", e => {
  e.preventDefault();
  sendMessage();
});

document.getElementById("clear-btn").addEventListener("click", () => {
  messagesEl.innerHTML = "";
  sessionId = null;
  greet();
});

async function greet() {
  try {
    const res = await fetch("/api/sample-questions");
    const data = await res.json();
    if (data.success) renderChips(data.questions);
  } catch (e) { /* chips are optional */ }
  addMessage(
    "Assalam o alaikum! I'm PakUni Guide. Ask me about admissions deadlines, " +
    "merit, entry tests, fees, scholarships, programs, documents, hostels or " +
    "eligibility — or tap a suggestion below.",
    "bot"
  );
}

greet();
inputEl.focus();

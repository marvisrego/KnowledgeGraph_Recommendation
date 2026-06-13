const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const clarificationPanel = document.getElementById("clarification-panel");
const clarificationForm = document.getElementById("clarification-form");
const clarificationCopy = document.getElementById("clarification-copy");
const clearChatButton = document.getElementById("clear-chat");

let pendingId = null;
let lastUserMessage = "";

function addMessage(role, meta, text) {
    const article = document.createElement("article");
    article.className = `message message-${role}`;

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.textContent = meta;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;

    article.append(metaNode, bubble);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function showError(message) {
    const banner = document.createElement("div");
    banner.className = "error-banner";
    banner.textContent = message;
    chatLog.appendChild(banner);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function setStatus(targetId, value) {
    const target = document.getElementById(targetId);
    if (target) {
        target.textContent = value;
    }
}

async function loadStatus() {
    try {
        const response = await fetch("/api/status");
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Unable to load runtime status.");
        }
        setStatus("runtime-state", payload.ready ? "Ready" : "Unavailable");
        setStatus("llm-state", payload.llm_enabled ? "Configured" : "Deterministic only");
        setStatus("role-count", String(payload.stats.role_count ?? "-"));
        setStatus("posting-count", String(payload.stats.posting_count ?? "-"));
    } catch (error) {
        setStatus("runtime-state", "Error");
        setStatus("llm-state", "Unknown");
        showError(error.message);
    }
}

function hideClarificationPanel() {
    clarificationPanel.classList.add("hidden");
    clarificationForm.innerHTML = "";
}

function renderClarificationFields(fields, message) {
    clarificationPanel.classList.remove("hidden");
    clarificationCopy.textContent = message;
    clarificationForm.innerHTML = "";

    for (const field of fields) {
        const wrapper = document.createElement("div");
        wrapper.className = "clarification-field";

        const label = document.createElement("label");
        label.setAttribute("for", `clarify-${field.name}`);
        label.textContent = field.label;

        const input = document.createElement("input");
        input.className = "clarification-input";
        input.id = `clarify-${field.name}`;
        input.name = field.name;
        input.placeholder = field.placeholder || "";
        input.autocomplete = "off";

        wrapper.append(label, input);
        clarificationForm.appendChild(wrapper);
    }

    const actions = document.createElement("div");
    actions.className = "clarification-actions";
    const submitButton = document.createElement("button");
    submitButton.type = "submit";
    submitButton.className = "clarification-submit";
    submitButton.textContent = "Continue";
    actions.appendChild(submitButton);
    clarificationForm.appendChild(actions);
}

async function sendChatRequest(body) {
    const response = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.message || "Request failed.");
    }
    return payload;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) {
        return;
    }
    lastUserMessage = message;
    addMessage("user", "You", message);
    messageInput.value = "";
    hideClarificationPanel();

    try {
        const payload = await sendChatRequest({message});
        if (payload.status === "clarification") {
            pendingId = payload.pending_id;
            addMessage("assistant", "Assistant", payload.message);
            renderClarificationFields(payload.fields, payload.message);
            return;
        }
        pendingId = null;
        addMessage("assistant", "Assistant", payload.answer);
    } catch (error) {
        showError(error.message);
    }
});

clarificationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!pendingId) {
        hideClarificationPanel();
        return;
    }

    const formData = new FormData(clarificationForm);
    const clarifications = {};
    for (const [key, value] of formData.entries()) {
        clarifications[key] = String(value);
    }

    try {
        const payload = await sendChatRequest({
            pending_id: pendingId,
            clarifications,
            message: lastUserMessage,
        });
        if (payload.status === "clarification") {
            pendingId = payload.pending_id;
            addMessage("assistant", "Assistant", payload.message);
            renderClarificationFields(payload.fields, payload.message);
            return;
        }
        pendingId = null;
        hideClarificationPanel();
        addMessage("assistant", "Assistant", payload.answer);
    } catch (error) {
        showError(error.message);
    }
});

clearChatButton.addEventListener("click", () => {
    const intro = chatLog.querySelector(".intro");
    chatLog.innerHTML = "";
    if (intro) {
        chatLog.appendChild(intro);
    }
    pendingId = null;
    hideClarificationPanel();
});

loadStatus();
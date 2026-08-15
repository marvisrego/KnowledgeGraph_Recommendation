const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const clearChatButton = document.getElementById("clear-chat");

let chatHistory = [];

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

function createChip(text) {
    const chip = document.createElement("span");
    chip.className = "course-chip";
    chip.textContent = text;
    return chip;
}

function addCourseRecommendations(meta, courseQuery, courses) {
    if (!Array.isArray(courses) || !courses.length) {
        return;
    }

    const article = document.createElement("article");
    article.className = "message message-assistant message-courses";

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.textContent = meta;

    const panel = document.createElement("div");
    panel.className = "course-panel";

    if (courseQuery) {
        const queryNode = document.createElement("p");
        queryNode.className = "course-query";
        queryNode.textContent = `Live Coursera query: ${courseQuery}`;
        panel.appendChild(queryNode);
    }

    const grid = document.createElement("div");
    grid.className = "course-grid";

    for (const course of courses) {
        const card = document.createElement("a");
        card.className = "course-card";
        card.href = course.canonical_url;
        card.target = "_blank";
        card.rel = "noreferrer noopener";

        const header = document.createElement("div");
        header.className = "course-card-header";

        const title = document.createElement("h3");
        title.className = "course-title";
        title.textContent = course.title || "Untitled course";

        const provider = document.createElement("p");
        provider.className = "course-provider";
        provider.textContent = (course.partner_names || []).join(", ") || "Coursera";

        header.append(title, provider);

        const chipRow = document.createElement("div");
        chipRow.className = "course-chip-row";
        chipRow.appendChild(createChip((course.content_type || "course").replace(/_/g, " ")));
        if (course.estimated_workload) {
            chipRow.appendChild(createChip(course.estimated_workload));
        }
        if (course.language_primary) {
            chipRow.appendChild(createChip(course.language_primary));
        }
        if (course.shareable_certificate) {
            chipRow.appendChild(createChip("Certificate"));
        }

        const summary = document.createElement("p");
        summary.className = "course-summary";
        summary.textContent = course.description || course.tagline || "Open on Coursera";

        card.append(header, chipRow, summary);
        grid.appendChild(card);
    }

    panel.appendChild(grid);
    article.append(metaNode, panel);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function renderAssistantPayload(payload) {
    addMessage("assistant", "Assistant", payload.message || payload.answer || "No answer returned.");
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
        setStatus("llm-state", payload.chat_model || "Unknown");
        setStatus("role-count", payload.graph_nodes != null ? String(payload.graph_nodes) : "-");
        setStatus("posting-count", payload.graph_edges != null ? String(payload.graph_edges) : "-");
    } catch (error) {
        setStatus("runtime-state", "Error");
        setStatus("llm-state", "Unknown");
        showError(error.message);
    }
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
    
    chatHistory.push({role: "user", content: message});
    addMessage("user", "You", message);
    messageInput.value = "";

    try {
        const payload = await sendChatRequest({messages: chatHistory});
        
        const assistantText = payload.message || payload.answer;
        if (assistantText) {
            chatHistory.push({role: "assistant", content: assistantText});
        }
        
        renderAssistantPayload(payload);
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
    chatHistory = [];
});

loadStatus();
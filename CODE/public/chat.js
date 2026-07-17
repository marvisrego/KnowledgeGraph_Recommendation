/* Configure marked for safe markdown rendering */
marked.setOptions({ breaks: true, gfm: true });
const renderer = new marked.Renderer();
renderer.link = function(href, title, text) {
    const titleAttr = title ? ` title="${title}"` : "";
    return `<a href="${href}"${titleAttr} target="_blank" rel="noreferrer noopener">${text}</a>`;
};
marked.use({ renderer });

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

    if (role === "assistant") {
        bubble.innerHTML = marked.parse(text);
    } else {
        bubble.textContent = text;
    }

    article.append(metaNode, bubble);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

const EXTERNAL_ICON_SVG = '<svg class="external-icon" width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2H3a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-3"/><path d="M10 2h4v4"/><path d="M14 2 7 9"/></svg>';

const BOOK_ICON_SVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>';

function createChip(text) {
    const chip = document.createElement("span");
    chip.className = "course-chip";
    chip.textContent = text;
    return chip;
}

function addCourseRecommendations(courses) {
    if (!Array.isArray(courses) || !courses.length) {
        return;
    }

    const article = document.createElement("article");
    article.className = "message message-assistant message-courses";

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.textContent = "Courses";

    const panel = document.createElement("div");
    panel.className = "course-panel";

    // Section header with icon
    const header = document.createElement("div");
    header.className = "course-section-header";
    header.innerHTML = `${BOOK_ICON_SVG}<span class="course-section-title">Recommended Courses</span><span class="course-section-source">via Coursera</span>`;
    panel.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "course-grid";

    for (const course of courses) {
        const card = document.createElement("a");
        card.className = "course-card";
        card.href = course.canonical_url;
        card.target = "_blank";
        card.rel = "noreferrer noopener";

        const cardHeader = document.createElement("div");
        cardHeader.className = "course-card-header";

        const title = document.createElement("h3");
        title.className = "course-title";
        title.innerHTML = `<span>${course.title || "Untitled course"}</span>${EXTERNAL_ICON_SVG}`;

        const provider = document.createElement("p");
        provider.className = "course-provider";
        provider.textContent = (course.partner_names || []).join(", ") || "Coursera";

        cardHeader.append(title, provider);

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

        card.append(cardHeader, chipRow, summary);
        grid.appendChild(card);
    }

    panel.appendChild(grid);
    article.append(metaNode, panel);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function renderAssistantPayload(payload) {
    const text = payload.message || payload.answer || "No answer returned.";
    addMessage("assistant", "Advisor", text);

    if (payload.courses && payload.courses.length > 0) {
        addCourseRecommendations(payload.courses);
    }
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

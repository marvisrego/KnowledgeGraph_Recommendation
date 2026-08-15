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
const preloader = document.getElementById("preloader");

let chatHistory = [];

/* ─── PRELOADER ─── */
window.addEventListener("load", () => {
    setTimeout(() => {
        preloader.classList.add("hidden");
        runEntryAnimations();
    }, 800);
});

/* ─── GSAP ENTRY ANIMATIONS ─── */
function runEntryAnimations() {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reducedMotion) return;

    const heroElements = document.querySelectorAll("[data-animate='fade-up']");
    gsap.fromTo(heroElements,
        { opacity: 0, y: 24 },
        {
            opacity: 1,
            y: 0,
            duration: 0.7,
            stagger: 0.1,
            ease: "power3.out",
            delay: 0.2
        }
    );

    const chatPanel = document.getElementById("chat-panel");
    gsap.fromTo(chatPanel,
        { opacity: 0, y: 20, scale: 0.98 },
        {
            opacity: 1,
            y: 0,
            scale: 1,
            duration: 0.8,
            ease: "power3.out",
            delay: 0.5
        }
    );
}

/* ─── MESSAGES ─── */
function addMessage(role, meta, text) {
    const article = document.createElement("article");
    article.className = `message message-${role}`;

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";

    if (role === "assistant") {
        metaNode.innerHTML = `<span class="meta-avatar">AI</span> ${meta}`;
    } else {
        metaNode.textContent = meta;
    }

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

/* ─── TYPING INDICATOR ─── */
function showTypingIndicator() {
    const article = document.createElement("article");
    article.className = "message message-assistant";
    article.id = "typing-indicator";

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.innerHTML = `<span class="meta-avatar">AI</span> Advisor`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble typing-indicator";
    bubble.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;

    article.append(metaNode, bubble);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) indicator.remove();
}

/* ─── COURSE CARDS ─── */
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
    metaNode.innerHTML = `<span class="meta-avatar">AI</span> Courses`;

    const panel = document.createElement("div");
    panel.className = "course-panel";

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

/* ─── EXPLORE PANELS (partial context: skills grid + roles grid) ─── */
function addExplorePanels(explore) {
    if (!explore || (!explore.roles || !explore.roles.length) && (!explore.skills || !explore.skills.length)) return;

    const article = document.createElement("article");
    article.className = "message message-assistant message-explore";

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.innerHTML = `<span class="meta-avatar">AI</span> Explore`;

    const wrapper = document.createElement("div");
    wrapper.className = "explore-wrapper";

    // ── Skills panel ──
    if (explore.skills && explore.skills.length) {
        const panel = document.createElement("div");
        panel.className = "explore-panel";

        const header = document.createElement("div");
        header.className = "explore-panel-header";
        header.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg><span>Skills You May Need</span>`;
        panel.appendChild(header);

        const grid = document.createElement("div");
        grid.className = "explore-skill-grid";
        explore.skills.forEach(sk => {
            const chip = document.createElement("span");
            chip.className = "explore-skill-chip";
            chip.textContent = sk.title;
            grid.appendChild(chip);
        });
        panel.appendChild(grid);
        wrapper.appendChild(panel);
    }

    // ── Roles panel ──
    if (explore.roles && explore.roles.length) {
        const panel = document.createElement("div");
        panel.className = "explore-panel";

        const header = document.createElement("div");
        header.className = "explore-panel-header";
        header.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/></svg><span>Roles You Could Target</span>`;
        panel.appendChild(header);

        const track = document.createElement("div");
        track.className = "explore-role-track";
        explore.roles.forEach(role => {
            const card = document.createElement("div");
            card.className = "explore-role-card";

            const badge = document.createElement("span");
            badge.className = `path-source-badge path-source-${role.source}`;
            badge.textContent = role.source.toUpperCase();

            const title = document.createElement("div");
            title.className = "path-role-title";
            title.textContent = role.title;

            if (role.prep) {
                const prep = document.createElement("div");
                prep.className = "path-role-prep";
                prep.textContent = role.prep;
                card.append(badge, title, prep);
            } else {
                card.append(badge, title);
            }

            if (role.description) {
                const desc = document.createElement("div");
                desc.className = "explore-role-desc";
                desc.textContent = role.description;
                card.appendChild(desc);
            }

            track.appendChild(card);
        });
        panel.appendChild(track);
        wrapper.appendChild(panel);
    }

    article.append(metaNode, wrapper);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

/* ─── CAREER PATH VISUAL ─── */
function addCareerPath(path) {
    if (!path || !path.roles || !path.roles.length) return;

    const article = document.createElement("article");
    article.className = "message message-assistant message-path";

    const metaNode = document.createElement("div");
    metaNode.className = "message-meta";
    metaNode.innerHTML = `<span class="meta-avatar">AI</span> Career Path`;

    const panel = document.createElement("div");
    panel.className = "path-panel";

    const header = document.createElement("div");
    header.className = "path-header";
    header.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg><span>Recommended roles</span><span class="path-header-order">accessible → aspirational</span>`;
    panel.appendChild(header);

    const track = document.createElement("div");
    track.className = "path-track";

    path.roles.forEach((role, i) => {
        // Role node
        const roleNode = document.createElement("div");
        roleNode.className = "path-role";

        const cardTop = document.createElement("div");
        cardTop.className = "path-card-top";

        const badge = document.createElement("span");
        badge.className = `path-source-badge path-source-${role.source}`;
        badge.textContent = (role.source || "KG").toUpperCase();
        cardTop.appendChild(badge);

        if (Number.isFinite(role.accessibility)) {
            const accessBadge = document.createElement("span");
            accessBadge.className = "path-accessibility-badge";
            accessBadge.textContent = `${Math.round(role.accessibility * 100)}% ready`;
            cardTop.appendChild(accessBadge);
        }

        const title = document.createElement("div");
        title.className = "path-role-title";
        title.textContent = role.title;

        if (role.prep) {
            const prep = document.createElement("div");
            prep.className = "path-role-prep";
            prep.textContent = role.prep;
            roleNode.append(cardTop, title, prep);
        } else {
            roleNode.append(cardTop, title);
        }

        // Evidence rail: one compact place for skill and population transition signals.
        const evidence = document.createElement("div");
        evidence.className = "path-evidence";
        if (Number.isFinite(role.accessibility)) {
            const readiness = document.createElement("div");
            readiness.className = "path-readiness";
            const readinessLabel = document.createElement("div");
            readinessLabel.className = "path-readiness-label";
            readinessLabel.textContent = `${role.have_count || 0} of ${role.required_skill_count || 0} required skills matched`;
            const meter = document.createElement("div");
            meter.className = "path-readiness-meter";
            meter.setAttribute("role", "progressbar");
            meter.setAttribute("aria-label", "Skill readiness");
            meter.setAttribute("aria-valuemin", "0");
            meter.setAttribute("aria-valuemax", "100");
            meter.setAttribute("aria-valuenow", String(Math.round(role.accessibility * 100)));
            const fill = document.createElement("span");
            fill.style.width = `${Math.max(0, Math.min(100, role.accessibility * 100))}%`;
            meter.appendChild(fill);
            readiness.append(readinessLabel, meter);
            evidence.appendChild(readiness);
        } else {
            const unavailable = document.createElement("div");
            unavailable.className = "path-evidence-unavailable";
            unavailable.textContent = role.gap_evidence === "no_user_skills"
                ? "Add a matching skill to calculate readiness"
                : "Comparable skill evidence unavailable";
            evidence.appendChild(unavailable);
        }

        if (role.transition) {
            const transition = document.createElement("div");
            transition.className = "path-transition-proof";
            const count = Number(role.transition.count || 0).toLocaleString();
            const probability = (Number(role.transition.probability || 0) * 100).toFixed(1);
            transition.textContent = `${count} observed moves · ${probability}%`;
            transition.title = "Observed population-level career transitions in the Karrierewege training split";
            evidence.appendChild(transition);
        }
        roleNode.appendChild(evidence);

        const appendSkillGroup = (label, skills, state) => {
            if (!skills || !skills.length) return;
            const group = document.createElement("div");
            group.className = "path-skill-group";
            const groupLabel = document.createElement("div");
            groupLabel.className = `path-skill-label path-skill-label-${state}`;
            groupLabel.textContent = label;
            const skillRow = document.createElement("div");
            skillRow.className = "path-skill-row";
            skills.forEach(sk => {
                const chip = document.createElement("span");
                chip.className = `path-skill-chip path-skill-${state}`;
                chip.textContent = sk.title;
                skillRow.appendChild(chip);
            });
            group.append(groupLabel, skillRow);
            roleNode.appendChild(group);
        };

        appendSkillGroup("Already have", role.have, "have");
        appendSkillGroup("Develop next", role.need, "need");

        // Backward-compatible fallback for paths built before skill-gap evidence.
        if ((!role.have || !role.have.length) && (!role.need || !role.need.length)) {
            const roleSkills = (path.skills || []).filter(s => s.roles.includes(role.id));
            appendSkillGroup("Role skills", roleSkills, "neutral");
        }

        track.appendChild(roleNode);

        // Arrow connector between roles (not after last)
        if (i < path.roles.length - 1) {
            const arrow = document.createElement("div");
            arrow.className = "path-arrow";
            arrow.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>`;
            track.appendChild(arrow);
        }
    });

    panel.appendChild(track);
    article.append(metaNode, panel);
    chatLog.appendChild(article);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function renderAssistantPayload(payload) {
    const text = payload.message || payload.answer || "No answer returned.";
    addMessage("assistant", "Advisor", text);

    if (payload.explore && (payload.explore.skills || payload.explore.roles)) {
        addExplorePanels(payload.explore);
    }

    if (payload.path && payload.path.roles && payload.path.roles.length > 0) {
        addCareerPath(payload.path);
    }

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
        if (targetId === "runtime-state") {
            const dot = target.querySelector(".status-dot");
            target.textContent = "";
            if (dot) target.appendChild(dot);
            target.appendChild(document.createTextNode(" " + value));
        } else {
            target.textContent = value;
        }
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
        setStatus("role-count", payload.graph_nodes != null ? Number(payload.graph_nodes).toLocaleString() : "-");
        setStatus("posting-count", payload.graph_edges != null ? Number(payload.graph_edges).toLocaleString() : "-");
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

    showTypingIndicator();

    try {
        const payload = await sendChatRequest({messages: chatHistory});

        hideTypingIndicator();

        const assistantText = payload.message || payload.answer;
        if (assistantText) {
            chatHistory.push({role: "assistant", content: assistantText});
        }

        renderAssistantPayload(payload);
    } catch (error) {
        hideTypingIndicator();
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

const SVG_NS = "http://www.w3.org/2000/svg";
const SAFE_MARKDOWN_TAGS = new Set([
    "P", "BR", "STRONG", "EM", "B", "I", "CODE", "PRE", "UL", "OL", "LI",
    "A", "BLOCKQUOTE", "H1", "H2", "H3", "H4", "HR", "TABLE", "THEAD",
    "TBODY", "TR", "TH", "TD"
]);
const DROP_MARKDOWN_TAGS = new Set([
    "SCRIPT", "STYLE", "IFRAME", "OBJECT", "EMBED", "FORM", "INPUT", "BUTTON",
    "SVG", "MATH", "LINK", "META"
]);
const ICONS = {
    book: [
        ["path", {d: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20"}],
        ["path", {d: "M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"}]
    ],
    briefcase: [
        ["rect", {x: "2", y: "7", width: "20", height: "14", rx: "2"}],
        ["path", {d: "M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"}]
    ],
    compass: [
        ["circle", {cx: "12", cy: "12", r: "3"}],
        ["path", {d: "M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"}]
    ],
    external: [
        ["path", {d: "M6 2H3a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-3"}],
        ["path", {d: "M10 2h4v4M14 2 7 9"}]
    ],
    route: [
        ["circle", {cx: "6", cy: "6", r: "2.5"}],
        ["circle", {cx: "18", cy: "6", r: "2.5"}],
        ["circle", {cx: "12", cy: "18", r: "2.5"}],
        ["path", {d: "M6 8.5v2A7.5 7.5 0 0 0 12 18M18 8.5v2A7.5 7.5 0 0 1 12 18"}]
    ],
    star: [
        ["path", {d: "m12 2 3.1 6.3 6.9 1-5 4.8 1.2 6.9-6.2-3.2L5.8 21 7 14.1 2 9.3l6.9-1Z"}]
    ],
    arrow: [
        ["path", {d: "M5 12h14M13 6l6 6-6 6"}]
    ]
};

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const clearChatButton = document.getElementById("clear-chat");
const sendButton = document.getElementById("send-btn");
const sendText = sendButton.querySelector(".send-text");
const composerStatus = document.getElementById("composer-status");
const runtimeMessage = document.getElementById("runtime-message");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let chatHistory = [];
let activeRequest = null;
let conversationVersion = 0;

function createIcon(name, size, className) {
    const icon = document.createElementNS(SVG_NS, "svg");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("width", String(size || 16));
    icon.setAttribute("height", String(size || 16));
    icon.setAttribute("fill", "none");
    icon.setAttribute("stroke", "currentColor");
    icon.setAttribute("stroke-width", "1.7");
    icon.setAttribute("stroke-linecap", "round");
    icon.setAttribute("stroke-linejoin", "round");
    icon.setAttribute("aria-hidden", "true");
    if (className) icon.setAttribute("class", className);

    for (const descriptor of ICONS[name] || []) {
        const node = document.createElementNS(SVG_NS, descriptor[0]);
        for (const [key, value] of Object.entries(descriptor[1])) {
            node.setAttribute(key, value);
        }
        icon.appendChild(node);
    }
    return icon;
}

function createMeta(label) {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const avatar = document.createElement("span");
    avatar.className = "meta-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "AI";
    const text = document.createElement("span");
    text.textContent = label;
    meta.append(avatar, text);
    return meta;
}

function createPanelHeader(iconName, title, trailingText) {
    const header = document.createElement("div");
    header.className = "panel-header";
    header.appendChild(createIcon(iconName, 17));
    const heading = document.createElement("span");
    heading.className = "panel-title";
    heading.textContent = title;
    header.appendChild(heading);
    if (trailingText) {
        const trailing = document.createElement("span");
        trailing.className = "panel-trailing";
        trailing.textContent = trailingText;
        header.appendChild(trailing);
    }
    return header;
}

function safeUrl(value, allowedProtocols) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
        const url = new URL(value, window.location.origin);
        const protocols = allowedProtocols || ["https:", "http:"];
        return protocols.includes(url.protocol) ? url.href : null;
    } catch (_error) {
        return null;
    }
}

function sanitizeMarkdown(html) {
    const template = document.createElement("template");
    template.innerHTML = html;
    const elements = Array.from(template.content.querySelectorAll("*"));

    for (const element of elements) {
        if (DROP_MARKDOWN_TAGS.has(element.tagName)) {
            element.remove();
            continue;
        }
        if (!SAFE_MARKDOWN_TAGS.has(element.tagName)) {
            element.replaceWith(...element.childNodes);
            continue;
        }

        const href = element.tagName === "A" ? element.getAttribute("href") : null;
        const title = element.tagName === "A" ? element.getAttribute("title") : null;
        for (const attribute of Array.from(element.attributes)) {
            element.removeAttribute(attribute.name);
        }

        if (element.tagName === "A") {
            const sanitizedHref = safeUrl(href, ["https:", "http:", "mailto:"]);
            if (!sanitizedHref) {
                element.replaceWith(...element.childNodes);
                continue;
            }
            element.setAttribute("href", sanitizedHref);
            if (title) element.setAttribute("title", title);
            if (!sanitizedHref.startsWith("mailto:")) {
                element.setAttribute("target", "_blank");
                element.setAttribute("rel", "noreferrer noopener");
            }
        }
    }
    return template.content;
}

function renderAssistantText(target, text) {
    const content = String(text || "");
    if (!window.marked || typeof window.marked.parse !== "function") {
        target.textContent = content;
        target.classList.add("plain-text-fallback");
        return;
    }
    try {
        const parsed = window.marked.parse(content, {breaks: true, gfm: true});
        target.replaceChildren(sanitizeMarkdown(parsed));
    } catch (_error) {
        target.textContent = content;
        target.classList.add("plain-text-fallback");
    }
}

function scrollToLatest(node) {
    window.requestAnimationFrame(() => {
        chatLog.scrollTo({
            top: chatLog.scrollHeight,
            behavior: reducedMotion.matches ? "auto" : "smooth"
        });
        if (node && chatLog.scrollHeight <= chatLog.clientHeight + 4) {
            node.scrollIntoView({block: "nearest", behavior: reducedMotion.matches ? "auto" : "smooth"});
        }
    });
}

function addMessage(role, meta, text) {
    const article = document.createElement("article");
    article.className = "message message-" + role;

    if (role === "assistant") {
        article.appendChild(createMeta(meta));
    } else {
        const metaNode = document.createElement("div");
        metaNode.className = "message-meta";
        metaNode.textContent = meta;
        article.appendChild(metaNode);
    }

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (role === "assistant") {
        renderAssistantText(bubble, text);
    } else {
        bubble.textContent = text;
    }
    article.appendChild(bubble);
    chatLog.appendChild(article);
    scrollToLatest(article);
    return article;
}

function showTypingIndicator() {
    hideTypingIndicator();
    const article = document.createElement("article");
    article.className = "message message-assistant";
    article.id = "typing-indicator";
    article.setAttribute("role", "status");
    article.setAttribute("aria-label", "Advisor is preparing a response");

    const bubble = document.createElement("div");
    bubble.className = "message-bubble typing-indicator";
    for (let index = 0; index < 3; index += 1) {
        const dot = document.createElement("span");
        dot.className = "typing-dot";
        dot.setAttribute("aria-hidden", "true");
        bubble.appendChild(dot);
    }
    const label = document.createElement("span");
    label.className = "typing-label";
    label.textContent = "Mapping your options";

    article.append(createMeta("Advisor"), bubble, label);
    chatLog.appendChild(article);
    scrollToLatest(article);
}

function hideTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) indicator.remove();
}

function enableHorizontalTrack(track, label) {
    track.tabIndex = 0;
    track.setAttribute("role", "region");
    track.setAttribute("aria-label", label);
    track.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const direction = document.dir === "rtl" ? -1 : 1;
        if (event.key === "Home") {
            track.scrollTo({left: 0, behavior: reducedMotion.matches ? "auto" : "smooth"});
        } else if (event.key === "End") {
            track.scrollTo({left: track.scrollWidth, behavior: reducedMotion.matches ? "auto" : "smooth"});
        } else {
            const amount = Math.max(track.clientWidth * 0.72, 220);
            const sign = event.key === "ArrowRight" ? 1 : -1;
            track.scrollBy({left: amount * sign * direction, behavior: reducedMotion.matches ? "auto" : "smooth"});
        }
    });
}

function wrapTrack(track) {
    const shell = document.createElement("div");
    shell.className = "track-shell";
    shell.appendChild(track);
    const hint = document.createElement("p");
    hint.className = "track-hint";
    hint.textContent = "Scroll or use arrow keys";
    shell.appendChild(hint);
    return shell;
}

function createChip(text, className) {
    const chip = document.createElement("span");
    chip.className = className || "course-chip";
    chip.textContent = String(text || "");
    return chip;
}

function addCourseRecommendations(courses) {
    if (!Array.isArray(courses) || courses.length === 0) return;

    const article = document.createElement("article");
    article.className = "message message-assistant message-courses";
    const panel = document.createElement("div");
    panel.className = "evidence-panel course-panel";
    panel.appendChild(createPanelHeader("book", "Recommended courses", "via Coursera"));

    const grid = document.createElement("div");
    grid.className = "course-grid";

    for (const course of courses) {
        const courseUrl = safeUrl(course.canonical_url);
        const card = document.createElement(courseUrl ? "a" : "article");
        card.className = "course-card";
        if (courseUrl) {
            card.href = courseUrl;
            card.target = "_blank";
            card.rel = "noreferrer noopener";
        } else {
            card.classList.add("is-disabled");
            card.setAttribute("aria-disabled", "true");
        }

        const cardHeader = document.createElement("div");
        cardHeader.className = "course-card-header";
        const title = document.createElement("h3");
        title.className = "course-title";
        const titleText = document.createElement("span");
        titleText.textContent = course.title || "Untitled course";
        title.appendChild(titleText);
        if (courseUrl) title.appendChild(createIcon("external", 14, "external-icon"));

        const provider = document.createElement("p");
        provider.className = "course-provider";
        provider.textContent = Array.isArray(course.partner_names) && course.partner_names.length
            ? course.partner_names.join(", ")
            : "Coursera";
        cardHeader.append(title, provider);

        const chipRow = document.createElement("div");
        chipRow.className = "course-chip-row";
        chipRow.appendChild(createChip(String(course.content_type || "course").replace(/_/g, " ")));
        if (course.estimated_workload) chipRow.appendChild(createChip(course.estimated_workload));
        if (course.language_primary) chipRow.appendChild(createChip(course.language_primary));
        if (course.shareable_certificate) chipRow.appendChild(createChip("Certificate"));

        const summary = document.createElement("p");
        summary.className = "course-summary";
        summary.textContent = course.description || course.tagline || "View course details";

        card.append(cardHeader, chipRow, summary);
        grid.appendChild(card);
    }

    panel.appendChild(grid);
    article.append(createMeta("Courses"), panel);
    chatLog.appendChild(article);
    scrollToLatest(article);
}

function normalizedSource(source) {
    return String(source || "").toLowerCase() === "onet" ? "onet" : "esco";
}

function createSourceBadge(source) {
    const badge = document.createElement("span");
    const normalized = normalizedSource(source);
    badge.className = "path-source-badge path-source-" + normalized;
    badge.textContent = normalized === "onet" ? "O*NET" : "ESCO";
    return badge;
}

function addExplorePanels(explore) {
    const skills = explore && Array.isArray(explore.skills) ? explore.skills : [];
    const roles = explore && Array.isArray(explore.roles) ? explore.roles : [];
    if (skills.length === 0 && roles.length === 0) return;

    const article = document.createElement("article");
    article.className = "message message-assistant message-explore";
    const wrapper = document.createElement("div");
    wrapper.className = "explore-wrapper";

    if (skills.length) {
        const panel = document.createElement("section");
        panel.className = "evidence-panel explore-panel";
        panel.appendChild(createPanelHeader("star", "Skills you may need"));
        const grid = document.createElement("div");
        grid.className = "explore-skill-grid";
        for (const skill of skills) {
            grid.appendChild(createChip(skill.title, "explore-skill-chip"));
        }
        panel.appendChild(grid);
        wrapper.appendChild(panel);
    }

    if (roles.length) {
        const panel = document.createElement("section");
        panel.className = "evidence-panel explore-panel";
        panel.appendChild(createPanelHeader("briefcase", "Roles you could target"));
        const track = document.createElement("div");
        track.className = "horizontal-track explore-role-track";
        enableHorizontalTrack(track, "Suggested roles. Use left and right arrow keys to browse.");

        for (const role of roles) {
            const card = document.createElement("article");
            card.className = "explore-role-card";
            const title = document.createElement("h3");
            title.className = "path-role-title";
            title.textContent = role.title || "Untitled role";
            card.append(createSourceBadge(role.source), title);
            if (role.prep) {
                const prep = document.createElement("p");
                prep.className = "path-role-prep";
                prep.textContent = role.prep;
                card.appendChild(prep);
            }
            if (role.description) {
                const description = document.createElement("p");
                description.className = "explore-role-desc";
                description.textContent = role.description;
                card.appendChild(description);
            }
            track.appendChild(card);
        }
        panel.appendChild(wrapTrack(track));
        wrapper.appendChild(panel);
    }

    article.append(createMeta("Explore"), wrapper);
    chatLog.appendChild(article);
    scrollToLatest(article);
}

function appendSkillGroup(roleNode, label, skills, state) {
    if (!Array.isArray(skills) || skills.length === 0) return;
    const group = document.createElement("div");
    group.className = "path-skill-group";
    const groupLabel = document.createElement("div");
    groupLabel.className = "path-skill-label path-skill-label-" + state;
    groupLabel.textContent = label;
    const skillRow = document.createElement("div");
    skillRow.className = "path-skill-row";
    for (const skill of skills) {
        skillRow.appendChild(createChip(skill.title, "path-skill-chip path-skill-" + state));
    }
    group.append(groupLabel, skillRow);
    roleNode.appendChild(group);
}

function addCareerPath(path) {
    if (!path || !Array.isArray(path.roles) || path.roles.length === 0) return;

    const article = document.createElement("article");
    article.className = "message message-assistant message-path";
    const panel = document.createElement("section");
    panel.className = "evidence-panel path-panel";
    panel.appendChild(createPanelHeader("route", "Recommended roles", "accessible → aspirational"));

    const track = document.createElement("div");
    track.className = "horizontal-track path-track";
    enableHorizontalTrack(track, "Recommended career path ordered from accessible to aspirational.");

    path.roles.forEach((role, index) => {
        const roleNode = document.createElement("article");
        roleNode.className = "path-role";

        const cardTop = document.createElement("div");
        cardTop.className = "path-card-top";
        cardTop.appendChild(createSourceBadge(role.source));
        const hasAccessibility = typeof role.accessibility === "number" && Number.isFinite(role.accessibility);
        if (hasAccessibility) {
            const accessBadge = document.createElement("span");
            accessBadge.className = "path-accessibility-badge";
            accessBadge.textContent = Math.round(role.accessibility * 100) + "% ready";
            cardTop.appendChild(accessBadge);
        }
        if (role.effort_band) {
            const effortBadge = document.createElement("span");
            effortBadge.className = "path-effort-badge effort-" + role.effort_band;
            effortBadge.textContent = role.effort_band.charAt(0).toUpperCase() + role.effort_band.slice(1) + " effort";
            if (typeof role.effort_score === "number") {
                effortBadge.title = "Transition Effort Score: " + (role.effort_score * 100).toFixed(0) + "%";
            }
            cardTop.appendChild(effortBadge);
        }

        const title = document.createElement("h3");
        title.className = "path-role-title";
        title.textContent = role.title || "Untitled role";
        roleNode.append(cardTop, title);
        if (role.prep) {
            const prep = document.createElement("p");
            prep.className = "path-role-prep";
            prep.textContent = role.prep;
            roleNode.appendChild(prep);
        }

        const evidence = document.createElement("div");
        evidence.className = "path-evidence";
        if (hasAccessibility) {
            const readiness = document.createElement("div");
            readiness.className = "path-readiness";
            const readinessLabel = document.createElement("div");
            readinessLabel.className = "path-readiness-label";
            readinessLabel.textContent = String(role.have_count || 0) + " of " + String(role.required_skill_count || 0) + " required skills matched";
            const meter = document.createElement("div");
            const percentage = Math.max(0, Math.min(100, role.accessibility * 100));
            meter.className = "path-readiness-meter";
            meter.setAttribute("role", "progressbar");
            meter.setAttribute("aria-label", "Skill readiness for " + title.textContent);
            meter.setAttribute("aria-valuemin", "0");
            meter.setAttribute("aria-valuemax", "100");
            meter.setAttribute("aria-valuenow", String(Math.round(percentage)));
            const fill = document.createElement("span");
            fill.style.width = percentage + "%";
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
            if (role.transition.evidence_type === "semantic_transition_backoff") {
                const support = Number(role.transition.neighbour_support || 0).toLocaleString();
                transition.classList.add("is-inferred");
                transition.textContent = "Semantic transition evidence · " + support + " related roles";
                transition.title = "Inferred from training transitions of semantically related ESCO roles; not a directly observed move";
            } else {
                const count = Number(role.transition.count || 0).toLocaleString();
                const probability = (Number(role.transition.probability || 0) * 100).toFixed(1);
                transition.textContent = count + " observed moves · " + probability + "%";
                transition.title = "Population-level career transitions observed in the Karrierewege training split";
            }
            evidence.appendChild(transition);
        }
        roleNode.appendChild(evidence);

        appendSkillGroup(roleNode, "Already have", role.have, "have");
        appendSkillGroup(roleNode, "Develop next", role.need, "need");
        if ((!role.have || role.have.length === 0) && (!role.need || role.need.length === 0)) {
            const fallbackSkills = Array.isArray(path.skills)
                ? path.skills.filter((skill) => Array.isArray(skill.roles) && skill.roles.includes(role.id))
                : [];
            appendSkillGroup(roleNode, "Role skills", fallbackSkills, "neutral");
        }
        track.appendChild(roleNode);

        if (index < path.roles.length - 1) {
            const arrow = document.createElement("div");
            arrow.className = "path-arrow";
            arrow.setAttribute("aria-hidden", "true");
            arrow.appendChild(createIcon("arrow", 20));
            track.appendChild(arrow);
        }
    });

    panel.appendChild(wrapTrack(track));
    article.append(createMeta("Career path"), panel);
    chatLog.appendChild(article);
    scrollToLatest(article);
}

function renderAssistantPayload(payload) {
    const text = payload.message || payload.answer || "No answer was returned.";
    addMessage("assistant", "Advisor", text);
    if (payload.explore && (payload.explore.skills || payload.explore.roles)) {
        addExplorePanels(payload.explore);
    }
    if (payload.path && Array.isArray(payload.path.roles) && payload.path.roles.length) {
        addCareerPath(payload.path);
    }
    if (Array.isArray(payload.explanations) && payload.explanations.length) {
        addExplanationChains(payload.explanations);
    }
    if (Array.isArray(payload.courses) && payload.courses.length) {
        addCourseRecommendations(payload.courses);
    }
}

function addExplanationChains(explanations) {
    const article = document.createElement("article");
    article.className = "chat-bubble assistant";
    const panel = document.createElement("div");
    panel.className = "explanation-panel";
    const header = createPanelHeader("graph", "Why these roles?", "");
    panel.appendChild(header);

    explanations.forEach((exp) => {
        const details = document.createElement("details");
        details.className = "explanation-details";
        const summary = document.createElement("summary");
        summary.className = "explanation-summary";
        summary.textContent = exp.role_title || "Role";
        details.appendChild(summary);

        const chain = document.createElement("div");
        chain.className = "explanation-chain";
        (exp.steps || []).forEach((step) => {
            const line = document.createElement("div");
            line.className = "explanation-step";
            const relation = step.relation || "";
            const target = step.target || "";
            const attrs = step.attributes || {};
            let text = "";
            if (relation === "TRANSITIONS_TO") {
                const pct = ((attrs.probability || 0) * 100).toFixed(1);
                text = "→ TRANSITIONS_TO (" + pct + "%, n=" + (attrs.count || 0) + ") → " + target;
            } else if (relation === "SIMILAR_TO") {
                text = "→ SIMILAR_TO (" + (attrs.similarity || 0).toFixed(2) + ") → " + target;
            } else if (relation === "REQUIRES") {
                const marker = (attrs.status || "").includes("have") ? "(you have this)" : "(you need this)";
                line.classList.add(attrs.transferable ? "step-have" : "step-need");
                text = "→ REQUIRES → " + target + " " + marker;
            }
            line.textContent = text;
            chain.appendChild(line);
        });
        details.appendChild(chain);
        panel.appendChild(details);
    });

    article.append(createMeta("Evidence"), panel);
    chatLog.appendChild(article);
    scrollToLatest(article);
}

function showError(message, retry) {
    const banner = document.createElement("div");
    banner.className = "error-banner";
    banner.setAttribute("role", "alert");
    const text = document.createElement("span");
    text.textContent = message;
    banner.appendChild(text);
    if (typeof retry === "function") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "error-retry";
        button.textContent = "Try again";
        button.addEventListener("click", () => {
            banner.remove();
            retry();
        }, {once: true});
        banner.appendChild(button);
    }
    chatLog.appendChild(banner);
    scrollToLatest(banner);
}

function setRuntimeState(state, label) {
    const target = document.getElementById("runtime-state");
    if (!target) return;
    target.dataset.state = state;
    const text = target.querySelector(".status-text");
    if (text) text.textContent = label;
}

function setText(targetId, value) {
    const target = document.getElementById(targetId);
    if (target) target.textContent = value;
}

async function parseJsonResponse(response, fallbackMessage) {
    const raw = await response.text();
    let payload = {};
    if (raw) {
        try {
            payload = JSON.parse(raw);
        } catch (_error) {
            if (!response.ok) throw new Error(fallbackMessage);
            throw new Error("The server returned an unreadable response. Please try again.");
        }
    }
    if (!response.ok) {
        throw new Error(payload.message || payload.error || fallbackMessage);
    }
    return payload;
}

async function loadStatus() {
    setRuntimeState("loading", "Loading");
    runtimeMessage.textContent = "";
    try {
        const response = await fetch("/api/status");
        const payload = await parseJsonResponse(response, "Unable to load runtime status.");
        setRuntimeState(payload.ready ? "ready" : "unavailable", payload.ready ? "Ready" : "Unavailable");
        setText("llm-state", payload.chat_model || "Unknown");
        setText("role-count", payload.graph_nodes != null ? Number(payload.graph_nodes).toLocaleString() : "—");
        setText("edge-count", payload.graph_edges != null ? Number(payload.graph_edges).toLocaleString() : "—");
        runtimeMessage.textContent = payload.ready
            ? "Graph and semantic index are available."
            : (payload.error || "Connect the configured graph and vector stores to enable advice.");
    } catch (error) {
        setRuntimeState("error", "Error");
        setText("llm-state", "Unknown");
        runtimeMessage.textContent = error.message || "Runtime status is unavailable.";
    }
}

async function sendChatRequest(body, signal) {
    const response = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
        signal: signal
    });
    return parseJsonResponse(response, "The advisor could not complete this request.");
}

function setRequestState(isLoading) {
    chatLog.setAttribute("aria-busy", String(isLoading));
    chatForm.setAttribute("aria-busy", String(isLoading));
    sendButton.disabled = isLoading;
    messageInput.readOnly = isLoading;
    sendButton.classList.toggle("is-loading", isLoading);
    sendText.textContent = isLoading ? "Thinking…" : "Get advice";
    composerStatus.textContent = isLoading ? "Mapping roles, evidence, and skill gaps…" : "";
}

function retryMessage(message) {
    const last = chatHistory[chatHistory.length - 1];
    if (last && last.role === "user" && last.content === message) {
        chatHistory.pop();
    }
    messageInput.value = message;
    messageInput.focus();
    chatForm.requestSubmit();
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (activeRequest) return;
    const message = messageInput.value.trim();
    if (!message) {
        messageInput.focus();
        return;
    }

    const request = {
        controller: new AbortController(),
        version: conversationVersion
    };
    activeRequest = request;
    chatHistory.push({role: "user", content: message});
    addMessage("user", "You", message);
    messageInput.value = "";
    setRequestState(true);
    showTypingIndicator();

    try {
        const payload = await sendChatRequest({messages: chatHistory}, request.controller.signal);
        if (request.version !== conversationVersion || activeRequest !== request) return;
        hideTypingIndicator();
        const assistantText = payload.message || payload.answer;
        if (assistantText) chatHistory.push({role: "assistant", content: assistantText});
        renderAssistantPayload(payload);
    } catch (error) {
        if (error.name === "AbortError" || request.version !== conversationVersion) return;
        hideTypingIndicator();
        const messageText = error instanceof TypeError
            ? "The advisor is unreachable. Check your connection and try again."
            : (error.message || "The advisor could not complete this request.");
        showError(messageText, () => retryMessage(message));
    } finally {
        if (activeRequest === request) {
            activeRequest = null;
            hideTypingIndicator();
            setRequestState(false);
            messageInput.focus();
        }
    }
});

clearChatButton.addEventListener("click", () => {
    conversationVersion += 1;
    if (activeRequest) activeRequest.controller.abort();
    activeRequest = null;
    setRequestState(false);
    hideTypingIndicator();
    const intro = chatLog.querySelector(".intro");
    chatLog.replaceChildren();
    if (intro) chatLog.appendChild(intro);
    chatHistory = [];
    messageInput.value = "";
    composerStatus.textContent = "New conversation started.";
    messageInput.focus();
});

function runEntryAnimations() {
    if (reducedMotion.matches || !window.gsap) return;
    const heroElements = document.querySelectorAll("[data-animate='fade-up']");
    window.gsap.fromTo(heroElements,
        {opacity: 0, y: 18},
        {opacity: 1, y: 0, duration: 0.55, stagger: 0.07, ease: "power3.out"}
    );
    window.gsap.fromTo(document.getElementById("chat-panel"),
        {opacity: 0, y: 14},
        {opacity: 1, y: 0, duration: 0.62, ease: "power3.out", delay: 0.12}
    );
}

runEntryAnimations();
loadStatus();

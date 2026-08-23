"use strict";

const ui = {
    workspace: document.getElementById("graph-canvas"), canvas: document.getElementById("cy"), state: document.getElementById("graph-state"),
    stateTitle: document.getElementById("state-title"), stateCopy: document.getElementById("state-copy"),
    retry: document.getElementById("retry-button"), stats: document.getElementById("stats-text"),
    inspector: document.getElementById("node-inspector"), inspectorClose: document.getElementById("inspector-close"),
    inspectorType: document.getElementById("inspector-type"), inspectorTitle: document.getElementById("inspector-title"),
    inspectorDescription: document.getElementById("inspector-description"), inspectorFacts: document.getElementById("inspector-facts"),
    tooltip: document.getElementById("node-tooltip"), tooltipTitle: document.getElementById("tooltip-title"),
    tooltipType: document.getElementById("tooltip-type"), fit: document.getElementById("btn-fit"),
    zoomIn: document.getElementById("btn-zoom-in"), zoomOut: document.getElementById("btn-zoom-out"),
};

let graph = null;
let loadController = null;
const supportsHover = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

function count(value) {
    return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "0";
}

function showState(title, copy, isError) {
    ui.state.hidden = false;
    ui.state.classList.toggle("is-error", Boolean(isError));
    ui.stateTitle.textContent = title;
    ui.stateCopy.textContent = copy;
    ui.retry.hidden = !isError;
}

function clearSelection() {
    ui.inspector.hidden = true;
    ui.workspace.classList.remove("inspector-open");
    ui.tooltip.hidden = true;
    if (graph) graph.elements().removeClass("dimmed highlighted show-label");
}

function addFact(label, value) {
    if (value === undefined || value === null || String(value).trim() === "") return;
    const row = document.createElement("div");
    const key = document.createElement("span");
    const content = document.createElement("strong");
    row.className = "inspector-fact";
    key.textContent = label;
    content.textContent = String(value);
    row.append(key, content);
    ui.inspectorFacts.append(row);
}

function inspectNode(node) {
    const data = node.data();
    const source = data.source ? " · " + String(data.source).toUpperCase() : "";
    ui.inspectorType.textContent = (data.type || "node") + source;
    ui.inspectorTitle.textContent = data.label || "Untitled node";
    ui.inspectorDescription.textContent = data.description || "No additional description is available for this node.";
    ui.inspectorFacts.replaceChildren();
    addFact("Preparation", data.job_zone);
    addFact("Source", data.source ? String(data.source).toUpperCase() : "");
    addFact("Connections", node.degree());
    ui.inspector.hidden = false;
    ui.workspace.classList.add("inspector-open");
}

function positionTooltip(event) {
    if (!event || !event.originalEvent || ui.tooltip.hidden) return;
    const margin = 12;
    const bounds = ui.tooltip.getBoundingClientRect();
    const left = Math.min(event.originalEvent.clientX + 16, window.innerWidth - bounds.width - margin);
    const top = Math.min(Math.max(margin, event.originalEvent.clientY - 8), window.innerHeight - bounds.height - margin);
    ui.tooltip.style.left = Math.max(margin, left) + "px";
    ui.tooltip.style.top = top + "px";
}

function bindInteractions(instance) {
    instance.on("tap", "node", function (event) {
        const selected = event.target;
        const neighbourhood = selected.closedNeighborhood();
        instance.elements().removeClass("dimmed highlighted show-label").addClass("dimmed");
        neighbourhood.removeClass("dimmed").addClass("highlighted");
        selected.addClass("show-label");
        inspectNode(selected);
    });
    instance.on("tap", function (event) { if (event.target === instance) clearSelection(); });
    instance.on("zoom", function () {
        instance.nodes("[type='skill'], [type='element'], [type='skill_group']")
            .toggleClass("zoom-label", instance.zoom() >= 1.35);
    });
    if (supportsHover) {
        instance.on("mouseover", "node", function (event) {
            const data = event.target.data();
            ui.tooltipTitle.textContent = data.label || "Untitled node";
            ui.tooltipType.textContent = (data.type || "node") + (data.source ? " · " + String(data.source).toUpperCase() : "");
            ui.tooltip.hidden = false;
            positionTooltip(event);
        });
        instance.on("mousemove", "node", positionTooltip);
        instance.on("mouseout", "node", function () { ui.tooltip.hidden = true; });
    }
}

function createGraph(data) {
    graph = window.cytoscape({
        container: ui.canvas,
        elements: { nodes: data.nodes, edges: data.edges },
        style: [
            { selector: "node", style: {
                "label": "data(label)", "font-family": "Outfit, sans-serif", "font-size": "11px", "font-weight": 500,
                "text-valign": "bottom", "text-margin-y": "7px", "text-wrap": "wrap", "text-max-width": "112px",
                "text-background-color": "#070A10", "text-background-opacity": .84, "text-background-padding": "2px",
                "color": "#A7B7C8", "width": 20, "height": 20,
            }},
            { selector: "node[source='onet']", style: { "background-color": "#4CC2EA", "width": 25, "height": 25, "color": "#EAF1F8" }},
            { selector: "node[source='esco'][type='role']", style: { "background-color": "#55D6A0", "width": 25, "height": 25, "color": "#EAF1F8" }},
            { selector: "node[type='skill'], node[type='element'], node[type='skill_group']", style: { "label": "", "shape": "diamond", "background-color": "#F2B45B", "width": 11, "height": 11 }},
            { selector: "node.orphan", style: { "label": "" }},
            { selector: "node.zoom-label, node.show-label", style: { "label": "data(label)", "font-size": "9px" }},
            { selector: "edge", style: { "width": 1, "line-color": "#38506A", "target-arrow-color": "#38506A", "target-arrow-shape": "triangle", "arrow-scale": .58, "curve-style": "bezier", "opacity": .44 }},
            { selector: "edge[relation='SIMILAR_TO']", style: { "line-color": "#4CC2EA", "target-arrow-color": "#4CC2EA", "width": 1.4, "opacity": .68 }},
            { selector: "edge[relation='TRANSITIONS_TO']", style: { "line-color": "#F2B45B", "target-arrow-color": "#F2B45B", "line-style": "dashed", "width": 1.7, "opacity": .78 }},
            { selector: "node:selected, node.highlighted", style: { "border-width": 2, "border-color": "#EAF1F8", "opacity": 1 }},
            { selector: "node.dimmed, edge.dimmed", style: { "opacity": .06 }},
        ],
        layout: { name: "cose", animate: false, nodeDimensionsIncludeLabels: true, nodeRepulsion: 9000, idealEdgeLength: 94, edgeElasticity: 90, gravity: .18, numIter: 650, randomize: true },
        wheelSensitivity: .26, minZoom: .12, maxZoom: 4,
    });
    graph.nodes().filter(function (node) { return node.degree() === 0; }).addClass("orphan");
    ui.canvas.setAttribute("tabindex", "0");
    bindInteractions(graph);
}

async function loadGraph() {
    if (loadController) loadController.abort();
    if (graph) { graph.destroy(); graph = null; }
    clearSelection();
    showState("Mapping the career network", "Loading a representative sample of roles, skills, and transitions.", false);
    ui.stats.textContent = "Preparing graph data…";
    if (typeof window.cytoscape !== "function") {
        showState("Graph renderer unavailable", "The graph library could not be loaded. Check your connection, then try again.", true);
        return;
    }
    loadController = new AbortController();
    try {
        const response = await fetch("/api/graph-data", { signal: loadController.signal });
        if (!response.ok) throw new Error("The server returned " + response.status + ".");
        const data = await response.json();
        if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges) || !data.nodes.length) throw new Error("No graph sample was returned.");
        const stats = data.stats || {};
        ui.stats.textContent = "Showing " + count(stats.shown_nodes || data.nodes.length) + " nodes · " + count(stats.shown_edges || data.edges.length) +
            " edges · sampled from " + count(stats.total_nodes || data.nodes.length) + " nodes and " + count(stats.total_edges || data.edges.length) + " edges";
        createGraph(data);
        ui.state.hidden = true;
    } catch (error) {
        if (error && error.name === "AbortError") return;
        showState("Unable to load the graph", error && error.message ? error.message : "An unexpected error occurred.", true);
        ui.stats.textContent = "Graph data unavailable";
    } finally {
        loadController = null;
    }
}

ui.fit.addEventListener("click", function () { if (graph) graph.fit(undefined, 48); });
ui.zoomIn.addEventListener("click", function () { if (graph) graph.zoom({ level: graph.zoom() * 1.25, renderedPosition: { x: graph.width() / 2, y: graph.height() / 2 } }); });
ui.zoomOut.addEventListener("click", function () { if (graph) graph.zoom({ level: graph.zoom() * .8, renderedPosition: { x: graph.width() / 2, y: graph.height() / 2 } }); });
ui.inspectorClose.addEventListener("click", clearSelection);
ui.retry.addEventListener("click", loadGraph);
loadGraph();

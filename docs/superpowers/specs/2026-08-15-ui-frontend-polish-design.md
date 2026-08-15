# Career Graph Studio UI and Frontend Polish Design

Date: 2026-08-15
Status: Approved design, pending implementation-plan review
Target branch: `dev`

## 1. Purpose

Modernize the existing Career Graph Studio interface into a production-ready, cohesive, responsive product while preserving its routes, API contracts, content order, graph behavior, and dark graph-science identity.

This is a targeted evolution, not a visual restart. The evidence-rich career path remains the product's visual signature. Decorative effects must support the career-advice task instead of competing with it.

## 2. User and product job

The primary users are students, graduates, career changers, and researchers exploring evidence-backed career options. The main page has one job: help a user describe their background and understand a trustworthy next career step. The graph page has one job: let a user inspect the sampled knowledge structure without confusing it for the entire graph.

## 3. Existing contracts

The implementation must preserve:

- `GET /` and `GET /graph`.
- `GET /api/status`, `POST /api/chat`, and `GET /api/graph-data` without payload changes.
- Conversation history and the existing New Chat flow.
- Response order: narrative, explore panels, career path, then courses.
- Partial-context recommendations, readiness evidence, owned skills, skill gaps, transition counts and probabilities, and accessible-to-aspirational ordering.
- Markdown formatting and safe external links.
- Graph pan, zoom, fit, node inspection, neighborhood highlighting, background reset, legend categories, and all existing edge types.
- Flask's existing `public/` static source of truth.

No backend, graph, retrieval, ranking, model, data-processing, or business-logic changes are part of this work.

## 4. Audit summary

The rendered and code audits identified these priority defects:

1. The graph header has no responsive rules and clips severely at tablet and mobile widths.
2. The fixed preloader can block the entire chat interface when an external script fails.
3. Chat requests can overlap, and an in-flight response can repopulate a newly cleared conversation.
4. Several controls are smaller than 44px, lack a consistent focus treatment, or rely on hover and `title` text.
5. Important small text and the primary gradient button do not consistently meet WCAG AA contrast.
6. All graph labels render simultaneously at 8-10px, producing collisions and unreadable disconnected rows.
7. Horizontal recommendation tracks have weak keyboard and continuation affordances.
8. Assistant Markdown and several interpolated values enter `innerHTML` without an allowlist.
9. The graph page has a separate palette, typography system, and component treatment.
10. `.agents/`, `.claude/`, `CODE/`, and `static/` are repository tooling or stale duplicates, not runtime sources.

## 5. Chosen direction

Design read: a preservation-focused product UI with a dark graph-science identity, moderate information density, and restrained motion.

Design dials:

- Design variance: 4/10. Stable and asymmetric enough to distinguish context from conversation.
- Motion intensity: 3/10. Feedback and state transitions only.
- Visual density: 5/10. Evidence remains visible without turning the interface into a dashboard.

The rejected alternatives were:

- CSS-only polish. It cannot fix request lifecycle, sanitization, preloader failure, or graph accessibility.
- A new application shell. It carries unnecessary route, structure, and interaction risk.

## 6. Visual system

### 6.1 Palette

The application stays dark-only to preserve its established identity.

- Deep space: `#070A10` for the page background.
- Graphite: `#0E141E` for primary panels.
- Slate: `#151E2B` for raised and interactive surfaces.
- Frost: `#EAF1F8` for primary text.
- Cyan: `#4CC2EA` for the single brand accent, focus, links, and primary action.
- Semantic colors: green for available/owned, amber for development/transition, and red for errors.

Indigo may remain only as a quiet tonal endpoint in the existing brand mark or evidence meter. It must not become a competing action color. Borders and shadows use cool blue-gray values derived from the surface palette.

### 6.2 Typography

- Outfit remains the display and interface face.
- JetBrains Mono remains the evidence, metric, source, and relationship face.
- Mobile interface text is at least 14px except nonessential metadata that has an accessible label.
- Body copy is constrained to a comfortable reading measure.
- The spelling `O*NET` is used consistently in visible interface copy.

### 6.3 Shape, spacing, and elevation

- Panels use 16px radii, controls and inputs use 10px radii, and compact badges use full pills.
- A consistent 4/8/12/16/24/32px spacing scale replaces isolated values.
- Elevation is reserved for the main workspace and genuinely interactive cards.
- Noninteractive status and evidence blocks do not lift on hover.
- The background keeps one restrained cyan atmospheric wash. Multiple large animated blobs and strong outer glows are removed or reduced.

## 7. Main chat layout

### Desktop, 1024px and wider

```text
+----------------------+  +------------------------------------------+
| Context and identity |  | Advisor toolbar                          |
| Product explanation  |  +------------------------------------------+
| Graph link            |  | Conversation and evidence               |
| Runtime evidence      |  |                                          |
| Data-source badges    |  |                                          |
+----------------------+  +------------------------------------------+
                          | Labeled composer and primary action       |
                          +------------------------------------------+
```

The context panel stays sticky. The conversation workspace receives a bounded desktop height, `min-height: 0`, a reliable scroll region, and a composer that remains reachable.

### Tablet, 721-1023px

The layout becomes one column. The context panel becomes a compact horizontal introduction with a two-column runtime summary. The chat immediately follows it and uses the remaining viewport without fixed-height assumptions.

### Mobile, 720px and narrower

The introduction is compressed, status values remain a two-column grid when content permits, and long values wrap safely. The chat toolbar and composer stack without clipping. Evidence and course cards become single-column or snap-scrolling collections as appropriate. Safe-area padding and `100dvh` are used where viewport height matters.

## 8. Chat components and states

### 8.1 Entry and dependency fallback

The preloader becomes progressive enhancement rather than a blocking dependency. The application shell is usable by default. A short entrance may run after DOM readiness, but it must have an unconditional timeout and a reduced-motion path. Missing Marked or GSAP cannot hide the application.

If Marked is unavailable, assistant output renders as readable plain text. GSAP is optional; the page remains static and complete without it.

### 8.2 Request lifecycle

One request controller owns the active chat request.

- Submitting disables the send button and prevents duplicate requests.
- The textarea remains readable and receives `aria-busy` or associated status feedback.
- The typing state has one unique instance and an accessible label.
- New Chat aborts or invalidates the active request before clearing the UI.
- A stale response cannot enter a newer conversation.
- Controls return to their normal state in a `finally` path.
- Non-JSON and network failures become concise recovery messages.

### 8.3 Messages and generated content

Assistant Markdown is parsed only when Marked exists, then passed through a local allowlist sanitizer. The sanitizer keeps the supported semantic tags and removes scripts, event handlers, unsafe attributes, and non-HTTP(S)/mailto link protocols. User text and API-provided course titles continue to use text nodes.

Markdown gains consistent styling for headings, blockquotes, lists, inline code, code blocks, tables, long links, and horizontal overflow.

### 8.4 Evidence and recommendations

The career-path rail is the signature component:

- Readiness, transition evidence, existing skills, and development gaps share one clear hierarchy.
- Readiness is communicated by text and meter, never color alone.
- Horizontal ordering and arrows remain intact on wide screens.
- Tracks receive a label, keyboard focus, scroll snapping, stronger scrollbars, and a visible continuation cue.
- On narrow screens, cards either stack or snap one at a time without hiding content beyond the viewport.

Explore and course panels reuse the same panel header, surface, spacing, and focus patterns. Course cards preserve their external destinations and gain clear focus, hover, active, and long-content handling.

### 8.5 Status and feedback

Runtime status uses semantic classes for loading, ready, unavailable, and error. The indicator is green only when ready. Error banners use `role="alert"`; transient status uses a polite live region. Disabled states remain legible and do not rely on opacity alone.

## 9. Graph workspace

The graph page adopts the shared palette, Outfit/JetBrains Mono typography, spacing, radii, focus system, and component surfaces while remaining a focused canvas rather than a second marketing page.

### 9.1 Header and legend

- Desktop uses a compact title row with back action and sampled statistics, plus a separate legend row when space requires it.
- Tablet and mobile wrap safely instead of compressing title text into columns.
- The legend can scroll horizontally on narrow screens and accurately shows dashed transition edges.
- The sampled wording remains explicit.

### 9.2 Canvas readability

- Role labels are visible by default.
- Skill labels appear at a useful zoom or when selected/connected.
- Labels use wrapping, maximum widths, and minimum zoomed font sizes.
- Node source/type is distinguished by shape as well as color.
- Existing semantic blue, green, and amber categories remain.
- Layout tuning improves separation without changing sampled data.

### 9.3 Inspection and controls

- Fit, zoom in, and zoom out remain fixed and become 44px controls with visible focus and `aria-label` values.
- Hover still previews node details on pointer devices.
- Tap or click opens a persistent, viewport-safe inspector so touch users can read the same information.
- Selecting another node clears the previous highlighted state before applying the new neighborhood.
- Background tap keeps the existing reset behavior.

### 9.4 Loading and failure

The graph loader validates `response.ok`, JSON shape, and empty data. Errors use text nodes, explain recovery, and provide a retry control. Missing Cytoscape produces a usable error state rather than an empty canvas. Reduced motion disables loader animation.

## 10. Accessibility and performance

- All interactive controls receive a consistent `:focus-visible` ring.
- Touch targets are at least 44x44px on touch layouts.
- Text and controls target WCAG 2.1 AA contrast.
- Color is never the only carrier of readiness, source, or status.
- Decorative SVGs are hidden from assistive technology; icon-only controls have accessible names.
- Dynamic regions use appropriate live-region behavior without announcing large result panels repeatedly.
- `prefers-reduced-motion` disables automatic motion and smooth scrolling.
- A reduced-transparency fallback replaces backdrop filters with solid surfaces.
- Motion is limited to transform and opacity and serves hierarchy, feedback, or state change.
- Mobile removes expensive continuous ambient animation.

## 11. Code organization

The implementation remains dependency-free and frontend-local:

- `templates/index.html`: semantic shell, accessibility hooks, and existing content.
- `public/chat.css`: authoritative tokens, layouts, shared panel patterns, responsive rules, states, and reduced-motion/transparency behavior.
- `public/chat.js`: safe DOM helpers, request lifecycle, rendering helpers, and existing API integration.
- `templates/graph.html`: self-contained graph workspace styles and interactions, using values aligned with the shared system.

No new framework, icon package, build step, API, or backend route is introduced. Shared rendering helpers replace repeated assistant-message and scrolling code where this can be done without changing output order.

## 12. Repository cleanup

The following tracked directories are removed because they are not runtime sources:

- `.agents/`: local agent skill packages.
- `.claude/`: editor-specific agent files.
- `CODE/`: stale root-application mirror.
- `static/`: stale asset mirror; Flask serves `public/`.

They are added to `.gitignore` to prevent accidental recommits. Relevant thesis documentation, tests, evaluation scripts, and Karrierewege result artifacts remain.

## 13. Verification plan

Automated checks:

- `node --check public/chat.js`.
- Python compilation and the existing unit-test suite.
- Flask test-client checks for `/`, `/graph`, `/api/status`, and `/api/graph-data`.
- Static scans for unsafe `innerHTML` interpolation, missing button labels, duplicate IDs, stale directory references, and unhandled non-OK fetches.

Rendered checks with the installed system Chrome:

- Chat and graph at 1440x900 or larger desktop.
- Chat and graph at approximately 820x1180 tablet.
- Chat and graph at 390x844 mobile.
- Long model names, long role/course titles, empty lists, status error, chat error, loading, partial context, full career path, and course cards.
- Keyboard traversal, visible focus, horizontal-track access, touch-sized controls, reduced motion, and CDN failure fallbacks.

Existing functional checks:

- New Chat resets history and ignores an old request.
- A second submit cannot overlap the first.
- Narrative, explore, path, and course output order is unchanged.
- Career evidence values and graph relation styles remain correct.
- Pan, zoom, fit, selection, neighborhood highlighting, and reset still work.

## 14. Acceptance criteria

The work is complete when:

1. No existing route, API payload, recommendation behavior, or graph interaction is removed.
2. Both pages share a coherent token, typography, spacing, shape, and focus system.
3. Neither page clips or creates unintended document-level horizontal scrolling at the tested breakpoints.
4. The chat remains usable when optional CDN resources fail.
5. Request state prevents duplicates and stale post-reset responses.
6. Generated Markdown and external links are sanitized without losing supported formatting.
7. All essential controls and text meet the documented accessibility targets.
8. Graph labels, header, legend, controls, and inspection work on desktop, tablet, and mobile.
9. Reduced-motion and reduced-transparency paths are present.
10. The repository cleanup is committed separately from the subsequent UI implementation where practical.

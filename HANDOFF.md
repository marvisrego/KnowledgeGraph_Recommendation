# GraphRAG Career Advisor — Session Handoff

## Project State (as of 2026-07-17)

Flask-based GraphRAG career chatbot for a thesis. **Fully built and running.**

- **Branch:** `dev`
- **Remote:** `https://github.com/marvisrego/KnowledgeGraph_Recommendation.git`
- **Server:** `python career_kg_web.py` → `http://127.0.0.1:8001`
- **Graph:** built and on disk — do NOT rebuild unless data changes
- **ChromaDB:** built and on disk — do NOT re-embed unless graph changes

---

## File Map

```
config.py                     Settings dataclass — reads from .env
src/
  onet_preprocessing.py       ONET Excel → nodes/edges DataFrames
  esco_preprocessing.py       ESCO CSV → nodes/edges DataFrames
  graph_build.py              Merge into nx.DiGraph; pickle save/load
  embeddings_index.py         Azure embed → ChromaDB; cosine alignment
  inference_pipeline.py       6-step pipeline (see below)
build_graph.py                CLI: python build_graph.py [--embed] [--align-only]
career_kg_web.py              Flask server — port 8001
coursera_client.py            Coursera search scraper (no API key needed)
templates/index.html          Chat UI (redesigned — Inter font, clean dark palette)
templates/graph.html          Interactive Cytoscape.js knowledge graph viewer
public/chat.js                Frontend JS — marked.js markdown, course cards wired
public/chat.css               Frontend CSS — redesigned, no glassmorphism/glows
test_apis.py                  Smoke-test for Azure embed + Cohere rerank
CODE/                         Clean deployable copy of all source files (git-safe)
```

---

## Graph Stats

| Stat | Value |
|---|---|
| Total nodes | 22,259 |
| Total edges | 235,501 |
| Role nodes | 4,055 (1,016 ONET + 3,039 ESCO) |
| SIMILAR_TO edges | 25,262 (12,631 pairs × 2 directions) |
| ChromaDB collection | 4,055 role embeddings |
| Similarity threshold | 0.45 (cosine, ONET↔ESCO alignment) |

**Output paths (already exist — do not rebuild unless told to):**
```
graph/graph.gpickle       NetworkX DiGraph
index/chroma/             ChromaDB persistent store
```

---

## API Configuration (.env)

```
CHAT_MODEL_API_KEY      = a312329574de4c5c9c103bf282bb1b79
CHAT_MODEL_ENDPOINT     = https://career.azure-api.net/career-graph-ai/openai/v1/responses
CHAT_MODEL              = gpt-5.4-nano

EMBED_MODEL_API_KEY     = a312329574de4c5c9c103bf282bb1b79
EMBED_MODEL_ENDPOINT    = https://career.azure-api.net/career-graph-ai
EMBED_MODEL             = text-embedding-3-large

COHERE_RERANK_API_KEY   = a312329574de4c5c9c103bf282bb1b79
COHERE_RERANK_ENDPOINT  = https://career.azure-api.net/career-graph-ai/providers/cohere/v2/rerank
COHERE_RERANK_MODEL     = Cohere-rerank-v4.0-pro

SIMILARITY_THRESHOLD    = 0.45
```

**API shape details:**
- Chat: OpenAI **Responses API** — POST to full endpoint, body `{"model":..., "input":[...messages...], "max_output_tokens":...}`, response at `output[*].type=="message" → content[0].text`
- Embed: POST to `EMBED_MODEL_ENDPOINT + "/openai/v1/embeddings"`, body `{"model":..., "input":[...]}`
- Rerank: POST to `COHERE_RERANK_ENDPOINT`, headers `{"api-key":...}`, body `{"model":..., "query":..., "documents":[...], "top_n":N}`, response at `results[*].relevance_score`
- All HTTP via `urllib.request` — no OpenAI SDK

**gpt-5.4-nano is a reasoning model** — it consumes tokens on internal chain-of-thought before outputting text. Token budgets in `inference_pipeline.py` are set accordingly:
- Routing: `max_tokens=2000`
- Generation: `max_tokens=4000`

**Available chat models on this Azure gateway** (all support Responses API):
`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4-pro`, `gpt-5.4`, `gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.2`, `gpt-5.1`, `gpt-5-pro`, `gpt-5`, `gpt-chat-latest`
*Avoid `-codex` and `-reasoning` variants.*

---

## 6-Step Inference Pipeline (`src/inference_pipeline.py`)

```
Full conversation history (messages[])
  │
  ▼
1. route_intent()        — Responses API: detect student vs professional,
                           check if enough context; returns follow-up Qs if not.
                           Passes full history so model remembers prior turns.
  │ {"has_context": true, "user_type": "student"|"professional"}
  ▼
2. retrieve_candidates() — Embed last query → ChromaDB top-50 role nodes
  │
  ▼
3. rerank_candidates()   — Cohere rerank → top-5 anchors
  │
  ▼
4. traverse_graph()      — NetworkX: for each anchor collect
                           REQUIRES (ONET IM≥3.0, ESCO essential+optional)
                           BELONGS_TO, BROADER_THAN, NARROWER_THAN, SIMILAR_TO
  │
  ▼
5. generate_response()   — Responses API with user_type-specific system prompt:
                           STUDENT  → conversational careers advisor tone
                           PROFESSIONAL → conversational career coach tone
                           Both: 3–4 natural paragraphs, no bullet lists/headers
                           Full history passed so model has conversation context.
  │
  ▼
6. fetch_coursera_courses() — scrape Coursera for anchor role titles +
                              essential skills → return list[dict] (structured)
```

**`run_query()` now returns a dict**, not a plain string:
```python
{"message": str, "courses": list[dict]}
```
`career_kg_web.py` unpacks this and sends `{"status": "ok", "message": "...", "courses": [...]}` to the frontend.

**Conversation memory:** the web layer sends the full `messages[]` array from the client on every request. Both routing and generation receive the full history so the model never re-asks for information already given.

---

## Prompts Summary

### Routing (`_INTENT_SYSTEM_PROMPT`)
- Detects STUDENT / PROFESSIONAL / UNKNOWN from conversation history
- STUDENT needs: degree + skill + goal
- PROFESSIONAL needs: current role + skill + next step
- Returns pure JSON: `{"has_context": bool, "user_type": "...", "followup_questions": "..."}`

### Generation (conversational tone — no markdown headings or bullets)
- `_GENERATION_SYSTEM_PROMPT_STUDENT` — speaks as a university careers advisor; 3–4 paragraphs; bold for role/skill names only; 2–3 role recommendations max
- `_GENERATION_SYSTEM_PROMPT_PROFESSIONAL` — speaks as a senior career coach; same format; mentions skill gaps and stepping-stone roles conversationally
- **Strict graph-only constraint:** every role/skill cited must appear in the graph triples

---

## Frontend

### Design (redesigned 2026-07-17)
- **Palette:** dark slate (`#0E1117` base, `#161B24` panels, `#1C2333` raised), blue accent `#4E8EE8`
- **Font:** Inter only (dropped Space Grotesk and JetBrains Mono for labels)
- **Removed:** scanning grid animation, glassmorphism blur, neon button glows, gradient h1
- **Panels:** solid opaque background, single `1px` border — clean editorial feel

### Markdown rendering
- `marked.js` CDN loaded in `index.html`; assistant bubbles use `innerHTML = marked.parse(text)`
- User bubbles still use `.textContent` (safe)

### Coursera card section
- `addCourseRecommendations(courses)` in `chat.js` — was dead code, now called from `renderAssistantPayload` when `payload.courses.length > 0`
- Each card: title + external-link icon, provider, chips (type/workload/language/certificate), description
- Section header with inline SVG book icon + "Recommended Courses · via Coursera"

### Knowledge Graph Viewer (`/graph`)
- Cytoscape.js full-screen interactive graph
- "Explore Knowledge Graph" button in hero sidebar opens `/graph` in a new tab
- `/api/graph-data` samples top-60 ONET + top-60 ESCO roles by degree + up to 160 skill nodes (essential REQUIRES edges only)
- Color coding: blue = ONET, green = ESCO, amber = skills; blue edges = SIMILAR_TO, grey = REQUIRES
- Click node → highlight neighbourhood; hover → tooltip with description + prep level
- Fit / zoom controls bottom-right; stats bar shows sampled vs total counts

---

## Node / Edge Schema

**Node types:** `role` (ONET+ESCO), `element` (ONET), `skill` (ESCO), `skill_group` (ESCO), `isco_group` (ESCO)

**Edge types:**
| relation | meaning | key attrs |
|---|---|---|
| REQUIRES | role → skill/element | `requirement_level` (float for ONET ≥3.0, 'essential'/'optional' for ESCO), `domain`, `source` |
| BELONGS_TO | ESCO role → isco_group | `source='esco'` |
| BROADER_THAN | concept → broader concept | `pillar`, `source` |
| NARROWER_THAN | concept → narrower concept | `pillar`, `source` |
| RELATED_TO | skill → skill | `source='esco'` |
| SIMILAR_TO | ONET role ↔ ESCO role | `similarity` float, `source='alignment'` |

---

## Running the Project

```bash
# Start server (graph already built)
python career_kg_web.py
# → http://127.0.0.1:8001

# Check health
curl http://127.0.0.1:8001/api/status
# → {"ready": true, "graph_loaded": true, "chroma_loaded": true, ...}

# Smoke-test APIs
python test_apis.py

# Kill stale server processes
python -c "
import subprocess
out = subprocess.check_output('netstat -ano', shell=True).decode()
for line in out.splitlines():
    if ':8001' in line and 'LISTEN' in line:
        pid = line.strip().split()[-1]
        subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
        print('Killed', pid)
"
```

**Rebuild only if needed:**
```bash
python build_graph.py             # rebuild graph only (no API calls)
python build_graph.py --embed     # rebuild graph + ChromaDB (calls Azure embed API ~41 batches)
python build_graph.py --align-only  # recompute SIMILAR_TO edges only (calls Azure embed API)
```

---

## Verification Checklist

```bash
# Graph sanity
python -c "
import pickle; G=pickle.load(open('graph/graph.gpickle','rb'))
print(G.number_of_nodes(), G.number_of_edges())
bad=[e for _,_,e in G.edges(data=True) if e.get('relation')=='REQUIRES' and e.get('source')=='onet' and str(e.get('requirement_level',99)).replace('.','',1).isdigit() and float(e.get('requirement_level',99))<3.0]
print('Bad ONET edges (should be 0):', len(bad))
"
# Expected: 22259 235501 / Bad ONET edges: 0

# API response shape check
curl -s -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I finished a BSc in Computer Science, I know Python and SQL, I want to go into data science."}]}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('status:', d['status']); print('courses:', len(d.get('courses',[])));  print('msg[:200]:', d['message'][:200])"
# Expected: status: ok, courses: 2-3, message starts with natural prose (no ## or ***)
```

---

## Git

```bash
# Commit author: marvisrego / marvisregodab@gmail.com
git config user.name "marvisrego"
git config user.email "marvisregodab@gmail.com"

# Push to dev
git push origin dev
```

---

## Known Issues / TODO

- [ ] `vercel.json` still references old `build_vercel_bundle.py` — update to `python build_graph.py --embed` for deployment
- [ ] `static/` directory mirrors `public/` — redundant, can be deleted
- [ ] Coursera client scrapes HTML (no API key) — may break if Coursera changes their page structure
- [ ] `career_kg_chat.py` is gutted (just a comment) — delete if not needed
- [ ] ESCO ISCO hierarchy join uses `iscoGroup` code string vs URI — verify edges are correct after any rebuild
- [ ] Knowledge graph viewer uses `cose` layout which is slow for >300 nodes — consider pre-computing layout positions and caching as JSON

---

## Example Prompts That Work

**Student:**
> "I just graduated with a BSc in Computer Science, I know Python, SQL, and basic machine learning. I want to go into data science or AI roles."

**Professional:**
> "I am a data analyst with 4 years experience using Python, SQL, and Tableau. I want to switch to a data science or machine learning role."

**Pattern:** `"I am a [role/degree] with [experience/year], I know [skills], and I want to [goal]."`

---

## Session Changes (2026-07-17)

1. **LLM tone rewrite** — both generation prompts rewritten to conversational mentor style; no `##` headings, no bullet lists; 2–3 role recommendations in natural paragraphs
2. **Coursera decoupled** — `_fetch_coursera_recommendations` replaced by `fetch_coursera_courses()` returning `list[dict]`; `run_query()` now returns `{"message", "courses"}` dict; web layer passes `courses` as separate JSON field
3. **Markdown rendering** — `marked.js` added; assistant bubbles render HTML instead of plain text; user bubbles unchanged
4. **Frontend redesign** — full CSS rewrite: new palette, Inter-only font, no animations/glows/glassmorphism; copy updated ("Get Advice", "New Chat", etc.)
5. **Knowledge graph viewer** — new `/graph` route + `templates/graph.html` with Cytoscape.js; new `/api/graph-data` endpoint sampling top-degree roles + essential skills; "Explore Knowledge Graph" button added to sidebar
6. **CODE/ folder created** — clean deployable copy of all source files with `.env.example` and `.gitignore`
7. **Git setup** — remote added (`marvisrego/KnowledgeGraph_Recommendation`), user set to `marvisrego`, committed and pushed to `dev`

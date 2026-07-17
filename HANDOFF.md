# GraphRAG Career Advisor — Session Handoff

## Project State (as of 2026-07-14)

Flask-based GraphRAG career chatbot for a thesis. **Fully built and running.**

- **Branch:** `dev`
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
templates/index.html          Chat UI
public/chat.js / chat.css     Frontend assets
test_apis.py                  Smoke-test for Azure embed + Cohere rerank
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
                           STUDENT  → entry-level roles, education, pathways
                           PROFESSIONAL → transitions, skill gaps, stepping stones
                           Full history passed so model has conversation context.
  │
  ▼
6. _fetch_coursera_recommendations() — scrape Coursera search for anchor role
                                       titles + essential skills → append 3 courses
```

**Conversation memory:** the web layer sends the full `messages[]` array from the client on every request. Both routing and generation receive the full history so the model never re-asks for information already given.

---

## Prompts Summary

### Routing (`_INTENT_SYSTEM_PROMPT`)
- Detects STUDENT / PROFESSIONAL / UNKNOWN from conversation history
- STUDENT needs: degree + skill + goal
- PROFESSIONAL needs: current role + skill + next step
- Returns pure JSON: `{"has_context": bool, "user_type": "...", "followup_questions": "..."}`

### Generation
- `_GENERATION_SYSTEM_PROMPT_STUDENT` — entry-level roles, education context, pathways to grow
- `_GENERATION_SYSTEM_PROMPT_PROFESSIONAL` — transitions, skill gaps vs ESCO essentials, stepping stones
- **Strict graph-only constraint:** every role/skill cited must appear in the graph triples

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

# Kill stale server processes (run in Python if taskkill /F doesn't work from bash)
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
```

---

## Known Issues / TODO

- [ ] `vercel.json` still references old `build_vercel_bundle.py` — update to `python build_graph.py --embed` for deployment
- [ ] `static/` directory mirrors `public/` — redundant, can be deleted
- [ ] Coursera client scrapes HTML (no API key) — may break if Coursera changes their page structure
- [ ] `career_kg_chat.py` is gutted (just a comment) — delete if not needed
- [ ] ESCO ISCO hierarchy join uses `iscoGroup` code string vs URI — verify edges are correct after any rebuild

---

## Example Prompts That Work

**Student:**
> "I just graduated with a BSc in Computer Science, I know Python, SQL, and basic machine learning. I want to go into data science or AI roles."

**Professional:**
> "I am a data analyst with 4 years experience using Python, SQL, and Tableau. I want to switch to a data science or machine learning role."

**Pattern:** `"I am a [role/degree] with [experience/year], I know [skills], and I want to [goal]."`

# GraphRAG Career Advisor — Session Handoff

## Project State (as of 2026-08-15)

Flask-based GraphRAG career chatbot for a thesis. **Fully built and running.**

- **Branch:** `dev`
- **Remote:** `https://github.com/marvisrego/KnowledgeGraph_Recommendation.git`
- **Server:** `python career_kg_web.py` → `http://127.0.0.1:8001`
- **Graph:** enriched 2026-08-15 with train-only Karrierewege transitions
- **ChromaDB:** role content is unchanged; runtime filters 123 stale pre-pruning records until a future index rebuild

### Implemented Novelty Contributions

See `novelty.md` and `ALL_STEPS.md` for the design, implementation, and evaluation record.

1. **Empirical Career Transition Edges** — 18,907 support-filtered `TRANSITIONS_TO` edges learned from training trajectories only
2. **Skill-Gap-Aware Career Path Ranking** — deterministic owned-skill evidence, ESCO-comparable requirements, and accessible-to-aspirational ordering

Validation and test remain held out. Direct-edge test baseline: Hits@5 `0.3702`, Hits@10 `0.5014`, MRR `0.2591`. Accepted embedding-smoothed results: Hits@5 `0.3722`, Hits@10 `0.5058`, MRR `0.2617`, source-role coverage `1.0000`, destination coverage `0.9532`.

---

## File Map

```
config.py                     Settings dataclass — reads from .env
src/
  onet_preprocessing.py       ONET Excel → nodes/edges DataFrames
  esco_preprocessing.py       ESCO CSV → nodes/edges DataFrames
  graph_build.py              Merge into typed nx.MultiDiGraph; prune; atomic pickle save/load
  embeddings_index.py         Azure embed → ChromaDB; cosine alignment
  karrierewege_preprocessing.py  Stream-clean trajectories; aggregate/add transitions
  transition_embedding.py     Semantic-neighbour smoothing over train-derived transitions
  skill_gap.py                Skill ownership, comparable requirements, accessibility ranking
  inference_pipeline.py       7-step pipeline (see below)
evaluation/
  evaluate_karrierewege.py    Held-out transition evaluation
  evaluate_embedding_transitions.py  Local-only validation-tuned, locked-test experiment
  ranking_ablation.py         Semantic/transition/skill-gap diagnostic ablation
tests/                        Deterministic offline unit tests
build_graph.py                CLI: graph, alignment, and transition build modes
career_kg_web.py              Flask server — port 8001
coursera_client.py            Coursera search scraper (no API key needed)
templates/index.html          Chat UI
templates/graph.html          Interactive Cytoscape.js knowledge graph viewer
public/chat.js                Frontend JS — career path visual, explore panels, course cards
public/chat.css               Frontend CSS
public/graph.js               Graph loading, interaction, selection, and recovery states
public/graph.css              Responsive graph workspace styling
test_apis.py                  Smoke-test for Azure embed + Cohere rerank
novelty.md                    Candidate research contributions for thesis paper (pick 1-2)
ALL_STEPS.md                  Chronological history, implementation, commands, and results
```

---

## Graph Stats (enriched 2026-08-15)

| Stat | Value |
|---|---|
| Total nodes | 19,225 |
| Total edges | 224,711 |
| Role nodes | 3,932 (893 ONET + 3,039 ESCO) |
| SIMILAR_TO edges | 132 (66 pairs × 2 directions) |
| TRANSITIONS_TO edges | 18,907 (train only, support ≥5) |
| Transition source roles | 765 |
| Transition incident roles | 824 |
| ChromaDB collection | 4,055 legacy records; 3,932 live graph roles; stale hits filtered at runtime |
| Similarity threshold | 0.65 (cosine, ONET↔ESCO alignment) |
| ONET importance threshold | 3.5 (IM scale) |

**Output paths:**
```
graph/graph.gpickle       NetworkX MultiDiGraph
graph/graph.pre-karrierewege.gpickle  Recoverable pre-enrichment backup
index/chroma/             ChromaDB persistent store
artifacts/karrierewege/   Quality, held-out evaluation, and ranking-ablation outputs
```

---

## API Configuration (.env)

```
CHAT_MODEL_API_KEY      = <set in .env>
CHAT_MODEL_ENDPOINT     = https://career.azure-api.net/career-graph-ai/openai/v1/responses
CHAT_MODEL              = gpt-5.4-nano

EMBED_MODEL_API_KEY     = <set in .env>
EMBED_MODEL_ENDPOINT    = https://career.azure-api.net/career-graph-ai
EMBED_MODEL             = text-embedding-3-large

COHERE_RERANK_API_KEY   = <set in .env>
COHERE_RERANK_ENDPOINT  = https://career.azure-api.net/career-graph-ai/providers/cohere/v2/rerank
COHERE_RERANK_MODEL     = Cohere-rerank-v4.0-pro

SIMILARITY_THRESHOLD    = 0.65
ONET_IMPORTANCE_THRESHOLD = 3.5
TRANSITION_MIN_COUNT    = 5
TRANSITION_CHUNK_SIZE   = 200000
TRANSITION_SMOOTHING_ENABLED = true
TRANSITION_SMOOTHING_NEIGHBOURS = 20
TRANSITION_SMOOTHING_DIRECT_WEIGHT = 0.90
TRANSITION_SMOOTHING_TEMPERATURE = 0.05
KARRIEREWEGE_MAX_INVALID_ROW_RATIO = 0.001
```

**API shape details:**
- Chat: OpenAI **Responses API** — POST to full endpoint, body `{"model":..., "input":[...messages...], "max_output_tokens":...}`, response at `output[*].type=="message" → content[0].text`
- Embed: POST to `EMBED_MODEL_ENDPOINT + "/openai/v1/embeddings"`, body `{"model":..., "input":[...]}`
- Rerank: POST to `COHERE_RERANK_ENDPOINT`, headers `{"api-key":...}`, body `{"model":..., "query":..., "documents":[...], "top_n":N}`, response at `results[*].relevance_score`
- All HTTP via `urllib.request` — no OpenAI SDK

**gpt-5.4-nano is a reasoning model** — it consumes tokens on internal chain-of-thought before outputting text. Token budgets in `inference_pipeline.py`:
- Routing: `max_tokens=2000`
- Generation: `max_tokens=1000` (responses are now 2–3 sentences)

**Available chat models on this Azure gateway** (all support Responses API):
`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4-pro`, `gpt-5.4`, `gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.2`, `gpt-5.1`, `gpt-5-pro`, `gpt-5`, `gpt-chat-latest`
*Avoid `-codex` and `-reasoning` variants.*

---

## 7-Step Inference Pipeline (`src/inference_pipeline.py`)

```
Full conversation history (messages[])
  │
  ▼
1. route_intent()        — Responses API: detect student vs professional,
                           check if enough context; returns follow-up Qs if not.
                           Extract explicit current role, owned skills, and goal.
  │ {"has_context", "user_type", "current_role", "skills", "career_goal"}
  ▼
2. retrieve_candidates() — Embed last query → ChromaDB top-50 role nodes;
  │                        filter stale IDs and add empirical destinations
  ▼
3. rerank_candidates()   — Cohere rerank → top-8 anchors, then order by
  │                        skill accessibility → transition → semantic evidence
  │
  ▼
4. traverse_graph()      — NetworkX: for each anchor collect
                           REQUIRES (ONET IM≥3.5, ESCO essential only)
                           BELONGS_TO, BROADER_THAN, NARROWER_THAN, SIMILAR_TO,
                           and bounded TRANSITIONS_TO evidence
  │
  │  ── If has_context==false: return explore_data (skills + roles panels)
  │     and follow-up questions. Skip steps 5–7.
  ▼
5. generate_response()   — Responses API with user_type-specific system prompt:
                           STUDENT  → 2–3 sentence careers advisor response
                           PROFESSIONAL → 2–3 sentence career coach response
                           Both: bold role/skill names only; no headings/bullets
                           Full history passed so model has conversation context.
  │
  ▼
6. fetch_coursera_courses() — scrape Coursera for anchor role titles +
                              essential skills → return list[dict] (structured, up to 5)
  │
  ▼
7. _build_path_data()    — roles, readiness, have/need skills, transition proof,
                           and SIMILAR_TO pairs → structured frontend data
```

**`run_query()` returns a dict:**
```python
{
  "message": str,
  "courses": list[dict],
  "path":    {"roles": [...], "skills": [...], "similar_pairs": [...]},  # full context
  "explore": {"roles": [...], "skills": [...]},                          # partial context only
  "evidence": {"current_role": {...}, "matched_skills": [...]}           # explicit evidence
}
```
`career_kg_web.py` passes all five fields to the frontend as JSON.

**Partial context flow:** when `has_context==false` (e.g. user only gives degree, no goal),
the pipeline still runs retrieval+traversal and returns:
- `message` — the follow-up questions from the router
- `explore.skills` — top-20 essential skills across matched roles (wrapping chip grid)
- `explore.roles` — all 8 matched role cards (horizontal scrollable track)
This gives the user something visual to explore while they answer the follow-up.

**Conversation memory:** the web layer sends the full `messages[]` array from the client on every request. Both routing and generation receive the full history so the model never re-asks for information already given.

---

## Prompts Summary

### Routing (`_INTENT_SYSTEM_PROMPT`)
- Detects STUDENT / PROFESSIONAL / UNKNOWN from conversation history
- STUDENT needs: degree + skill + goal
- PROFESSIONAL needs: current role + skill + next step
- Returns pure JSON: `{"has_context": bool, "user_type": "...", "followup_questions": "..."}`

### Generation (2–3 sentences — no markdown headings or bullets)
- `_GENERATION_SYSTEM_PROMPT_STUDENT` — 1 short paragraph; names best-fit role in bold; one strength + one skill to develop
- `_GENERATION_SYSTEM_PROMPT_PROFESSIONAL` — 1 short paragraph; names best-fit next role in bold; one strength + one skill gap
- **Strict graph-only constraint:** every role/skill cited must appear in the graph triples

---

## Frontend

### Chat response layout (per assistant turn)
1. **Text bubble** — 2–3 sentence advice from the LLM
2. **Career Path visual** (full context only) — horizontal scrollable track of role cards with essential skill chips and arrow connectors between roles
3. **Explore panels** (partial context only) — two stacked panels: skills chip grid + roles scrollable track
4. **Coursera courses** — up to 5 cards (title, provider, chips, description)

### Career Path visual (`addCareerPath` in `chat.js`)
- Each role card: ONET/ESCO source badge (blue/green), title, prep level, essential skill chips
- Arrow connectors between cards; horizontal scroll with thin scrollbar
- Skill chips wrap (`word-break: break-word`); titles truncated at 38 chars on the backend

### Explore panels (`addExplorePanels` in `chat.js`)
- Skills panel: wrapping grid of `explore-skill-chip` elements (up to 20 skills, sorted by frequency across roles)
- Roles panel: horizontal scrollable `explore-role-card` track with description snippet

### Design
- **Palette:** `#070A10` deep space, `#0E141E` graphite, `#151E2B` slate, `#EAF1F8` frost, `#4CC2EA` cyan
- **Font:** Outfit + JetBrains Mono
- Calm scientific-instrument direction with one restrained atmospheric wash, evidence-first hierarchy, and optional GSAP entry motion
- Responsive two-column desktop shell, compact tablet layout, and single-column mobile flow with 44–48px controls
- Sanitized assistant Markdown, safe external links/course titles, explicit loading/error states, and reduced-motion/transparency support
- One active chat request at a time; New Chat aborts and invalidates stale responses

### Knowledge Graph Viewer (`/graph`)
- Responsive Cytoscape.js workspace with a scroll-safe header and legend
- `/api/graph-data` samples top-60 ONET + top-60 ESCO roles by degree + up to 160 skill nodes
- Color coding: blue = ONET, green = ESCO, amber = skills; dashed amber edges = TRANSITIONS_TO
- Click/tap node → highlight neighbourhood and open a persistent inspector; pointer hover → clamped summary tooltip
- Skill and isolated-node labels are progressively disclosed to reduce visual noise
- Non-2xx, empty-data, and Cytoscape CDN failures produce a retryable error state

---

## Node / Edge Schema

**Node types:** `role` (ONET+ESCO), `element` (ONET), `skill` (ESCO), `skill_group` (ESCO), `isco_group` (ESCO)

**Edge types:**
| relation | meaning | key attrs |
|---|---|---|
| REQUIRES | role → skill/element | `requirement_level` (float ≥3.5 for ONET, `'essential'` only for ESCO), `domain`, `source` |
| BELONGS_TO | ESCO role → isco_group | `source='esco'` — URI-resolved (fixed) |
| BROADER_THAN | concept → broader concept | `pillar`, `source` |
| NARROWER_THAN | concept → narrower concept | `pillar`, `source` |
| SIMILAR_TO | ONET role ↔ ESCO role | `similarity` float ≥0.65, `source='alignment'` |
| TRANSITIONS_TO | ESCO role → empirically observed next ESCO role | `count`, `probability`, `source_total`, `split='train'`, `source='karrierewege'` |

**Removed edge types (KG quality cleanup):**
- `RELATED_TO` — generic skill-skill taxonomy links; excluded at build time
- ESCO `optional` REQUIRES edges — excluded at build time and inference time
- Work Values / Work Styles ONET element nodes — excluded at build time

---

## KG Quality Improvements (2026-07-27)

| # | File | What changed |
|---|---|---|
| 1 | `onet_preprocessing.py` | Excluded Work Values + Work Styles element nodes (`_EXCLUDED_DOMAINS`) |
| 2 | `esco_preprocessing.py` | Fixed BELONGS_TO edges: `iscoGroup` code → URI via ISCOGroups_en.csv lookup |
| 3 | `esco_preprocessing.py` | `_build_skill_skill_edges` drops RELATED_TO; keeps only BROADER_THAN/NARROWER_THAN |
| 4 | `graph_build.py` | New `prune_graph()`: removes roles with <3 REQUIRES edges (123 removed) + isolates (2,906 removed) |
| 5 | `.env` | `SIMILARITY_THRESHOLD` raised 0.45 → 0.65 (SIMILAR_TO pairs: 12,631 → 66) |
| 6 | `.env` | `ONET_IMPORTANCE_THRESHOLD` added at 3.5 (was 3.0 hardcoded) |
| 7 | `inference_pipeline.py` | ESCO REQUIRES: `essential` only (dropped `essential/optional` and `optional`) |

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
python build_graph.py --embed     # rebuild graph + ChromaDB (calls Azure embed API ~40 batches)
python build_graph.py --align-only  # recompute SIMILAR_TO edges only (calls Azure embed API)
python build_graph.py --transitions-only  # replace train-derived transition edges only
python evaluation/evaluate_karrierewege.py  # validation/test metrics
python evaluation/ranking_ablation.py       # fixed-query ranking diagnostic
```

---

## Verification Checklist

```bash
# Graph sanity
python -c "
import pickle; G=pickle.load(open('graph/graph.gpickle','rb'))
print(G.number_of_nodes(), G.number_of_edges())
bad=[e for _,_,e in G.edges(data=True) if e.get('relation')=='REQUIRES' and e.get('source')=='onet' and str(e.get('requirement_level',99)).replace('.','',1).isdigit() and float(e.get('requirement_level',99))<3.5]
print('Bad ONET edges (should be 0):', len(bad))
optional=[e for _,_,e in G.edges(data=True) if e.get('relation')=='REQUIRES' and e.get('source')=='esco' and e.get('requirement_level')!='essential']
print('ESCO non-essential edges (should be 0):', len(optional))
"
# Expected: 19225 224711 / Bad ONET edges: 0 / ESCO non-essential: 0

# API response shape check
curl -s -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I finished a BSc in Computer Science, I know Python and SQL, I want to go into data science."}]}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('status:', d['status']); print('courses:', len(d.get('courses',[])));  print('path roles:', len(d.get('path',{}).get('roles',[])));  print('msg[:200]:', d['message'][:200])"
# Expected: status: ok, courses: 3-5, path roles: 3-8, message is 2-3 sentences

# Partial context check (should return explore panels, not full path)
curl -s -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I studied Computer Science"}]}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('has_path:', bool(d.get('path',{}).get('roles'))); print('explore_roles:', len(d.get('explore',{}).get('roles',[]))); print('explore_skills:', len(d.get('explore',{}).get('skills',[])))"
# Expected: has_path: False, explore_roles: 8, explore_skills: up to 20
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

- [ ] Coursera client scrapes HTML (no API key) — may break if Coursera changes their page structure
- [ ] Knowledge graph viewer uses `cose` layout which is slow for >300 nodes — consider pre-computing layout positions and caching as JSON
- [ ] SIMILAR_TO edges are now only 66 pairs (threshold 0.65) — may be too few for cross-framework traversal; monitor recommendation quality and lower threshold to 0.60 if needed
- [ ] ONET roles have NO ISCO codes in the graph (0/893). ESCO roles all have them (3039/3039). Need SOC→ISCO-08 crosswalk from BLS for contribution #3.
- [ ] ChromaDB contains 4,055 records while the pruned graph has 3,932 roles. Runtime filtering prevents invalid candidates; rebuild the role index later to remove the 123 stale records.
- [x] `.agents/`, `.claude/`, `CODE/`, `static/`, graph binaries, and the local vector index are retained locally but ignored on GitHub. The deployed runtime should connect to external graph/vector storage.

---

## Additional Datasets (in `data/`)

| Dataset | Location | Rows | Format | Status |
|---|---|---|---|---|
| **Karrierewege** | `data/Karrierewege/` (train/test/validation splits) | 2,480,369 career steps | CSV: _id, experience_order, preferredLabel_en/de, description_en/de, skills | **IMPLEMENTED** — 18,907 train-only empirical transition edges |
| LinkedIn | `data/Linkedin/postings.csv` + subdirs | 3,383,601 postings | CSV: job_id, title, description, skills_desc, salary, company | Skip — skills too coarse (36 categories) |
| Monster.com | `data/monster_com-job_sample.csv` | ~22,000 postings | CSV: job_title, job_description, sector | Skip — small, noisy |
| Job descriptions | `data/job-descriptions/training_data.csv` | ~32,000 | CSV: company, description, title, length | Skip — NLP training only |

### Karrierewege Dataset Details
- **Origin:** "capadict" project; German career transition data with ESCO labels
- **Structure:** Each row = one career step for one person. Grouped by `_id`, ordered by `experience_order`
- **568,888 person-disjoint trajectories** spanning 1,295 ESCO occupations
- **Exact ESCO graph linkage:** 1,295/1,295 normalized English occupation labels (100% row coverage)
- **Prior mapping:** `artifacts/karrierewege_onet_mapping_summary.json` — 77.1% of ESCO titles matched to ONET SOC codes (999/1295 titles, covering 91.6% of data rows)
- **Skills column:** contains Python list of ESCO skill strings per career step
- **Train/test/val split:** ready for quantitative evaluation

---

## Example Prompts That Work

**Student (full context → path visual):**
> "I just graduated with a BSc in Computer Science, I know Python, SQL, and basic machine learning. I want to go into data science or AI roles."

**Professional (full context → path visual):**
> "I am a data analyst with 4 years experience using Python, SQL, and Tableau. I want to switch to a data science or machine learning role."

**Partial context → explore panels + follow-up questions:**
> "I studied Computer Science"
> "I am a software engineer"

**Pattern for full context:** `"I am a [role/degree] with [experience/year], I know [skills], and I want to [goal]."`

---

## Session Changes (2026-07-27)

1. **Shorter LLM responses** — generation prompts cut to 2–3 sentences (1 paragraph); `max_tokens` dropped 4000 → 1000; supervisor feedback: less text, more visual
2. **Linear career path visual** — `_build_path_data()` in `inference_pipeline.py`; `addCareerPath()` + CSS in `chat.js`/`chat.css`; horizontal scrollable role cards with skill chips + arrow connectors
3. **Explore panels for partial context** — `_build_explore_data()` in `inference_pipeline.py`; `addExplorePanels()` + CSS; when user only gives degree/role with no goal, shows skills grid + roles track alongside follow-up questions
4. **More recommendations** — `rerank_top_n` 5 → 8 in `config.py`; Coursera limit 3 → 5
5. **KG quality — 7 improvements** — see KG Quality Improvements table above; graph rebuilt and re-embedded; node count reduced from 22,259 → 19,225; SIMILAR_TO pairs from 12,631 → 66
6. **Skill chip overflow fix** — `white-space: normal; word-break: break-word` on `.path-skill-chip`; skill titles truncated at 38 chars on backend; scroll track `padding-right` to prevent last card clipping

---

## Session Changes (2026-08-01)

1. **Novelty contributions brainstormed** — 6 candidate research contributions identified and documented in `novelty.md`
2. **Dataset assessment** — evaluated all datasets in `data/` folder. Decision: use Karrierewege (2.48M ESCO-native career steps), skip LinkedIn/Monster/job-descriptions
3. **Top novelty: Empirical Transition Edges** — integrate Karrierewege as `TRANSITIONS_TO` edges (new edge type) weighted by frequency. Strongest contribution because grounded in real data.
4. **Key finding: ONET has no ISCO codes** — verified that 0/893 ONET role nodes have `isco_group` attribute. ESCO has all 3039. Contribution #3 (ISCO alignment validation) requires a SOC→ISCO crosswalk from BLS.
5. **Supervisor guidance:** wants 4-5 novelty ideas to pick from; evaluation comes separately later; KG improvements may continue later
6. **Next research option:** Search for additional structured skills, transitions, or demand datasets that map to ESCO/ONET.

---

## Session Changes (2026-08-15)

1. **Leakage-safe Karrierewege integration** — training builds the graph; validation/test are evaluation-only.
2. **Streaming cleaning** — handles chunk boundaries, exact/conflicting duplicates, missing orders, gaps, and self-transitions without loading description fields.
3. **Typed multigraph** — migrated to `MultiDiGraph`, preserving all 130 taxonomy/transition pair collisions.
4. **Graph enrichment** — added 18,907 support-≥5 transition edges while keeping 19,225 nodes and 132 alignment edges.
5. **Skill-gap ranking** — added deterministic possession checks, ESCO requirements, ONET alignment fallback, missing-evidence states, and stable ranking.
6. **Evidence UI** — role cards now show readiness, observed moves, owned skills, and development gaps.
7. **Evaluation** — validation Hits@5 0.3716/MRR 0.2586; test Hits@5 0.3702/MRR 0.2591.
8. **Verification** — 20 offline tests pass; full and partial-context live smoke tests return HTTP 200.

### Production UI/frontend pass

1. **Cohesive visual system** — unified the studio and graph viewer around the deep-space/graphite/slate palette, Outfit UI typography, JetBrains Mono evidence labels, consistent spacing, borders, and controls.
2. **Responsive application shell** — added bounded desktop chat, compact tablet presentation, narrow-screen layout, safe-area padding, scroll-snap evidence tracks, and touch-sized controls.
3. **Safe rendering** — assistant Markdown now passes through a strict allowlist sanitizer; course titles and API error messages use DOM text nodes; CDN failures degrade to usable text or explicit recovery states.
4. **Request integrity** — duplicate submissions are locked, New Chat aborts active work, and stale in-flight responses cannot enter a cleared conversation.
5. **Graph usability** — added responsive navigation/legend, progressive labels, persistent tap/click inspection, viewport-safe hover details, accessible controls, and retryable load failures.
6. **Validation** — JavaScript syntax, Python compilation, 20 offline unit tests, four local HTTP routes, desktop/tablet/mobile render checks, and blocked-CDN fallback renders passed. `/api/chat` was not invoked during the visual pass to avoid unnecessary external model calls.
7. **Deployment boundary** — `HANDOFF.md` and `ALL_STEPS.md` are versioned project records but excluded from Vercel uploads; graph and Chroma assets remain local-only pending external database integration.

### Embedding-smoothed transition pass

1. **Why embeddings are used** — title mapping was already complete, so raw Karrierewege rows were not re-embedded. Existing `text-embedding-3-large` ESCO role vectors identify semantically related transition-source roles.
2. **Leakage boundary** — only graph edges marked `split='train'` contribute destination distributions; validation selects parameters and the frozen configuration is applied to test once.
3. **Selected configuration** — 20 neighbours, direct-transition weight 0.90, softmax temperature 0.05.
4. **Locked test improvement** — MRR 0.259056 → 0.261703, Hits@5 0.370239 → 0.372183, Hits@10 0.501359 → 0.505809, and destination coverage 0.871500 → 0.953196.
5. **Runtime behavior** — direct observations retain counts/probabilities. Smoothed-only roles are marked `semantic_transition_backoff`, shown as inferred evidence, and never represented as directly observed moves.
6. **Failure behavior** — disabled smoothing, missing vectors, incompatible Chroma data, or scorer errors fall back to the original direct transition expansion.
7. **Vector scope** — 3,039/3,039 live ESCO vectors loaded in evaluation; runtime preloads only 765 transition-source vectors and caches per-role rankings. No embedding API call or raw person-level embedding is used.
8. **Verification** — 28 offline tests pass; a mocked `/api/chat` request initialized the real-vector smoother, returned a ranked result, and exposed no smoothing error without calling external models.

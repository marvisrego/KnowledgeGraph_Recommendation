# GraphRAG Career Advisor

A Flask-based intelligent career guidance chatbot powered by a knowledge graph combining ONET and ESCO career taxonomies. The system uses GraphRAG pattern with embeddings, reranking, and graph traversal to provide personalized career recommendations and course suggestions.

## Overview

**Status:** Fully built and running  
**Server:** `python career_kg_web.py` → `http://127.0.0.1:8001`  
**Remote:** `https://github.com/marvisrego/KnowledgeGraph_Recommendation`

The system is designed for thesis work and provides career advice to both students (degree + skills + goals) and professionals (current role + skills + next steps). It integrates with Coursera for course recommendations and includes an interactive knowledge graph viewer.

## Quick Start

### Prerequisites
- Python 3.8+
- Azure API credentials (chat, embeddings, rerank)
- Coursera access (scraping, no API key needed)

### Installation & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (.env file with Azure API keys)
# See API Configuration section below

# Start the server
python career_kg_web.py
# → http://127.0.0.1:8001
```

## Project Structure

```
config.py                     Settings dataclass — reads from .env
src/
  onet_preprocessing.py       ONET Excel → nodes/edges DataFrames
  esco_preprocessing.py       ESCO CSV → nodes/edges DataFrames
  graph_build.py              Merge into nx.DiGraph; pickle save/load
  embeddings_index.py         Azure embed → ChromaDB; cosine alignment
  inference_pipeline.py       6-step inference pipeline
build_graph.py                CLI for graph building (with optional embed/align)
career_kg_web.py              Flask server — port 8001
coursera_client.py            Coursera search scraper
templates/
  index.html                  Chat UI (dark, clean design)
  graph.html                  Interactive Cytoscape.js knowledge graph viewer
public/
  chat.js                     Frontend chat logic — marked.js markdown rendering
  chat.css                    Frontend styling
CODE/                         Clean deployable copy of all source files
test_apis.py                  Smoke-test for Azure APIs
```

## Graph Statistics

| Metric | Value |
|--------|-------|
| Total nodes | 22,259 |
| Total edges | 235,501 |
| Role nodes | 4,055 (1,016 ONET + 3,039 ESCO) |
| SIMILAR_TO edges | 25,262 (aligned ONET↔ESCO pairs) |
| ChromaDB embeddings | 4,055 role embeddings |
| Similarity threshold | 0.45 (cosine) |

**Pre-built artifacts (do not rebuild unless data changes):**
```
graph/graph.gpickle       NetworkX DiGraph
index/chroma/             ChromaDB persistent store
```

## API Configuration

Create a `.env` file with your Azure API credentials:

```env
CHAT_MODEL_API_KEY      = <your-api-key>
CHAT_MODEL_ENDPOINT     = https://career.azure-api.net/career-graph-ai/openai/v1/responses
CHAT_MODEL              = gpt-5.4-nano

EMBED_MODEL_API_KEY     = <your-api-key>
EMBED_MODEL_ENDPOINT    = https://career.azure-api.net/career-graph-ai
EMBED_MODEL             = text-embedding-3-large

COHERE_RERANK_API_KEY   = <your-api-key>
COHERE_RERANK_ENDPOINT  = https://career.azure-api.net/career-graph-ai/providers/cohere/v2/rerank
COHERE_RERANK_MODEL     = Cohere-rerank-v4.0-pro

SIMILARITY_THRESHOLD    = 0.45
```

### API Details
- **Chat:** OpenAI Responses API — Responses are reasoning models consuming tokens on internal chain-of-thought
- **Embeddings:** Azure text-embedding-3-large
- **Rerank:** Cohere rerank-v4.0-pro
- **HTTP:** All calls via `urllib.request` (no SDKs)

## 6-Step Inference Pipeline

The system processes queries through a structured pipeline:

```
1. route_intent()         → Detect user type (student/professional); validate context
2. retrieve_candidates()  → Embed query; fetch top-50 role candidates from ChromaDB
3. rerank_candidates()    → Cohere rerank → top-5 anchors
4. traverse_graph()       → NetworkX: collect REQUIRES, BELONGS_TO, BROADER_THAN, 
                            NARROWER_THAN, SIMILAR_TO edges for each anchor
5. generate_response()    → LLM generation with user-type-specific system prompt
6. fetch_coursera_courses() → Scrape Coursera for course recommendations
```

**Output:** `{"message": str, "courses": list[dict]}`

## Frontend

### Design
- **Palette:** Dark slate (#0E1117 base, #161B24 panels, #1C2333 raised), blue accent #4E8EE8
- **Typography:** Inter (clean, editorial feel)
- **Interaction:** Markdown rendering with marked.js; course cards with metadata

### Features
- **Chat interface:** Full conversation history passed to both routing and generation steps
- **Markdown rendering:** Assistant messages render as HTML; user messages plain text
- **Course recommendations:** Structured cards with title, provider, metadata, description
- **Knowledge graph viewer:** Interactive Cytoscape.js visualization at /graph
  - Samples top-60 ONET + top-60 ESCO roles by degree
  - ~160 essential skill nodes
  - Color coding: blue (ONET), green (ESCO), amber (skills)
  - Click nodes to highlight neighbourhoods; zoom/fit controls

## Example Prompts

### Student
> "I just graduated with a BSc in Computer Science, I know Python, SQL, and basic machine learning. I want to go into data science or AI roles."

### Professional
> "I am a data analyst with 4 years experience using Python, SQL, and Tableau. I want to switch to a data science or machine learning role."

### Pattern
`"I am a [role/degree] with [experience/years], I know [skills], and I want to [goal]."`

## Verification & Testing

### Graph Sanity Check
```bash
python -c "
import pickle
G = pickle.load(open('graph/graph.gpickle','rb'))
print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
"
```

### API Health Check
```bash
curl http://127.0.0.1:8001/api/status
```

### Smoke Test
```bash
python test_apis.py
```

## Building the Graph

### Full rebuild (with embeddings)
```bash
python build_graph.py --embed
```

### Graph only (no API calls)
```bash
python build_graph.py
```

### Alignment only (recompute SIMILAR_TO edges)
```bash
python build_graph.py --align-only
```

> **Note:** Do not rebuild unless data changes. Pre-built graph and ChromaDB are already on disk.

## Node & Edge Schema

**Node Types:**
- `role` — ONET and ESCO career roles
- `element` — ONET job elements
- `skill` — ESCO skills
- `skill_group`, `isco_group` — ESCO classifications

**Edge Relations:**
| Type | Meaning | Attributes |
|------|---------|------------|
| REQUIRES | role → skill/element | requirement_level, domain, source |
| BELONGS_TO | ESCO role → isco_group | source='esco' |
| BROADER_THAN | concept → broader concept | pillar, source |
| NARROWER_THAN | concept → narrower concept | pillar, source |
| RELATED_TO | skill → skill | source='esco' |
| SIMILAR_TO | ONET role ↔ ESCO role | similarity, source='alignment' |

## Known Issues & TODO

- [ ] vercel.json still references old build_vercel_bundle.py
- [ ] static/ directory mirrors public/ — redundant, can be deleted
- [ ] Coursera client scrapes HTML (no API key) — may break if Coursera changes page structure
- [ ] career_kg_chat.py is gutted — delete if not needed
- [ ] Knowledge graph viewer uses cose layout (slow for >300 nodes) — consider pre-computing layout positions

## License

This is thesis work. See repository for licensing details.

## Contact

**Author:** marvisrego  
**Repository:** https://github.com/marvisrego/KnowledgeGraph_Recommendation

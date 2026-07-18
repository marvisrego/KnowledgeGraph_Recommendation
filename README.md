# GraphRAG Career Advisor

An intelligent career guidance chatbot powered by a knowledge graph combining ONET and ESCO career taxonomies. Uses GraphRAG pattern with embeddings, reranking, and graph traversal to provide personalized career recommendations.

## What It Does

- **Career Advising:** Provides personalized career recommendations for students and professionals
- **Skill Matching:** Identifies relevant roles based on user skills and background
- **Course Recommendations:** Integrates Coursera to suggest courses for skill development
- **Knowledge Graph:** Interactive visualization of career pathways and skill relationships

## Architecture

### Core Components
- **Graph Database:** NetworkX DiGraph with 22,259 nodes and 235,501 edges
- **Embedding Index:** Azure text-embedding-3-large + ChromaDB for semantic search
- **Reranker:** Cohere rerank-v4.0-pro for ranking candidate roles
- **LLM:** Azure OpenAI Responses API (gpt-5.4-nano) for conversational responses

### 6-Step Inference Pipeline
1. **route_intent()** — Detect user type (student/professional); validate context
2. **retrieve_candidates()** — Embed query; fetch top-50 role candidates from ChromaDB
3. **rerank_candidates()** — Cohere rerank → top-5 anchors
4. **traverse_graph()** — Collect related skills, roles, and prerequisites
5. **generate_response()** — LLM generation with context-aware system prompt
6. **fetch_coursera_courses()** — Scrape Coursera for course recommendations

### Data Schema
- **Nodes:** ONET roles (1,016), ESCO roles (3,039), skills, job elements
- **Edges:** REQUIRES, SIMILAR_TO, BELONGS_TO, BROADER_THAN, NARROWER_THAN, RELATED_TO

## Quick Start

```bash
# Install and run
pip install -r requirements.txt
python career_kg_web.py
# → http://127.0.0.1:8001
```

## Configuration

Create `.env` with Azure API credentials:
```env
CHAT_MODEL_API_KEY = <key>
EMBED_MODEL_API_KEY = <key>
COHERE_RERANK_API_KEY = <key>
```

See HANDOFF.md for full configuration details.

## Project Structure

```
config.py                  Settings and environment loading
src/
  onet_preprocessing.py    ONET Excel processing
  esco_preprocessing.py    ESCO CSV processing
  graph_build.py           NetworkX graph construction
  embeddings_index.py      Azure embeddings + ChromaDB
  inference_pipeline.py    6-step query pipeline
career_kg_web.py           Flask server
coursera_client.py         Course recommendations scraper
templates/
  index.html               Chat UI (dark design, Inter font)
  graph.html               Interactive Cytoscape.js viewer
public/
  chat.js / chat.css       Frontend logic and styling
```

## Example Prompts

**Student:** "I just graduated with a BSc in Computer Science, I know Python and SQL, I want to go into data science."

**Professional:** "I am a data analyst with 4 years experience using Python and SQL. I want to switch to machine learning."

## Verification

```bash
# Check graph integrity
python -c "import pickle; G=pickle.load(open('graph/graph.gpickle','rb')); print(G.number_of_nodes(), G.number_of_edges())"

# Test API health
curl http://127.0.0.1:8001/api/status
```

## Status

Fully built and running. Graph and embeddings are pre-computed. No rebuild needed unless source data changes.

## Author

marvisrego | https://github.com/marvisrego/KnowledgeGraph_Recommendation

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
- **Embedding Index:** Text embeddings + vector database for semantic search
- **Reranking:** Multi-stage reranking for candidate role selection
- **LLM:** Conversational AI for context-aware career guidance

### 6-Step Inference Pipeline
1. **route_intent()** — Detect user type (student/professional); validate context
2. **retrieve_candidates()** — Embed query; fetch top-50 role candidates
3. **rerank_candidates()** — Rank candidates by relevance
4. **traverse_graph()** — Collect related skills, roles, and prerequisites
5. **generate_response()** — LLM generation with context-aware system prompt
6. **fetch_coursera_courses()** — Recommend courses for skill development

### Data Schema
- **Nodes:** ONET roles (1,016), ESCO roles (3,039), skills, job elements
- **Edges:** REQUIRES, SIMILAR_TO, BELONGS_TO, BROADER_THAN, NARROWER_THAN, RELATED_TO

## Quick Start

```bash
pip install -r requirements.txt
python career_kg_web.py
# → http://127.0.0.1:8001
```

## Configuration

Create `.env` file with API credentials for:
- LLM provider (chat API)
- Embeddings service
- Reranking service

## Project Structure

```
config.py                  Settings and environment loading
src/
  onet_preprocessing.py    ONET data processing
  esco_preprocessing.py    ESCO data processing
  graph_build.py           Graph construction
  embeddings_index.py      Embeddings and indexing
  inference_pipeline.py    Query processing pipeline
career_kg_web.py           Flask web server
coursera_client.py         Course recommendations
templates/
  index.html               Chat interface
  graph.html               Knowledge graph viewer
public/
  chat.js / chat.css       Frontend components
```

## Example Prompts

**Student:** "I graduated with a BSc in Computer Science, I know Python and SQL, I want to go into data science."

**Professional:** "I am a data analyst with 4 years experience. I want to switch to a machine learning role."

## Verification

```bash
python -c "import pickle; G=pickle.load(open('graph/graph.gpickle','rb')); print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')"
curl http://127.0.0.1:8001/api/status
```

## Status

Fully functional. Pre-computed graph and embeddings included.

## Author

marvisrego

# GraphRAG Career Advisor

An intelligent career guidance chatbot powered by a knowledge graph combining ONET and ESCO career taxonomies. Uses GraphRAG pattern with embeddings, reranking, and graph traversal to provide personalized career recommendations.

## What It Does

- **Career Advising:** Personalized career recommendations for students and professionals
- **Skill Matching:** Identifies relevant roles based on skills and background
- **Course Recommendations:** Suggests relevant courses for skill development
- **Knowledge Graph:** Interactive visualization of career pathways and skill relationships

## Architecture

### Core Components
- **Graph Database:** NetworkX DiGraph (22,259 nodes, 235,501 edges)
- **Embedding Index:** Semantic search via embeddings and vector database
- **Reranking:** Multi-stage candidate ranking for career roles
- **LLM:** Conversational AI for context-aware guidance

### 6-Step Pipeline
1. **route_intent()** — Detect user type; validate context
2. **retrieve_candidates()** — Semantic search for relevant roles
3. **rerank_candidates()** — Rank candidates by relevance
4. **traverse_graph()** — Collect related skills and prerequisites
5. **generate_response()** — Generate personalized advice
6. **fetch_coursera_courses()** — Recommend courses

### Knowledge Graph
- **Nodes:** ONET roles (1,016), ESCO roles (3,039), skills, elements
- **Edges:** REQUIRES, SIMILAR_TO, BELONGS_TO, BROADER_THAN, NARROWER_THAN, RELATED_TO

## Quick Start

Set up environment variables and run the Flask server.

## Project Structure

```
config.py                  Settings
src/
  onet_preprocessing.py    ONET data processing
  esco_preprocessing.py    ESCO data processing
  graph_build.py           Graph construction
  embeddings_index.py      Embeddings and indexing
  inference_pipeline.py    Query pipeline
career_kg_web.py           Flask server
coursera_client.py         Course recommendations
templates/
  index.html               Chat interface
  graph.html               Graph viewer
public/
  chat.js / chat.css       Frontend
```

## Status

Fully functional with pre-computed graph and embeddings.

## Author

marvisrego

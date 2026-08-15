# GraphRAG Career Advisor

A Flask career-guidance application backed by an ONET, ESCO, and Karrierewege knowledge graph. It combines semantic retrieval, reranking, graph traversal, skill-gap analysis, observed career-transition evidence, and course recommendations.

## Run locally

1. Install Python 3.13 and the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add the three API keys.

3. Start the application:

   ```powershell
   python career_kg_web.py
   ```

4. Open `http://127.0.0.1:8001`.

Local development reads the graph from `graph/graph.gpickle` and the Chroma index from `index/chroma/`. These data files, raw datasets, research notes, and experimental outputs stay on the local machine and are intentionally excluded from Git.

## Deploy to Vercel

Connect this repository and deploy the `dev` branch. Vercel detects `app.py` as the Flask entry point. Configure these environment variables in the Vercel project:

- `CHAT_MODEL_API_KEY`
- `EMBED_MODEL_API_KEY`
- `COHERE_RERANK_API_KEY`

Common endpoint, model, and ranking settings can be overridden with the optional variables documented in `.env.example`.

The deployment branch does not ship a graph or vector index. Before production deployment, connect `load_graph()` and `load_chroma_collection()` to the external graph and vector databases you choose, then configure that provider's connection variables in Vercel. The current loaders remain file-based for local development.

## Runtime structure

```text
app.py                  Vercel Flask entry point
career_kg_web.py        Routes and application factory
config.py               Environment-based settings
coursera_client.py      Course lookup integration
src/                     Retrieval, graph, and skill-gap pipeline
templates/               Chat and graph pages
public/                  Browser JavaScript and CSS
external data stores     Graph and vector data (not committed)
```

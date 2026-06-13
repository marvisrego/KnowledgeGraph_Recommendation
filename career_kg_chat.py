from __future__ import annotations

import argparse
import csv
import difflib
import importlib
import json
import math
import os
import pickle
import re
import sys
import textwrap
import urllib.error
import urllib.request

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TRIPLE_EXTRACTION_SYSTEM_PROMPT = textwrap.dedent(
    """
    You extract knowledge graph triples from career-domain documents.

    Return ONLY valid JSON.
    The JSON must be an object with one key named "triples".
    The value of "triples" must be a list of objects.
    Each object must have exactly these keys:
    - "node_1"
    - "edge"
    - "node_2"

    Allowed edge labels include:
    - REQUIRES
    - TRANSITIONS_TO
    - HAS_SKILL
    - REQUIRES_EDUCATION
    - RELATED_TO

    Do not add prose, markdown, comments, or explanations.
    """
).strip()


QUERY_ROUTING_SYSTEM_PROMPT = textwrap.dedent(
    """
    You normalize a career-assistance query into retrieval fields.

    Return ONLY valid JSON with these keys:
    - question_type: one of ["role_recommendation", "skill_gap", "role_information", "general"]
    - current_role: string or null
    - target_role: string or null
    - current_skills: array of strings
    - preferred_industry: string or null
    - location: string or null
    - experience_level: string or null
    - is_vague: boolean
    - missing_fields: array of strings

    Set is_vague=true when the request does not contain enough information for grounded retrieval.
    """
).strip()


GROUNDED_RESPONSE_SYSTEM_PROMPT = textwrap.dedent(
    """
    You answer career questions using ONLY the supplied retrieval context.

    Rules:
    - Do not use outside knowledge.
    - Do not hallucinate missing skills, education, salaries, or transitions.
    - If the context is insufficient, say that clearly.
    - Cite the available evidence in plain language.
    - Keep the answer practical and concise.
    """
).strip()


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class MissingDependencyError(RuntimeError):
    pass


@dataclass
class Settings:
    root_dir: Path
    onet_dir: Path
    postings_path: Path
    jobs_dir: Path
    companies_dir: Path
    mappings_dir: Path
    artifacts_dir: Path
    graph_cache_path: Path
    nvidia_api_key: str | None
    nvidia_chat_api_key: str | None
    nvidia_embed_api_key: str | None
    nvidia_rerank_api_key: str | None
    nvidia_chat_model: str
    nvidia_embed_model: str
    nvidia_rerank_model: str
    use_nvidia_embeddings: bool
    use_reranker: bool
    postings_chunk_size: int
    postings_limit: int | None
    graph_top_k: int
    vector_top_k: int
    title_match_cutoff: float

    @property
    def resolved_chat_api_key(self) -> str | None:
        return self.nvidia_chat_api_key or self.nvidia_api_key

    @property
    def resolved_embed_api_key(self) -> str | None:
        return self.nvidia_embed_api_key or self.nvidia_api_key

    @property
    def resolved_rerank_api_key(self) -> str | None:
        return self.nvidia_rerank_api_key or self.nvidia_api_key

    @classmethod
    def from_env(cls, root_dir: Path) -> "Settings":
        env_path = root_dir / ".env"
        env_values = load_env_file(env_path)
        merged = {**env_values, **os.environ}

        def get_path(name: str, defaults: list[str]) -> Path:
            configured = merged.get(name)
            if configured:
                path = Path(configured)
                return path if path.is_absolute() else root_dir / path
            for default in defaults:
                candidate = root_dir / default
                if candidate.exists():
                    return candidate
            return root_dir / defaults[0]

        return cls(
            root_dir=root_dir,
            onet_dir=get_path("ONET_DIR", ["ONET_data/ONET", "ONET_data/db_30_3_excel"]),
            postings_path=get_path("POSTINGS_PATH", ["ONET_data/Linkedin/postings.csv", "postings.csv"]),
            jobs_dir=get_path("JOBS_DIR", ["ONET_data/Linkedin/jobs", "jobs"]),
            companies_dir=get_path("COMPANIES_DIR", ["ONET_data/Linkedin/companies", "companies"]),
            mappings_dir=get_path("MAPPINGS_DIR", ["ONET_data/Linkedin/mappings", "mappings"]),
            artifacts_dir=get_path("ARTIFACTS_DIR", ["artifacts"]),
            graph_cache_path=get_path("GRAPH_CACHE_PATH", ["artifacts/career_kg_bundle.pkl"]),
            nvidia_api_key=blank_to_none(merged.get("NVIDIA_API_KEY")),
            nvidia_chat_api_key=blank_to_none(merged.get("NVIDIA_CHAT_API_KEY")),
            nvidia_embed_api_key=blank_to_none(merged.get("NVIDIA_EMBED_API_KEY")),
            nvidia_rerank_api_key=blank_to_none(merged.get("NVIDIA_RERANK_API_KEY")),
            nvidia_chat_model=merged.get("NVIDIA_CHAT_MODEL", "nemotron-3-ultra-550b-a55b"),
            nvidia_embed_model=merged.get("NVIDIA_EMBED_MODEL", "nv-embed-v1"),
            nvidia_rerank_model=merged.get("NVIDIA_RERANK_MODEL", "rerank-qa-mistral-4b"),
            use_nvidia_embeddings=as_bool(merged.get("USE_NVIDIA_EMBEDDINGS"), default=False),
            use_reranker=as_bool(merged.get("USE_RERANKER"), default=False),
            postings_chunk_size=as_int(merged.get("POSTINGS_CHUNK_SIZE"), default=5000),
            postings_limit=as_optional_int(merged.get("POSTINGS_LIMIT", "500")),
            graph_top_k=as_int(merged.get("GRAPH_TOP_K"), default=8),
            vector_top_k=as_int(merged.get("VECTOR_TOP_K"), default=5),
            title_match_cutoff=as_float(merged.get("TITLE_MATCH_CUTOFF"), default=0.82),
        )


@dataclass
class VectorDocument:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    document: VectorDocument
    score: float


@dataclass
class QueryState:
    original_query: str
    question_type: str = "general"
    current_role: str | None = None
    target_role: str | None = None
    current_skills: list[str] = field(default_factory=list)
    preferred_industry: str | None = None
    location: str | None = None
    experience_level: str | None = None
    is_vague: bool = False
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    role_code: str | None
    role_title: str | None
    graph_facts: list[str]
    vector_results: list[SearchResult]
    missing_skills: list[str]
    recommended_roles: list[dict[str, Any]]
    context_text: str


@dataclass
class GraphBundle:
    graph: Any
    role_lookup: dict[str, str]
    role_titles: dict[str, str]
    vector_documents: list[VectorDocument]
    stats: dict[str, Any]


class InMemoryVectorIndex:
    def __init__(self, documents: list[VectorDocument], idf: dict[str, float], vectors: list[dict[str, float]], norms: list[float]):
        self.documents = documents
        self.idf = idf
        self.vectors = vectors
        self.norms = norms

    @classmethod
    def build(cls, documents: list[VectorDocument]) -> "InMemoryVectorIndex":
        document_frequency: Counter[str] = Counter()
        tokenized_documents: list[list[str]] = []
        for document in documents:
            tokens = tokenize(document.text)
            tokenized_documents.append(tokens)
            document_frequency.update(set(tokens))

        total_documents = max(len(documents), 1)
        idf = {
            token: math.log((1 + total_documents) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }

        vectors: list[dict[str, float]] = []
        norms: list[float] = []
        for tokens in tokenized_documents:
            tf = Counter(tokens)
            vector = {token: (count / max(len(tokens), 1)) * idf[token] for token, count in tf.items() if token in idf}
            vectors.append(vector)
            norms.append(math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0)

        return cls(documents=documents, idf=idf, vectors=vectors, norms=norms)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_vector = {
            token: (count / max(len(query_tokens), 1)) * self.idf.get(token, 0.0)
            for token, count in query_tf.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0

        scored_results: list[SearchResult] = []
        for document, document_vector, document_norm in zip(self.documents, self.vectors, self.norms, strict=False):
            dot_product = 0.0
            for token, query_weight in query_vector.items():
                dot_product += query_weight * document_vector.get(token, 0.0)
            if dot_product <= 0.0:
                continue
            score = dot_product / (query_norm * document_norm)
            scored_results.append(SearchResult(document=document, score=score))

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:top_k]


class GraphBundleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name in {"GraphBundle", "VectorDocument"}:
            return globals()[name]
        return super().find_class(module, name)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def as_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def as_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_label(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").lower())).strip()


def build_node_id(kind: str, key: str) -> str:
    return f"{kind}::{normalize_label(key) or key}"


def ensure_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"Missing dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc


def read_table(path: Path, required_columns: Iterable[str]):
    pandas = ensure_dependency("pandas", "pip install -r requirements.txt")
    dataframe = pandas.read_excel(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")
    return dataframe


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return str(value).strip()


def add_or_merge_node(graph: Any, node_id: str, **attributes: Any) -> None:
    if graph.has_node(node_id):
        for key, value in attributes.items():
            if value not in (None, ""):
                graph.nodes[node_id][key] = value
        return
    graph.add_node(node_id, **attributes)


def load_graph_bundle(settings: Settings) -> GraphBundle | None:
    if not settings.graph_cache_path.exists():
        return None
    with settings.graph_cache_path.open("rb") as handle:
        payload = GraphBundleUnpickler(handle).load()

    if isinstance(payload, GraphBundle):
        return payload

    if isinstance(payload, dict):
        vector_documents = []
        for document in payload.get("vector_documents", []):
            if isinstance(document, VectorDocument):
                vector_documents.append(document)
            else:
                vector_documents.append(
                    VectorDocument(
                        document_id=str(document.get("document_id", "")),
                        text=str(document.get("text", "")),
                        metadata=dict(document.get("metadata", {})),
                    )
                )
        return GraphBundle(
            graph=payload["graph"],
            role_lookup=dict(payload["role_lookup"]),
            role_titles=dict(payload["role_titles"]),
            vector_documents=vector_documents,
            stats=dict(payload["stats"]),
        )

    raise ValueError("Unsupported graph bundle cache format.")


def save_graph_bundle(settings: Settings, bundle: GraphBundle) -> None:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph": bundle.graph,
        "role_lookup": bundle.role_lookup,
        "role_titles": bundle.role_titles,
        "vector_documents": [
            {
                "document_id": document.document_id,
                "text": document.text,
                "metadata": document.metadata,
            }
            for document in bundle.vector_documents
        ],
        "stats": bundle.stats,
    }
    with settings.graph_cache_path.open("wb") as handle:
        pickle.dump(payload, handle)


def build_graph_bundle(settings: Settings, postings_limit_override: int | None = None) -> GraphBundle:
    networkx = ensure_dependency("networkx", "pip install -r requirements.txt")

    graph = networkx.MultiDiGraph()
    role_lookup: dict[str, str] = {}
    role_titles: dict[str, str] = {}
    vector_documents: list[VectorDocument] = []
    role_posting_skill_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    occupations = read_table(
        settings.onet_dir / "Occupation Data.xlsx",
        ["O*NET-SOC Code", "Title", "Description"],
    )
    essential_skills = read_table(
        settings.onet_dir / "Essential Skills.xlsx",
        ["O*NET-SOC Code", "Title", "Element Name", "Scale ID", "Data Value"],
    )
    knowledge = read_table(
        settings.onet_dir / "Knowledge.xlsx",
        ["O*NET-SOC Code", "Title", "Element Name", "Scale ID", "Data Value"],
    )
    education = read_table(
        settings.onet_dir / "Education.xlsx",
        ["O*NET-SOC Code", "Title", "Scale ID", "Category", "Data Value"],
    )
    education_categories = read_table(
        settings.onet_dir / "Education Categories.xlsx",
        ["Scale ID", "Category", "Category Description"],
    )
    related_occupations = read_table(
        settings.onet_dir / "Related Occupations.xlsx",
        ["O*NET-SOC Code", "Title", "Related O*NET-SOC Code", "Related Title", "Relatedness Tier", "Index"],
    )

    for row in occupations.to_dict(orient="records"):
        code = safe_text(row["O*NET-SOC Code"])
        title = safe_text(row["Title"])
        description = safe_text(row["Description"])
        role_id = build_node_id("role", code)
        add_or_merge_node(
            graph,
            role_id,
            node_type="Role",
            code=code,
            title=title,
            description=description,
            posting_count=0,
        )
        role_lookup[normalize_label(title)] = code
        role_titles[code] = title

    for row in essential_skills.to_dict(orient="records"):
        if safe_text(row.get("Scale ID")) != "IM":
            continue
        code = safe_text(row["O*NET-SOC Code"])
        role_id = build_node_id("role", code)
        skill_name = safe_text(row["Element Name"])
        skill_id = build_node_id("skill", skill_name)
        add_or_merge_node(graph, skill_id, node_type="Skill", name=skill_name)
        graph.add_edge(
            role_id,
            skill_id,
            relation="REQUIRES_SKILL",
            weight=safe_float(row.get("Data Value")) or 0.0,
            source="onet_essential_skills",
        )

    for row in knowledge.to_dict(orient="records"):
        if safe_text(row.get("Scale ID")) != "IM":
            continue
        code = safe_text(row["O*NET-SOC Code"])
        role_id = build_node_id("role", code)
        knowledge_name = safe_text(row["Element Name"])
        knowledge_id = build_node_id("knowledge", knowledge_name)
        add_or_merge_node(graph, knowledge_id, node_type="KnowledgeArea", name=knowledge_name)
        graph.add_edge(
            role_id,
            knowledge_id,
            relation="REQUIRES_KNOWLEDGE",
            weight=safe_float(row.get("Data Value")) or 0.0,
            source="onet_knowledge",
        )

    education_label_map = {
        int(row["Category"]): safe_text(row["Category Description"])
        for row in education_categories.to_dict(orient="records")
        if safe_text(row.get("Scale ID")) == "RL"
    }
    dominant_education: dict[str, tuple[int, float]] = {}
    for row in education.to_dict(orient="records"):
        if safe_text(row.get("Scale ID")) != "RL":
            continue
        code = safe_text(row["O*NET-SOC Code"])
        category = int(row["Category"])
        percent = safe_float(row.get("Data Value")) or 0.0
        current_best = dominant_education.get(code)
        if current_best is None or percent > current_best[1]:
            dominant_education[code] = (category, percent)

    for code, (category, percent) in dominant_education.items():
        role_id = build_node_id("role", code)
        label = education_label_map.get(category, f"Education category {category}")
        education_id = build_node_id("education", label)
        add_or_merge_node(graph, education_id, node_type="EducationLevel", name=label, category=category)
        graph.add_edge(
            role_id,
            education_id,
            relation="REQUIRES_EDUCATION",
            weight=percent,
            category=category,
            source="onet_education",
        )

    for row in related_occupations.to_dict(orient="records"):
        source_code = safe_text(row["O*NET-SOC Code"])
        target_code = safe_text(row["Related O*NET-SOC Code"])
        graph.add_edge(
            build_node_id("role", source_code),
            build_node_id("role", target_code),
            relation="RELATED_ROLE",
            relatedness_tier=safe_text(row["Relatedness Tier"]),
            rank=int(row["Index"]),
            source="onet_related_occupations",
        )

    pandas = ensure_dependency("pandas", "pip install -r requirements.txt")
    skill_name_by_abr = build_mapping_dict(
        pandas.read_csv(settings.mappings_dir / "skills.csv", dtype=str),
        key_column="skill_abr",
        value_column="skill_name",
    )
    industry_name_by_id = build_mapping_dict(
        pandas.read_csv(settings.mappings_dir / "industries.csv", dtype=str),
        key_column="industry_id",
        value_column="industry_name",
    )
    job_skills_by_id = build_grouped_lookup(
        pandas.read_csv(settings.jobs_dir / "job_skills.csv", dtype=str),
        key_column="job_id",
        value_column="skill_abr",
        mapping=skill_name_by_abr,
    )
    job_industries_by_id = build_grouped_lookup(
        pandas.read_csv(settings.jobs_dir / "job_industries.csv", dtype=str),
        key_column="job_id",
        value_column="industry_id",
        mapping=industry_name_by_id,
    )
    salaries_by_job = build_salary_lookup(pandas.read_csv(settings.jobs_dir / "salaries.csv", low_memory=False))
    companies_by_id = build_company_lookup(pandas.read_csv(settings.companies_dir / "companies.csv", dtype=str, low_memory=False))
    company_industries_by_id = build_grouped_lookup(
        pandas.read_csv(settings.companies_dir / "company_industries.csv", dtype=str),
        key_column="company_id",
        value_column="industry",
    )
    company_specialities_by_id = build_grouped_lookup(
        pandas.read_csv(settings.companies_dir / "company_specialities.csv", dtype=str),
        key_column="company_id",
        value_column="speciality",
    )

    role_titles_lower = list(role_lookup.keys())
    total_postings = 0
    matched_postings = 0
    postings_limit = postings_limit_override if postings_limit_override is not None else settings.postings_limit

    posting_columns = [
        "job_id",
        "company_name",
        "title",
        "description",
        "company_id",
        "formatted_experience_level",
        "skills_desc",
        "location",
        "work_type",
        "normalized_salary",
        "formatted_work_type",
    ]
    postings_iter = pandas.read_csv(
        settings.postings_path,
        usecols=posting_columns,
        dtype=str,
        low_memory=False,
        chunksize=settings.postings_chunk_size,
    )
    for chunk in postings_iter:
        if postings_limit is not None and total_postings >= postings_limit:
            break
        if postings_limit is not None:
            remaining = postings_limit - total_postings
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        for row in chunk.to_dict(orient="records"):
            total_postings += 1
            job_id = safe_text(row.get("job_id"))
            posting_title = safe_text(row.get("title"))
            description = safe_text(row.get("description"))
            company_id = safe_text(row.get("company_id"))
            company_name = safe_text(row.get("company_name"))
            matched_role_code = match_role_code(posting_title, role_lookup, role_titles_lower, settings.title_match_cutoff)
            matched_role_id = build_node_id("role", matched_role_code) if matched_role_code else None
            if matched_role_id and graph.has_node(matched_role_id):
                matched_postings += 1
                graph.nodes[matched_role_id]["posting_count"] = int(graph.nodes[matched_role_id].get("posting_count", 0)) + 1

            posting_id = build_node_id("posting", job_id or f"row-{total_postings}")
            posting_text = "\n".join(
                part for part in [posting_title, safe_text(row.get("skills_desc")), description] if part
            )
            add_or_merge_node(
                graph,
                posting_id,
                node_type="JobPosting",
                job_id=job_id,
                title=posting_title,
                company_name=company_name,
                location=safe_text(row.get("location")),
                experience_level=safe_text(row.get("formatted_experience_level")),
                work_type=safe_text(row.get("formatted_work_type")) or safe_text(row.get("work_type")),
                normalized_salary=safe_text(row.get("normalized_salary")) or safe_text(salaries_by_job.get(job_id)),
                matched_role_code=matched_role_code,
                source="postings.csv",
            )

            if matched_role_id and graph.has_node(matched_role_id):
                graph.add_edge(posting_id, matched_role_id, relation="POSTING_FOR_ROLE", source="postings.csv")

            skills_for_posting = job_skills_by_id.get(job_id, [])
            for skill_name in skills_for_posting:
                skill_id = build_node_id("skill", skill_name)
                add_or_merge_node(graph, skill_id, node_type="Skill", name=skill_name)
                graph.add_edge(posting_id, skill_id, relation="POSTING_REQUIRES_SKILL", source="postings.csv")
                if matched_role_id:
                    role_posting_skill_counts[(matched_role_id, skill_name)] += 1

            industries_for_posting = job_industries_by_id.get(job_id, [])
            for industry_name in industries_for_posting:
                industry_id = build_node_id("industry", industry_name)
                add_or_merge_node(graph, industry_id, node_type="Industry", name=industry_name)
                graph.add_edge(posting_id, industry_id, relation="IN_INDUSTRY", source="postings.csv")

            if company_id:
                company_metadata = companies_by_id.get(company_id, {})
                company_node_id = build_node_id("company", company_id)
                add_or_merge_node(
                    graph,
                    company_node_id,
                    node_type="Company",
                    company_id=company_id,
                    name=company_metadata.get("name") or company_name,
                    description=company_metadata.get("description"),
                    city=company_metadata.get("city"),
                    country=company_metadata.get("country"),
                    url=company_metadata.get("url"),
                )
                graph.add_edge(posting_id, company_node_id, relation="POSTED_BY", source="postings.csv")

                for industry_name in company_industries_by_id.get(company_id, []):
                    industry_id = build_node_id("industry", industry_name)
                    add_or_merge_node(graph, industry_id, node_type="Industry", name=industry_name)
                    graph.add_edge(company_node_id, industry_id, relation="IN_INDUSTRY", source="companies.csv")

                for speciality in company_specialities_by_id.get(company_id, []):
                    speciality_id = build_node_id("speciality", speciality)
                    add_or_merge_node(graph, speciality_id, node_type="Speciality", name=speciality)
                    graph.add_edge(company_node_id, speciality_id, relation="HAS_SPECIALITY", source="companies.csv")

            if posting_text:
                vector_documents.append(
                    VectorDocument(
                        document_id=posting_id,
                        text=posting_text,
                        metadata={
                            "job_id": job_id,
                            "title": posting_title,
                            "matched_role_code": matched_role_code,
                            "matched_role_title": role_titles.get(matched_role_code or "", ""),
                            "company_name": company_name,
                            "location": safe_text(row.get("location")),
                        },
                    )
                )

    for (role_id, skill_name), count in role_posting_skill_counts.items():
        skill_id = build_node_id("skill", skill_name)
        add_or_merge_node(graph, skill_id, node_type="Skill", name=skill_name)
        graph.add_edge(
            role_id,
            skill_id,
            relation="OBSERVED_SKILL_DEMAND",
            weight=count,
            source="job_postings",
        )

    stats = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "role_count": sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "Role"),
        "posting_count": total_postings,
        "matched_posting_count": matched_postings,
        "vector_document_count": len(vector_documents),
    }
    return GraphBundle(graph=graph, role_lookup=role_lookup, role_titles=role_titles, vector_documents=vector_documents, stats=stats)


def build_mapping_dict(dataframe, key_column: str, value_column: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in dataframe.to_dict(orient="records"):
        key = safe_text(row.get(key_column))
        value = safe_text(row.get(value_column))
        if key and value:
            mapping[key] = value
    return mapping


def build_grouped_lookup(dataframe, key_column: str, value_column: str, mapping: dict[str, str] | None = None) -> dict[str, list[str]]:
    lookup: defaultdict[str, list[str]] = defaultdict(list)
    for row in dataframe.to_dict(orient="records"):
        key = safe_text(row.get(key_column))
        value = safe_text(row.get(value_column))
        if mapping is not None:
            value = mapping.get(value, value)
        if key and value and value not in lookup[key]:
            lookup[key].append(value)
    return dict(lookup)


def build_salary_lookup(dataframe) -> dict[str, float]:
    salary_lookup: dict[str, float] = {}
    for row in dataframe.to_dict(orient="records"):
        job_id = safe_text(row.get("job_id"))
        salary = safe_float(row.get("med_salary"))
        if salary is None:
            salary = safe_float(row.get("max_salary"))
        if salary is None:
            salary = safe_float(row.get("min_salary"))
        if job_id and salary is not None:
            salary_lookup[job_id] = salary
    return salary_lookup


def build_company_lookup(dataframe) -> dict[str, dict[str, str]]:
    company_lookup: dict[str, dict[str, str]] = {}
    for row in dataframe.to_dict(orient="records"):
        company_id = safe_text(row.get("company_id"))
        if not company_id:
            continue
        company_lookup[company_id] = {
            "name": safe_text(row.get("name")),
            "description": safe_text(row.get("description")),
            "city": safe_text(row.get("city")),
            "country": safe_text(row.get("country")),
            "url": safe_text(row.get("url")),
        }
    return company_lookup


def match_role_code(title: str, role_lookup: dict[str, str], role_titles_lower: list[str], cutoff: float) -> str | None:
    normalized_title = normalize_label(title)
    if not normalized_title:
        return None
    if normalized_title in role_lookup:
        return role_lookup[normalized_title]
    match = difflib.get_close_matches(normalized_title, role_titles_lower, n=1, cutoff=cutoff)
    if match:
        return role_lookup[match[0]]
    return None


def parse_json_block(raw_text: str) -> Any:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty JSON response")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start_object = raw_text.find("{")
        start_array = raw_text.find("[")
        start_positions = [index for index in [start_object, start_array] if index >= 0]
        if not start_positions:
            raise
        start = min(start_positions)
        end_object = raw_text.rfind("}")
        end_array = raw_text.rfind("]")
        end = max(end_object, end_array)
        if end <= start:
            raise
        return json.loads(raw_text[start : end + 1])


def call_nvidia_chat(settings: Settings, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 800) -> str:
    api_key = settings.resolved_chat_api_key
    if not api_key:
        raise RuntimeError("NVIDIA_CHAT_API_KEY is not set. Provide it directly or use NVIDIA_API_KEY as a shared fallback.")

    payload = {
        "model": settings.nvidia_chat_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    http_request = urllib.request.Request(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA chat request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NVIDIA chat request failed: {exc}") from exc

    payload = json.loads(raw_response)
    return payload["choices"][0]["message"]["content"]


def route_query(settings: Settings, user_query: str) -> QueryState:
    heuristic = heuristic_route_query(user_query)
    if not settings.resolved_chat_api_key:
        return heuristic

    messages = [
        {"role": "system", "content": QUERY_ROUTING_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    try:
        raw_response = call_nvidia_chat(settings, messages, temperature=0.0, max_tokens=300)
        parsed = parse_json_block(raw_response)
    except Exception:
        return heuristic

    if not isinstance(parsed, dict):
        return heuristic

    return QueryState(
        original_query=user_query,
        question_type=safe_text(parsed.get("question_type")) or heuristic.question_type,
        current_role=blank_to_none(safe_text(parsed.get("current_role"))) or heuristic.current_role,
        target_role=blank_to_none(safe_text(parsed.get("target_role"))) or heuristic.target_role,
        current_skills=[safe_text(skill) for skill in parsed.get("current_skills", []) if safe_text(skill)] or heuristic.current_skills,
        preferred_industry=blank_to_none(safe_text(parsed.get("preferred_industry"))) or heuristic.preferred_industry,
        location=blank_to_none(safe_text(parsed.get("location"))) or heuristic.location,
        experience_level=blank_to_none(safe_text(parsed.get("experience_level"))) or heuristic.experience_level,
        is_vague=bool(parsed.get("is_vague")),
        missing_fields=[safe_text(field) for field in parsed.get("missing_fields", []) if safe_text(field)],
    )


def heuristic_route_query(user_query: str) -> QueryState:
    lowered = user_query.lower()
    query_state = QueryState(original_query=user_query)

    if any(phrase in lowered for phrase in ["become", "transition", "move into", "switch to"]):
        query_state.question_type = "skill_gap"
    elif any(phrase in lowered for phrase in ["recommend", "suggest", "best career", "better job"]):
        query_state.question_type = "role_recommendation"
    elif any(phrase in lowered for phrase in ["what is", "tell me about", "education", "salary", "role"]):
        query_state.question_type = "role_information"

    become_match = re.search(r"become an? ([a-z0-9\-\s]+)", lowered)
    if become_match:
        query_state.target_role = become_match.group(1).strip(" .,!?")

    work_as_match = re.search(r"work as an? ([a-z0-9\-\s]+)", lowered)
    if work_as_match and not query_state.target_role:
        query_state.target_role = work_as_match.group(1).strip(" .,!?")

    skill_match = re.search(r"know ([a-z0-9,\-\s]+)", lowered)
    if skill_match:
        query_state.current_skills = split_skill_list(skill_match.group(1))

    if "i am a" in lowered:
        after = lowered.split("i am a", 1)[1].split(" and ", 1)[0]
        query_state.current_role = after.strip(" .,!?")

    if query_state.question_type in {"skill_gap", "role_recommendation"} and not query_state.target_role and not query_state.current_role:
        query_state.is_vague = True
        query_state.missing_fields.append("target_role")
    if query_state.question_type in {"skill_gap", "role_recommendation"} and not query_state.current_skills:
        query_state.is_vague = True
        query_state.missing_fields.append("current_skills")
    if not query_state.original_query.strip() or len(tokenize(query_state.original_query)) < 4:
        query_state.is_vague = True

    return query_state


def split_skill_list(raw_skills: str) -> list[str]:
    pieces = re.split(r",|/| and ", raw_skills)
    cleaned = [piece.strip().title() for piece in pieces if piece.strip()]
    return cleaned


def get_clarification_fields(query_state: QueryState) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    if query_state.question_type in {"skill_gap", "role_recommendation"} and not query_state.target_role:
        fields.append(
            {
                "name": "target_role",
                "label": "Target role",
                "placeholder": "Machine Learning Engineer, Data Scientist, Product Analyst...",
            }
        )
    if query_state.question_type in {"skill_gap", "role_recommendation"} and not query_state.current_skills:
        fields.append(
            {
                "name": "current_skills",
                "label": "Current skills",
                "placeholder": "Python, SQL, Excel, communication...",
            }
        )
    if not query_state.current_role and query_state.question_type == "role_recommendation":
        fields.append(
            {
                "name": "current_role",
                "label": "Current role or background",
                "placeholder": "Student, junior analyst, marketing coordinator...",
            }
        )
    if not query_state.preferred_industry:
        fields.append(
            {
                "name": "preferred_industry",
                "label": "Preferred industry",
                "placeholder": "Optional: fintech, healthcare, education...",
            }
        )
    if not query_state.location:
        fields.append(
            {
                "name": "location",
                "label": "Preferred location",
                "placeholder": "Optional: Dubai, remote, Bengaluru...",
            }
        )
    return fields


def apply_clarification_answers(query_state: QueryState, answers: dict[str, str]) -> QueryState:
    for field_name, raw_value in answers.items():
        value = raw_value.strip()
        if not value:
            continue
        if field_name == "current_skills":
            query_state.current_skills = split_skill_list(value)
        elif hasattr(query_state, field_name):
            setattr(query_state, field_name, value)

    query_state.is_vague = False
    query_state.missing_fields = []
    remaining_fields = get_clarification_fields(query_state)
    if remaining_fields:
        query_state.is_vague = True
        query_state.missing_fields = [field["name"] for field in remaining_fields]
    return query_state


def collect_clarifications(query_state: QueryState) -> QueryState:
    prompt_lookup = {
        "target_role": "What target role are you interested in? ",
        "current_skills": "What skills do you already have? Separate them with commas. ",
        "current_role": "What is your current role or background? ",
        "preferred_industry": "Do you have a preferred industry? Press Enter to skip. ",
        "location": "Do you have a preferred location? Press Enter to skip. ",
    }
    answers: dict[str, str] = {}
    for field in get_clarification_fields(query_state):
        answers[field["name"]] = input(prompt_lookup[field["name"]]).strip()

    apply_clarification_answers(query_state, answers)
    return query_state


def retrieve_context(bundle: GraphBundle, vector_index: InMemoryVectorIndex, query_state: QueryState, settings: Settings) -> RetrievedContext:
    graph = bundle.graph
    role_code = resolve_role_code(bundle, vector_index, query_state)
    graph_facts: list[str] = []
    missing_skills: list[str] = []
    recommended_roles: list[dict[str, Any]] = []
    role_title = bundle.role_titles.get(role_code) if role_code else None

    if role_code:
        role_id = build_node_id("role", role_code)
        role_node = graph.nodes[role_id]
        graph_facts.append(f"Resolved target role: {role_node.get('title')} ({role_code})")
        graph_facts.extend(build_role_fact_block(graph, role_id, settings.graph_top_k))

        required_skills = {
            normalize_label(skill): skill
            for skill in extract_neighbor_names(graph, role_id, "REQUIRES_SKILL", limit=settings.graph_top_k)
        }
        observed_skills = {
            normalize_label(skill): skill
            for skill in extract_neighbor_names(graph, role_id, "OBSERVED_SKILL_DEMAND", limit=settings.graph_top_k)
        }
        current_skills_normalized = {normalize_label(skill) for skill in query_state.current_skills}
        for skill_key, skill_name in {**required_skills, **observed_skills}.items():
            if current_skills_normalized and skill_key not in current_skills_normalized:
                missing_skills.append(skill_name)
    elif query_state.current_skills:
        recommended_roles = recommend_roles(bundle, query_state.current_skills, limit=settings.graph_top_k)
        if recommended_roles:
            graph_facts.append("Recommended roles based on overlapping observed or required skills:")
            for recommendation in recommended_roles:
                graph_facts.append(
                    f"- {recommendation['title']} (score={recommendation['score']:.2f}, matched_skills={', '.join(recommendation['matched_skills'])})"
                )

    vector_query = query_state.original_query
    if role_title and role_title.lower() not in vector_query.lower():
        vector_query = f"{vector_query}\nTarget role: {role_title}"
    vector_results = vector_index.search(vector_query, settings.vector_top_k)
    context_lines = ["GRAPH CONTEXT:"]
    context_lines.extend(graph_facts or ["- No graph facts found."])
    if missing_skills:
        context_lines.append("- Missing skills relative to the resolved target role: " + ", ".join(missing_skills[: settings.graph_top_k]))
    if recommended_roles:
        context_lines.append("- Candidate roles: " + "; ".join(item["title"] for item in recommended_roles[: settings.graph_top_k]))
    context_lines.append("VECTOR CONTEXT:")
    if vector_results:
        for result in vector_results:
            snippet = shorten(result.document.text, 320)
            context_lines.append(
                f"- score={result.score:.3f} | title={result.document.metadata.get('title', '')} | role={result.document.metadata.get('matched_role_title', '')} | evidence={snippet}"
            )
    else:
        context_lines.append("- No vector evidence found.")

    return RetrievedContext(
        role_code=role_code,
        role_title=role_title,
        graph_facts=graph_facts,
        vector_results=vector_results,
        missing_skills=missing_skills[: settings.graph_top_k],
        recommended_roles=recommended_roles[: settings.graph_top_k],
        context_text="\n".join(context_lines),
    )


def resolve_role_code(bundle: GraphBundle, vector_index: InMemoryVectorIndex, query_state: QueryState) -> str | None:
    for candidate in [query_state.target_role, query_state.current_role]:
        if not candidate:
            continue
        match = match_role_code(candidate, bundle.role_lookup, list(bundle.role_lookup.keys()), cutoff=0.72)
        if match:
            return match

    if query_state.target_role:
        ranked = vector_index.search(query_state.target_role, 10)
        scores: defaultdict[str, float] = defaultdict(float)
        for result in ranked:
            matched_role_code = safe_text(result.document.metadata.get("matched_role_code"))
            if matched_role_code:
                scores[matched_role_code] += result.score
        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]

    return None


def build_role_fact_block(graph: Any, role_id: str, limit: int) -> list[str]:
    role_data = graph.nodes[role_id]
    facts = [
        f"Description: {shorten(safe_text(role_data.get('description')), 240)}",
        f"Observed posting count: {role_data.get('posting_count', 0)}",
    ]

    required_skills = extract_neighbor_names(graph, role_id, "REQUIRES_SKILL", limit=limit)
    if required_skills:
        facts.append("Top required skills: " + ", ".join(required_skills))

    observed_skills = extract_neighbor_names(graph, role_id, "OBSERVED_SKILL_DEMAND", limit=limit)
    if observed_skills:
        facts.append("Observed posting skills: " + ", ".join(observed_skills))

    knowledge_areas = extract_neighbor_names(graph, role_id, "REQUIRES_KNOWLEDGE", limit=limit)
    if knowledge_areas:
        facts.append("Top knowledge areas: " + ", ".join(knowledge_areas))

    education_levels = extract_neighbor_names(graph, role_id, "REQUIRES_EDUCATION", limit=3)
    if education_levels:
        facts.append("Typical education signal: " + ", ".join(education_levels))

    related_roles = extract_neighbor_names(graph, role_id, "RELATED_ROLE", limit=limit)
    if related_roles:
        facts.append("Related occupations: " + ", ".join(related_roles))

    return facts


def extract_neighbor_names(graph: Any, source_node: str, relation: str, limit: int) -> list[str]:
    matches: list[tuple[float, str]] = []
    for _, target, edge_data in graph.out_edges(source_node, data=True):
        if edge_data.get("relation") != relation:
            continue
        target_data = graph.nodes[target]
        label = safe_text(target_data.get("name") or target_data.get("title") or target_data.get("code"))
        weight = safe_float(edge_data.get("weight")) or 0.0
        rank = safe_float(edge_data.get("rank")) or 999.0
        sort_score = -rank if relation == "RELATED_ROLE" else weight
        matches.append((sort_score, label))

    matches.sort(key=lambda item: item[0], reverse=True)
    output: list[str] = []
    for _, label in matches:
        if label and label not in output:
            output.append(label)
        if len(output) >= limit:
            break
    return output


def recommend_roles(bundle: GraphBundle, current_skills: list[str], limit: int) -> list[dict[str, Any]]:
    graph = bundle.graph
    normalized_user_skills = {normalize_label(skill) for skill in current_skills if normalize_label(skill)}
    recommendations: list[dict[str, Any]] = []
    max_postings = max(
        (int(data.get("posting_count", 0)) for _, data in graph.nodes(data=True) if data.get("node_type") == "Role"),
        default=1,
    )

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "Role":
            continue
        skill_pool = extract_neighbor_names(graph, node_id, "OBSERVED_SKILL_DEMAND", limit=12)
        if not skill_pool:
            skill_pool = extract_neighbor_names(graph, node_id, "REQUIRES_SKILL", limit=12)
        normalized_role_skills = {normalize_label(skill): skill for skill in skill_pool if normalize_label(skill)}
        matched_skills = [skill for key, skill in normalized_role_skills.items() if key in normalized_user_skills]
        if not matched_skills:
            continue

        overlap_score = len(matched_skills) / max(len(normalized_role_skills), 1)
        demand_score = int(data.get("posting_count", 0)) / max(max_postings, 1)
        total_score = 0.7 * overlap_score + 0.3 * demand_score
        recommendations.append(
            {
                "code": safe_text(data.get("code")),
                "title": safe_text(data.get("title")),
                "score": total_score,
                "matched_skills": matched_skills,
            }
        )

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations[:limit]


def answer_with_llm(settings: Settings, query_state: QueryState, retrieved_context: RetrievedContext) -> str:
    messages = [
        {"role": "system", "content": GROUNDED_RESPONSE_SYSTEM_PROMPT},
        {
            "role": "user", "content": f"Original user question:\n{query_state.original_query}\n\nRetrieved context:\n{retrieved_context.context_text}",
        },
    ]
    try:
        return call_nvidia_chat(settings, messages, temperature=0.1, max_tokens=700)
    except Exception:
        return deterministic_answer(query_state, retrieved_context)


def deterministic_answer(query_state: QueryState, retrieved_context: RetrievedContext) -> str:
    lines = []
    if retrieved_context.role_title:
        lines.append(f"Closest role match: {retrieved_context.role_title}.")
    if retrieved_context.missing_skills:
        lines.append("Skills that appear missing from the retrieved evidence: " + ", ".join(retrieved_context.missing_skills) + ".")
    if retrieved_context.recommended_roles:
        lines.append(
            "Recommended roles from the current graph evidence: "
            + "; ".join(item["title"] for item in retrieved_context.recommended_roles[:5])
            + "."
        )
    if retrieved_context.vector_results:
        top_result = retrieved_context.vector_results[0]
        lines.append(
            "Top supporting posting evidence: "
            + shorten(top_result.document.text, 260)
            + "."
        )
    if not lines:
        lines.append("The retrieved context was not sufficient to answer this question confidently.")
    lines.append("Evidence summary:")
    lines.extend(f"- {fact}" for fact in retrieved_context.graph_facts[:6])
    return "\n".join(lines)


def shorten(text: str, max_length: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def load_pdf_text(pdf_path: Path) -> str:
    loader_module = ensure_dependency("langchain_community.document_loaders", "pip install -r requirements.txt")
    loader = loader_module.PyPDFLoader(str(pdf_path))
    pages = loader.load()
    return "\n\n".join(page.page_content for page in pages)


def extract_triples(settings: Settings, text: str) -> list[dict[str, str]]:
    if not settings.resolved_chat_api_key:
        raise RuntimeError("NVIDIA_CHAT_API_KEY is required for PDF triple extraction, or set NVIDIA_API_KEY as a shared fallback.")
    messages = [
        {"role": "system", "content": TRIPLE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": text[:12000]},
    ]
    raw_response = call_nvidia_chat(settings, messages, temperature=0.0, max_tokens=1200)
    parsed = parse_json_block(raw_response)
    if isinstance(parsed, dict):
        triples = parsed.get("triples", [])
    else:
        triples = parsed
    valid_triples = []
    for triple in triples:
        node_1 = safe_text(triple.get("node_1"))
        edge = safe_text(triple.get("edge"))
        node_2 = safe_text(triple.get("node_2"))
        if node_1 and edge and node_2:
            valid_triples.append({"node_1": node_1, "edge": edge, "node_2": node_2})
    return valid_triples


def print_stats(bundle: GraphBundle) -> None:
    print(json.dumps(bundle.stats, indent=2))


def load_or_build_bundle(settings: Settings, *, rebuild: bool = False, postings_limit_override: int | None = None) -> GraphBundle:
    if not rebuild:
        cached_bundle = load_graph_bundle(settings)
        if cached_bundle is not None:
            return cached_bundle

    if os.environ.get("VERCEL"):
        raise RuntimeError(
            "No cached graph bundle is available in this deployment. Run the Vercel build step to generate artifacts/career_kg_bundle.pkl before the function is bundled."
        )

    bundle = build_graph_bundle(settings, postings_limit_override=postings_limit_override)
    save_graph_bundle(settings, bundle)
    return bundle


def prepare_runtime(settings: Settings, *, rebuild: bool = False, postings_limit_override: int | None = None) -> tuple[GraphBundle, InMemoryVectorIndex]:
    bundle = load_or_build_bundle(settings, rebuild=rebuild, postings_limit_override=postings_limit_override)
    vector_index = InMemoryVectorIndex.build(bundle.vector_documents)
    return bundle, vector_index


def run_query_turn(settings: Settings, bundle: GraphBundle, vector_index: InMemoryVectorIndex, query_state: QueryState) -> dict[str, Any]:
    retrieved_context = retrieve_context(bundle, vector_index, query_state, settings)
    answer = answer_with_llm(settings, query_state, retrieved_context) if settings.resolved_chat_api_key else deterministic_answer(query_state, retrieved_context)
    return {
        "status": "answer",
        "answer": answer,
        "graph_facts": retrieved_context.graph_facts,
        "missing_skills": retrieved_context.missing_skills,
        "recommended_roles": retrieved_context.recommended_roles,
        "role_title": retrieved_context.role_title,
        "context_text": retrieved_context.context_text,
    }


def command_build(args: argparse.Namespace, settings: Settings) -> int:
    bundle = build_graph_bundle(settings, postings_limit_override=args.postings_limit)
    save_graph_bundle(settings, bundle)
    print("Graph build complete.")
    print_stats(bundle)
    return 0


def command_chat(args: argparse.Namespace, settings: Settings) -> int:
    bundle = load_graph_bundle(settings)
    if bundle is None or args.rebuild:
        bundle = build_graph_bundle(settings, postings_limit_override=args.postings_limit)
        save_graph_bundle(settings, bundle)

    vector_index = InMemoryVectorIndex.build(bundle.vector_documents)
    print("Career KG chat is ready. Type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            return 0

        query_state = route_query(settings, user_query)
        if query_state.is_vague:
            print("I need a bit more context before I query the graph.")
            query_state = collect_clarifications(query_state)

        retrieved_context = retrieve_context(bundle, vector_index, query_state, settings)
        answer = answer_with_llm(settings, query_state, retrieved_context) if settings.resolved_chat_api_key else deterministic_answer(query_state, retrieved_context)
        print("\nAssistant:")
        print(answer)

    return 0


def command_extract_pdf(args: argparse.Namespace, settings: Settings) -> int:
    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = settings.root_dir / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = load_pdf_text(pdf_path)
    triples = extract_triples(settings, text)
    print(json.dumps({"triples": triples}, indent=2))
    return 0


def command_inspect(args: argparse.Namespace, settings: Settings) -> int:
    bundle = load_graph_bundle(settings)
    if bundle is None:
        print("No cached graph bundle found. Run the build command first.")
        return 1
    print_stats(bundle)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Career KG chat prototype using O*NET and job posting data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the graph bundle and cache it to disk.")
    build_parser.add_argument("--postings-limit", type=int, default=None, help="Optional override for the number of postings to ingest.")

    chat_parser = subparsers.add_parser("chat", help="Start the interactive KG-grounded chat loop.")
    chat_parser.add_argument("--rebuild", action="store_true", help="Rebuild the graph bundle before starting chat.")
    chat_parser.add_argument("--postings-limit", type=int, default=None, help="Optional override used only when rebuild is enabled.")

    pdf_parser = subparsers.add_parser("extract-pdf", help="Extract JSON triples from a sample PDF with NVIDIA NIM.")
    pdf_parser.add_argument("pdf_path", help="Path to the PDF file.")

    subparsers.add_parser("inspect", help="Show cached graph statistics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env(Path(__file__).resolve().parent)
    try:
        if args.command == "build":
            return command_build(args, settings)
        if args.command == "chat":
            return command_chat(args, settings)
        if args.command == "extract-pdf":
            return command_extract_pdf(args, settings)
        if args.command == "inspect":
            return command_inspect(args, settings)
        parser.error(f"Unsupported command: {args.command}")
    except MissingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
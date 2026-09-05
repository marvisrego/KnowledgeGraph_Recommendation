"""Neo4j Aura persistence and NetworkX compatibility loading."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import networkx as nx


NODE_LABELS = {
    "role": "Role",
    "skill": "Skill",
    "element": "Element",
    "isco_group": "ISCOGroup",
    "skill_group": "SkillGroup",
    "qualification": "Qualification",
    "job_zone": "JobZone",
}
RELATION_TYPES = {
    "REQUIRES",
    "BELONGS_TO",
    "BROADER_THAN",
    "NARROWER_THAN",
    "SIMILAR_TO",
    "SAME_ISCO_GROUP",
    "TRANSITIONS_TO",
    "TYPICALLY_REQUIRES_QUALIFICATION",
    "IN_JOB_ZONE",
}


def validate_neo4j_settings(settings) -> None:
    missing = [
        name
        for name, value in (
            ("KG_URI", settings.kg_uri),
            ("KG_USER", settings.kg_user),
            ("KG_PASS", settings.kg_pass),
            ("KG_ID", settings.kg_id),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise RuntimeError(f"Missing required Neo4j settings: {', '.join(missing)}")
    if settings.neo4j_batch_size < 1:
        raise RuntimeError("NEO4J_BATCH_SIZE must be positive")


def create_neo4j_driver(settings):
    """Create and verify one official Neo4j driver without logging secrets."""
    validate_neo4j_settings(settings)
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.kg_uri,
            auth=(settings.kg_user, settings.kg_pass),
        )
        if settings.neo4j_database:
            driver.verify_connectivity(database=settings.neo4j_database)
        else:
            driver.verify_connectivity()
        return driver
    except Exception as exc:
        try:
            driver.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        raise RuntimeError("Neo4j Aura connectivity check failed") from exc


def _primitive(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        items = [_primitive(item) for item in value]
        items = [item for item in items if item is not None]
        if not items:
            return []
        types = {type(item) for item in items}
        return items if len(types) == 1 and types <= {str, int, float, bool} else [str(item) for item in items]
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def normalize_properties(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): converted
        for key, value in data.items()
        if (converted := _primitive(value)) is not None
    }


def entity_key(graph_id: str, entity_id: str) -> str:
    return f"{graph_id}::{entity_id}"


def node_upload_rows(graph: nx.MultiDiGraph, graph_id: str) -> dict[str, list[dict]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        node_type = str(data.get("type", ""))
        label = NODE_LABELS.get(node_type)
        if label is None:
            raise ValueError(f"Unsupported node type: {node_type!r}")
        identifier = str(node_id)
        properties = normalize_properties(data)
        properties.update(
            {
                "graph_id": graph_id,
                "entity_id": identifier,
                "entity_key": entity_key(graph_id, identifier),
            }
        )
        grouped[label].append({"entity_key": properties["entity_key"], "properties": properties})
    return dict(grouped)


def relationship_upload_rows(graph: nx.MultiDiGraph, graph_id: str) -> dict[str, list[dict]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for source, target, key, data in graph.edges(keys=True, data=True):
        relation = str(data.get("relation", "")).upper()
        if relation not in RELATION_TYPES:
            raise ValueError(f"Unsupported relationship type: {relation!r}")
        source_id, target_id = str(source), str(target)
        edge_key = f"{graph_id}::{source_id}::{relation}::{target_id}::{key}"
        properties = normalize_properties(data)
        properties.update(
            {
                "graph_id": graph_id,
                "edge_key": edge_key,
                "original_edge_key": str(key),
            }
        )
        grouped[relation].append(
            {
                "source_key": entity_key(graph_id, source_id),
                "target_key": entity_key(graph_id, target_id),
                "edge_key": edge_key,
                "properties": properties,
            }
        )
    return dict(grouped)


def batches(rows: Sequence[dict], size: int) -> Iterator[list[dict]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(rows), size):
        yield list(rows[offset : offset + size])


class Neo4jGraphStore:
    """Dedicated-project Aura operations through one shared driver."""

    def __init__(self, driver, *, database: str | None, graph_id: str, batch_size: int = 1000) -> None:
        if not graph_id:
            raise ValueError("graph_id is required")
        self.driver = driver
        self.database = database
        self.graph_id = graph_id
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings) -> "Neo4jGraphStore":
        driver = create_neo4j_driver(settings)
        return cls(
            driver,
            database=settings.neo4j_database,
            graph_id=str(settings.kg_id),
            batch_size=settings.neo4j_batch_size,
        )

    def close(self) -> None:
        self.driver.close()

    def _write(self, query: str, **parameters: Any) -> None:
        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(lambda tx: tx.run(query, **parameters).consume())
        except Exception as exc:
            raise RuntimeError("Neo4j Aura write operation failed") from exc

    def _read(self, query: str, **parameters: Any) -> list[dict]:
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(
                    lambda tx: [dict(record) for record in tx.run(query, **parameters)]
                )
        except Exception as exc:
            raise RuntimeError("Neo4j Aura read operation failed") from exc

    def clear_database(self) -> None:
        """Clear the dedicated Aura database. Called only by the rebuild command."""
        self._write("MATCH (n) DETACH DELETE n")

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT career_entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_key IS UNIQUE",
            "CREATE INDEX career_graph_id IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)",
            "CREATE INDEX career_entity_id IF NOT EXISTS FOR (n:Entity) ON (n.entity_id)",
            "CREATE INDEX career_normalized_title IF NOT EXISTS FOR (n:Entity) ON (n.normalized_title)",
            "CREATE INDEX career_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX career_source IF NOT EXISTS FOR (n:Entity) ON (n.source)",
        )
        for statement in statements:
            self._write(statement)

    def upload_graph(self, graph: nx.MultiDiGraph) -> dict[str, int]:
        node_groups = node_upload_rows(graph, self.graph_id)
        relationship_groups = relationship_upload_rows(graph, self.graph_id)
        for label in sorted(node_groups):
            if label not in NODE_LABELS.values():
                raise ValueError(f"Unsupported node label: {label}")
            query = (
                f"UNWIND $rows AS row "
                f"MERGE (n:Entity:{label} {{entity_key: row.entity_key}}) "
                "SET n += row.properties"
            )
            for batch in batches(node_groups[label], self.batch_size):
                self._write(query, rows=batch)

        for relation in sorted(relationship_groups):
            if relation not in RELATION_TYPES:
                raise ValueError(f"Unsupported relationship type: {relation}")
            query = (
                "UNWIND $rows AS row "
                "MATCH (source:Entity {entity_key: row.source_key}) "
                "MATCH (target:Entity {entity_key: row.target_key}) "
                f"MERGE (source)-[edge:{relation} {{edge_key: row.edge_key}}]->(target) "
                "SET edge += row.properties"
            )
            for batch in batches(relationship_groups[relation], self.batch_size):
                self._write(query, rows=batch)
        return {
            "nodes": sum(len(rows) for rows in node_groups.values()),
            "relationships": sum(len(rows) for rows in relationship_groups.values()),
        }

    def counts(self) -> dict[str, Any]:
        rows = self._read(
            "MATCH (n:Entity {graph_id: $graph_id}) "
            "WITH count(n) AS nodes "
            "OPTIONAL MATCH ()-[r {graph_id: $graph_id}]->() "
            "RETURN nodes, count(r) AS relationships",
            graph_id=self.graph_id,
        )
        return rows[0] if rows else {"nodes": 0, "relationships": 0}

    def representative_queries(self) -> dict[str, Any]:
        relation_rows = self._read(
            "MATCH (:Entity {graph_id: $graph_id})-[r]->(:Entity {graph_id: $graph_id}) "
            "RETURN type(r) AS relation, count(*) AS count ORDER BY relation",
            graph_id=self.graph_id,
        )
        sample = self._read(
            "MATCH (role:Role {graph_id: $graph_id})-[r:REQUIRES]->(skill:Entity) "
            "RETURN role.entity_id AS role_id, role.title AS role, "
            "skill.entity_id AS skill_id, skill.title AS skill LIMIT 5",
            graph_id=self.graph_id,
        )
        transitions = self._read(
            "MATCH (source:Role {graph_id: $graph_id})-[r:TRANSITIONS_TO]->(target:Role) "
            "RETURN source.title AS source, target.title AS target, r.count AS count "
            "ORDER BY count DESC LIMIT 5",
            graph_id=self.graph_id,
        )
        return {
            "relationship_types": {row["relation"]: row["count"] for row in relation_rows},
            "sample_requirements": sample,
            "sample_transitions": transitions,
        }

    def load_networkx(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        node_rows = self._read(
            "MATCH (n:Entity {graph_id: $graph_id}) "
            "RETURN n.entity_id AS entity_id, properties(n) AS properties",
            graph_id=self.graph_id,
        )
        for row in node_rows:
            properties = dict(row["properties"])
            entity_id = str(row["entity_id"])
            for key in ("graph_id", "entity_id", "entity_key"):
                properties.pop(key, None)
            graph.add_node(entity_id, **properties)

        edge_rows = self._read(
            "MATCH (source:Entity {graph_id: $graph_id})-[r]->(target:Entity {graph_id: $graph_id}) "
            "RETURN source.entity_id AS source_id, target.entity_id AS target_id, "
            "type(r) AS relation, properties(r) AS properties",
            graph_id=self.graph_id,
        )
        for row in edge_rows:
            properties = dict(row["properties"])
            key = str(properties.pop("original_edge_key", properties.get("edge_key", row["relation"])))
            properties.pop("graph_id", None)
            properties.pop("edge_key", None)
            properties["relation"] = str(row["relation"])
            graph.add_edge(str(row["source_id"]), str(row["target_id"]), key=key, **properties)
        return graph

    def validate_graph(self, expected: nx.MultiDiGraph) -> dict[str, Any]:
        counts = self.counts()
        expected_counts = {
            "nodes": expected.number_of_nodes(),
            "relationships": expected.number_of_edges(),
        }
        if counts != expected_counts:
            raise RuntimeError(
                f"Neo4j count mismatch: expected {expected_counts}, received {counts}"
            )
        queries = self.representative_queries()
        if not queries["sample_requirements"]:
            raise RuntimeError("Neo4j validation found no role requirements")
        if not queries["sample_transitions"]:
            raise RuntimeError("Neo4j validation found no training transitions")
        return {"counts": counts, **queries}


def load_graph_from_neo4j(settings) -> nx.MultiDiGraph:
    store = Neo4jGraphStore.from_settings(settings)
    try:
        graph = store.load_networkx()
    finally:
        store.close()
    if not graph:
        raise RuntimeError("Neo4j Aura graph is empty; run python rebuild_databases.py")
    return graph

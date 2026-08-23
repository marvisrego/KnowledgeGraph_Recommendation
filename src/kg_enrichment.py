"""Knowledge graph enrichment: skill IDF and ISCO-based role attributes.

These derived attributes are used by:
- Transition Effort Score (IDF-weighted skill gap magnitude, transferability)
- Link Prediction (Adamic-Adar, Resource Allocation, IDF-weighted Jaccard)
- ISCO Alignment Validation (ISCO 2-digit proximity)
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx


def compute_skill_idf(G: nx.MultiDiGraph) -> dict[str, float]:
    """Compute IDF for each skill node: log(total_roles / roles_requiring_skill).

    Rare skills get higher IDF; common skills get lower IDF.
    Stored on skill nodes as G.nodes[skill_id]['idf'].
    Returns the mapping {skill_id: idf_value}.
    """
    role_nodes = [
        nid for nid, d in G.nodes(data=True)
        if d.get("type") == "role"
    ]
    total_roles = len(role_nodes)
    if total_roles == 0:
        return {}

    skill_role_count: dict[str, int] = {}
    for role_id in role_nodes:
        for _, target, data in G.out_edges(role_id, data=True):
            if data.get("relation") != "REQUIRES":
                continue
            node_data = G.nodes.get(target)
            if node_data and node_data.get("type") in ("skill", "element"):
                skill_role_count[str(target)] = skill_role_count.get(str(target), 0) + 1

    idf_map: dict[str, float] = {}
    for skill_id, count in skill_role_count.items():
        idf_map[skill_id] = math.log(total_roles / count) if count > 0 else 0.0

    for skill_id, idf_val in idf_map.items():
        if G.has_node(skill_id):
            G.nodes[skill_id]["idf"] = idf_val

    return idf_map


def compute_isco_2digit(G: nx.MultiDiGraph) -> dict[str, str]:
    """Derive 2-digit ISCO code for each ESCO role from its isco_group attribute.

    The isco_group field (e.g. "2511") is truncated to 2 digits ("25").
    Stored as G.nodes[role_id]['isco_2digit'].
    Returns the mapping {role_id: isco_2digit_code}.
    """
    isco_map: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") != "role" or data.get("source") != "esco":
            continue
        isco_raw = str(data.get("isco_group", "")).strip()
        if len(isco_raw) >= 2 and isco_raw[:2].isdigit():
            code_2d = isco_raw[:2]
            isco_map[str(nid)] = code_2d
            G.nodes[nid]["isco_2digit"] = code_2d

    return isco_map


def role_skills(role_id: str, G: nx.MultiDiGraph) -> set[str]:
    """Return the set of skill/element node IDs required by a role."""
    skills: set[str] = set()
    if not G.has_node(role_id):
        return skills
    for _, target, data in G.out_edges(role_id, data=True):
        if data.get("relation") != "REQUIRES":
            continue
        node_data = G.nodes.get(target)
        if node_data and node_data.get("type") in ("skill", "element"):
            skills.add(str(target))
    return skills


def skill_degree(skill_id: str, G: nx.MultiDiGraph) -> int:
    """Count how many roles require this skill (in-degree for REQUIRES relation)."""
    count = 0
    for source, _, data in G.in_edges(skill_id, data=True):
        if data.get("relation") == "REQUIRES":
            count += 1
    return count


def enrich_graph(G: nx.MultiDiGraph, crosswalk_path: Path | None = None) -> nx.MultiDiGraph:
    """Run all enrichments on the graph. Mutates in-place and returns it.

    - Computes skill IDF for all skill/element nodes
    - Computes ISCO 2-digit codes for all ESCO roles
    - Optionally maps ONET SOC codes to ISCO via a crosswalk file
    """
    print("[kg_enrichment] Computing skill IDF …")
    idf_map = compute_skill_idf(G)
    print(f"[kg_enrichment] IDF computed for {len(idf_map)} skills/elements.")

    print("[kg_enrichment] Computing ISCO 2-digit codes …")
    isco_map = compute_isco_2digit(G)
    print(f"[kg_enrichment] ISCO 2-digit assigned to {len(isco_map)} ESCO roles.")

    if crosswalk_path and crosswalk_path.exists():
        mapped = add_onet_isco_codes(G, crosswalk_path)
        print(f"[kg_enrichment] ISCO codes mapped to {mapped} ONET roles via crosswalk.")

    return G


def add_onet_isco_codes(G: nx.MultiDiGraph, crosswalk_path: Path) -> int:
    """Map ONET SOC codes to ISCO-08 using a SOC→ISCO crosswalk CSV.

    Expected CSV columns: SOC_Code (or soc_code), ISCO_Code (or isco_code).
    The SOC code in the CSV may be 6-digit (XX-XXXX) or 8-digit (XX-XXXX.XX).
    ONET node IDs are 8-digit SOC codes like "11-1011.00".

    Returns count of successfully mapped ONET roles.
    """
    import csv

    crosswalk: dict[str, str] = {}
    with open(crosswalk_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            soc = ""
            isco = ""
            for key, val in row.items():
                key_lower = key.strip().lower().replace(" ", "_").replace("-", "_")
                if "soc" in key_lower and not soc:
                    soc = str(val).strip()
                elif "isco" in key_lower and not isco:
                    isco = str(val).strip()
            if soc and isco and len(isco) >= 2:
                crosswalk[soc] = isco
                soc_6 = soc.split(".")[0] if "." in soc else soc
                if soc_6 not in crosswalk:
                    crosswalk[soc_6] = isco

    mapped = 0
    for nid, data in G.nodes(data=True):
        if data.get("type") != "role" or data.get("source") != "onet":
            continue
        node_id = str(nid)
        isco = crosswalk.get(node_id)
        if not isco:
            soc_6 = node_id.split(".")[0] if "." in node_id else node_id
            isco = crosswalk.get(soc_6)
        if isco and len(isco) >= 2:
            G.nodes[nid]["isco_group"] = isco
            G.nodes[nid]["isco_2digit"] = isco[:2]
            mapped += 1

    return mapped

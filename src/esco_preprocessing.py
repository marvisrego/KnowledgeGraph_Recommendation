"""
ESCO preprocessing: loads CSV files from the ESCO directory and returns
unified nodes and edges as pandas DataFrames.

Node schema columns:  id, type, source, title, description, [extra attrs...]
Edge schema columns:  src, dst, relation, source, [extra attrs...]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv(esco_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(esco_dir / name, low_memory=False)


def _normalize_relation_type(raw: str) -> str | None:
    """Map ESCO relationType (plain text OR full URI) to 'essential'/'optional'."""
    rel = str(raw).strip().lower()
    if rel in ("essential", "optional"):
        return rel
    if "relatedessentialskill" in rel:
        return "essential"
    if "relatedoptionalskill" in rel:
        return "optional"
    return None  # unknown — skip


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def _build_occupation_nodes(esco_dir: Path) -> pd.DataFrame:
    occ = _csv(esco_dir, "occupations_en.csv")

    # Concrete occupations only (not occupation groups)
    occ = occ[occ["conceptType"] == "Occupation"].copy()

    nodes = pd.DataFrame(
        {
            "id": occ["conceptUri"],
            "type": "role",
            "source": "esco",
            "title": occ["preferredLabel"],
            "description": occ["definition"].fillna(""),
            "isco_group": occ["iscoGroup"].fillna(""),
            "esco_code": occ["code"].fillna(""),
        }
    )

    # --- Green share enrichment ---
    try:
        gs = _csv(esco_dir, "greenShareOcc_en.csv")[["conceptUri", "greenShare"]]
        nodes = nodes.merge(gs, left_on="id", right_on="conceptUri", how="left").drop(
            columns=["conceptUri"], errors="ignore"
        )
        nodes = nodes.rename(columns={"greenShare": "green_share"})
    except Exception:
        nodes["green_share"] = None

    # --- Research occupation flag ---
    try:
        ro = _csv(esco_dir, "researchOccupationsCollection_en.csv")[["conceptUri"]]
        research_uris = set(ro["conceptUri"])
        nodes["is_research_occupation"] = nodes["id"].isin(research_uris)
    except Exception:
        nodes["is_research_occupation"] = False

    return nodes.reset_index(drop=True)


def _build_skill_nodes(esco_dir: Path) -> pd.DataFrame:
    sk = _csv(esco_dir, "skills_en.csv")

    nodes = pd.DataFrame(
        {
            "id": sk["conceptUri"],
            "type": "skill",
            "source": "esco",
            "title": sk["preferredLabel"],
            "description": sk["definition"].fillna(""),
            "concept_type": sk["conceptType"].fillna(""),
            "skill_type": sk["skillType"].fillna(""),
            "reuse_level": sk["reuseLevel"].fillna(""),
        }
    )

    skill_uris = set(nodes["id"])

    # --- Boolean tag flags from topical collections ---
    tag_files = {
        "is_digital": "digitalSkillsCollection_en.csv",
        "is_digcomp": "digCompSkillsCollection_en.csv",
        "is_green_skill": "greenSkillsCollection_en.csv",
        "is_language_skill": "languageSkillsCollection_en.csv",
        "is_transversal_skill": "transversalSkillsCollection_en.csv",
        "is_research_skill": "researchSkillsCollection_en.csv",
    }
    for flag, fname in tag_files.items():
        try:
            col = _csv(esco_dir, fname)[["conceptUri"]]
            tagged = set(col["conceptUri"])
            nodes[flag] = nodes["id"].isin(tagged)
        except Exception:
            nodes[flag] = False

    return nodes.reset_index(drop=True)


def _build_isco_group_nodes(esco_dir: Path) -> pd.DataFrame:
    ig = _csv(esco_dir, "ISCOGroups_en.csv")
    return pd.DataFrame(
        {
            "id": ig["conceptUri"],
            "type": "isco_group",
            "source": "esco",
            "title": ig["preferredLabel"],
            "description": ig["description"].fillna(""),
            "code": ig["code"].fillna(""),
        }
    ).reset_index(drop=True)


def _build_skill_group_nodes(esco_dir: Path) -> pd.DataFrame:
    sg = _csv(esco_dir, "skillGroups_en.csv")
    return pd.DataFrame(
        {
            "id": sg["conceptUri"],
            "type": "skill_group",
            "source": "esco",
            "title": sg["preferredLabel"],
            "description": sg["description"].fillna(""),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def _build_occ_skill_edges(esco_dir: Path) -> pd.DataFrame:
    """REQUIRES edges: occupation → skill."""
    rel = _csv(esco_dir, "occupationSkillRelations_en.csv")

    rows = []
    for _, row in rel.iterrows():
        req_level = _normalize_relation_type(row["relationType"])
        if req_level is None:
            continue
        rows.append(
            {
                "src": row["occupationUri"],
                "dst": row["skillUri"],
                "relation": "REQUIRES",
                "source": "esco",
                "requirement_level": req_level,
                "raw_relation_type": row["relationType"],
                "skill_type": row.get("skillType", ""),
            }
        )
    return pd.DataFrame(rows)


def _build_isco_hierarchy_edges(esco_dir: Path, occ_uris: set[str]) -> pd.DataFrame:
    """BELONGS_TO edges: occupation → isco_group.
    Also BROADER_THAN/NARROWER_THAN between ISCO groups from ISCOGroups.
    """
    occ = _csv(esco_dir, "occupations_en.csv")
    occ = occ[occ["conceptType"] == "Occupation"]

    belongs = []
    for _, row in occ.iterrows():
        if pd.notna(row.get("iscoGroup")) and str(row["iscoGroup"]).strip():
            belongs.append(
                {
                    "src": row["conceptUri"],
                    "dst": str(row["iscoGroup"]).strip(),
                    "relation": "BELONGS_TO",
                    "source": "esco",
                }
            )

    # ISCO group broader/narrower: use broaderRelationsOccPillar which covers
    # group-level hierarchy too (conceptType == 'ISCOGroup')
    try:
        br = _csv(esco_dir, "broaderRelationsOccPillar_en.csv")
        isco_rows = br[br["conceptType"] == "ISCOGroup"]
        for _, row in isco_rows.iterrows():
            # narrower → broader
            belongs.append(
                {
                    "src": row["conceptUri"],
                    "dst": row["broaderUri"],
                    "relation": "BROADER_THAN",
                    "source": "esco",
                }
            )
            belongs.append(
                {
                    "src": row["broaderUri"],
                    "dst": row["conceptUri"],
                    "relation": "NARROWER_THAN",
                    "source": "esco",
                }
            )
    except Exception:
        pass

    return pd.DataFrame(belongs)


def _build_occ_hierarchy_edges(esco_dir: Path) -> pd.DataFrame:
    """BROADER_THAN/NARROWER_THAN edges between occupation concepts."""
    br = _csv(esco_dir, "broaderRelationsOccPillar_en.csv")
    occ_rows = br[br["conceptType"] == "Occupation"]

    edges = []
    for _, row in occ_rows.iterrows():
        edges.append(
            {
                "src": row["conceptUri"],
                "dst": row["broaderUri"],
                "relation": "BROADER_THAN",
                "source": "esco",
                "pillar": "occupations",
            }
        )
        edges.append(
            {
                "src": row["broaderUri"],
                "dst": row["conceptUri"],
                "relation": "NARROWER_THAN",
                "source": "esco",
                "pillar": "occupations",
            }
        )
    return pd.DataFrame(edges)


def _build_skill_hierarchy_edges(esco_dir: Path) -> pd.DataFrame:
    """BROADER_THAN/NARROWER_THAN edges from skillsHierarchy, broaderRelationsSkillPillar."""
    edges: list[dict] = []

    # skillsHierarchy: level columns contain URI pairs
    try:
        sh = _csv(esco_dir, "skillsHierarchy_en.csv")
        level_cols = [c for c in sh.columns if c.startswith("Level") and "URI" in c]
        for i in range(len(level_cols) - 1):
            parent_col = level_cols[i]
            child_col = level_cols[i + 1]
            pairs = sh[[parent_col, child_col]].dropna()
            for _, row in pairs.iterrows():
                p, c = str(row[parent_col]).strip(), str(row[child_col]).strip()
                if p and c and p != c:
                    edges.append(
                        {"src": p, "dst": c, "relation": "NARROWER_THAN", "source": "esco", "pillar": "skills"}
                    )
                    edges.append(
                        {"src": c, "dst": p, "relation": "BROADER_THAN", "source": "esco", "pillar": "skills"}
                    )
    except Exception:
        pass

    # broaderRelationsSkillPillar
    try:
        br = _csv(esco_dir, "broaderRelationsSkillPillar_en.csv")
        for _, row in br.iterrows():
            edges.append(
                {
                    "src": row["conceptUri"],
                    "dst": row["broaderUri"],
                    "relation": "BROADER_THAN",
                    "source": "esco",
                    "pillar": "skills",
                }
            )
            edges.append(
                {
                    "src": row["broaderUri"],
                    "dst": row["conceptUri"],
                    "relation": "NARROWER_THAN",
                    "source": "esco",
                    "pillar": "skills",
                }
            )
    except Exception:
        pass

    return pd.DataFrame(edges)


def _build_skill_skill_edges(esco_dir: Path) -> pd.DataFrame:
    """RELATED_TO / BROADER_THAN edges between skills from skillSkillRelations."""
    ss = _csv(esco_dir, "skillSkillRelations_en.csv")

    def _rel_label(raw: str) -> str:
        r = str(raw).strip().lower()
        if "broader" in r:
            return "BROADER_THAN"
        if "narrower" in r:
            return "NARROWER_THAN"
        return "RELATED_TO"

    edges = []
    for _, row in ss.iterrows():
        edges.append(
            {
                "src": row["originalSkillUri"],
                "dst": row["relatedSkillUri"],
                "relation": _rel_label(row["relationType"]),
                "source": "esco",
                "pillar": "skills",
            }
        )
    return pd.DataFrame(edges)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_esco(esco_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes_df, edges_df) for all ESCO data.

    nodes_df columns: id, type, source, title, description, [isco_group,
                      esco_code, green_share, is_research_occupation,
                      concept_type, skill_type, reuse_level, is_digital, ...]
    edges_df columns: src, dst, relation, source, [requirement_level,
                      raw_relation_type, skill_type, pillar, ...]
    """
    occ_nodes = _build_occupation_nodes(esco_dir)
    skill_nodes = _build_skill_nodes(esco_dir)
    isco_nodes = _build_isco_group_nodes(esco_dir)
    sg_nodes = _build_skill_group_nodes(esco_dir)

    nodes = pd.concat([occ_nodes, skill_nodes, isco_nodes, sg_nodes], ignore_index=True)

    occ_uris = set(occ_nodes["id"])

    occ_skill_edges = _build_occ_skill_edges(esco_dir)
    isco_edges = _build_isco_hierarchy_edges(esco_dir, occ_uris)
    occ_hier_edges = _build_occ_hierarchy_edges(esco_dir)
    skill_hier_edges = _build_skill_hierarchy_edges(esco_dir)
    skill_skill_edges = _build_skill_skill_edges(esco_dir)

    edges = pd.concat(
        [occ_skill_edges, isco_edges, occ_hier_edges, skill_hier_edges, skill_skill_edges],
        ignore_index=True,
    )

    return nodes, edges

"""
ONET preprocessing: loads Excel files from the ONET directory and returns
unified nodes and edges as pandas DataFrames.

Node schema columns:  id, type, source, title, description, [extra attrs...]
Edge schema columns:  src, dst, relation, source, [extra attrs...]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xl(onet_dir: Path, name: str) -> Path:
    """Return path to an ONET xlsx file (space-separated on-disk names)."""
    return onet_dir / name


def _read_xl(onet_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = _xl(onet_dir, name)
    return pd.read_excel(path, **kwargs)


def _domain_from_element_id(element_id: str) -> str:
    """Derive high-level domain from the ONET Element ID prefix."""
    prefix = str(element_id).split(".")[0]
    mapping = {
        "1": "Tasks",
        "2": "Tools and Technology",
        "3": "Knowledge",
        "4": "Skills",
        "4.A": "Basic Skills",
        "4.B": "Cross-Functional Skills",
        "5": "Abilities",
        "6": "Work Values",
        "7": "Work Styles",
        "1.A": "Generalized Work Activities",
        "1.B": "Detailed Work Activities",
        "1.C": "Organizational Context",
        "2.A": "Tools",
        "2.B": "Technology",
    }
    # Try longest prefix match
    for key in sorted(mapping.keys(), key=len, reverse=True):
        if str(element_id).startswith(key):
            return mapping[key]
    return f"Domain-{prefix}"


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def _build_role_nodes(onet_dir: Path) -> pd.DataFrame:
    """Load occupation data + job zone + education enrichments."""
    occ = _read_xl(onet_dir, "Occupation Data.xlsx")
    occ = occ.rename(columns={"O*NET-SOC Code": "id", "Title": "title", "Description": "description"})
    occ["type"] = "role"
    occ["source"] = "onet"

    # --- Job Zone enrichment ---
    jz = _read_xl(onet_dir, "Job Zones.xlsx")[["O*NET-SOC Code", "Job Zone"]]
    jz = jz.rename(columns={"O*NET-SOC Code": "id"})

    jz_ref = _read_xl(onet_dir, "Job Zone Reference.xlsx")[["Job Zone", "Name"]]
    jz_ref = jz_ref.rename(columns={"Name": "job_zone_title"})

    jz = jz.merge(jz_ref, on="Job Zone", how="left").rename(columns={"Job Zone": "job_zone"})
    occ = occ.merge(jz, on="id", how="left")

    # --- Education enrichment ---
    # Education.xlsx: pick the row with highest Data Value per occupation as
    # the "typical" education level; join category description via Education Categories.
    edu = _read_xl(onet_dir, "Education.xlsx")
    edu = edu[edu["Scale ID"] == "RL"]  # RL = Required Level
    edu = edu.sort_values("Data Value", ascending=False)
    edu_top = edu.groupby("O*NET-SOC Code").first().reset_index()
    edu_top = edu_top[["O*NET-SOC Code", "Category"]]

    edu_cat = _read_xl(onet_dir, "Education Categories.xlsx")
    edu_cat = edu_cat[edu_cat["Scale ID"] == "RL"][["Category", "Category Description"]]
    edu_cat = edu_cat.rename(columns={"Category Description": "typical_education_level"})

    edu_top = edu_top.merge(edu_cat, on="Category", how="left").drop(columns=["Category"])
    edu_top = edu_top.rename(columns={"O*NET-SOC Code": "id"})
    occ = occ.merge(edu_top, on="id", how="left")

    # --- Career interest / RIASEC enrichment ---
    interests_map = _build_interests_map(onet_dir)
    occ = occ.merge(interests_map, on="id", how="left")

    return occ


def _build_interests_map(onet_dir: Path) -> pd.DataFrame:
    """Build occupation → interests list mapping."""
    # Interests Illustrative Occupations links interest Element ID → O*NET-SOC Code
    ill = _read_xl(onet_dir, "Interests Illustrative Occupations.xlsx")
    ill = ill.rename(columns={"O*NET-SOC Code": "id", "Element ID": "interest_element_id"})

    # Career Interest Types gives the interest name
    cit = _read_xl(onet_dir, "Career Interest Types.xlsx")
    cit_map = (
        cit[["Element ID", "Element Name"]]
        .drop_duplicates()
        .rename(columns={"Element ID": "interest_element_id", "Element Name": "interest_name"})
    )

    # Career Interest Type Keywords
    kw = _read_xl(onet_dir, "Career Interest Type Keywords.xlsx")
    kw_map = (
        kw.groupby("Element ID")["Keyword"]
        .apply(list)
        .reset_index()
        .rename(columns={"Element ID": "interest_element_id", "Keyword": "keywords"})
    )

    ill = ill.merge(cit_map, on="interest_element_id", how="left")
    ill = ill.merge(kw_map, on="interest_element_id", how="left")

    interests = (
        ill.groupby("id")
        .apply(
            lambda g: pd.Series(
                {
                    "interests": g["interest_name"].dropna().unique().tolist(),
                    "interests_keywords": sum(
                        (kw for kw in g["keywords"].dropna() if isinstance(kw, list)), []
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return interests


_EXCLUDED_DOMAINS = {"Work Values", "Work Styles"}

def _build_element_nodes(onet_dir: Path) -> pd.DataFrame:
    """Load Content Model Reference as element nodes.

    Work Values and Work Styles are personality/preference dimensions, not
    career-relevant skills — they are excluded to keep the graph clean.
    """
    cm = _read_xl(onet_dir, "Content Model Reference.xlsx")
    cm = cm.rename(
        columns={"Element ID": "id", "Element Name": "title", "Description": "description"}
    )
    cm["type"] = "element"
    cm["source"] = "onet"
    cm["domain"] = cm["id"].apply(_domain_from_element_id)
    cm = cm[~cm["domain"].isin(_EXCLUDED_DOMAINS)].reset_index(drop=True)
    return cm


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def _build_requires_edges(onet_dir: Path) -> pd.DataFrame:
    """Build REQUIRES edges from Knowledge, Abilities, Essential Skills.

    Filters: Scale ID == 'IM', Not Relevant != 'Y', Recommend Suppress != 'Y',
             Data Value >= 3.0.
    Optionally appends LV (level) score to the same edge.
    """
    domain_files = {
        "knowledge": "Knowledge.xlsx",
        "ability": "Abilities.xlsx",
        "skill": "Essential Skills.xlsx",
    }

    all_edges: list[pd.DataFrame] = []

    for domain, filename in domain_files.items():
        df = _read_xl(onet_dir, filename)

        # --- IM (Importance) rows only ---
        im = df[df["Scale ID"] == "IM"].copy()
        im = im[im.get("Not Relevant", pd.Series(dtype=str)).fillna("N") != "Y"]
        im = im[im.get("Recommend Suppress", pd.Series(dtype=str)).fillna("N") != "Y"]
        im = im[im["Data Value"] >= 3.0]

        edges = pd.DataFrame(
            {
                "src": im["O*NET-SOC Code"].values,
                "dst": im["Element ID"].values,
                "relation": "REQUIRES",
                "source": "onet",
                "requirement_level": im["Data Value"].values,
                "scale_id": "IM",
                "domain": domain,
            }
        )

        # --- Optional: merge LV (level) score ---
        lv = df[df["Scale ID"] == "LV"][["O*NET-SOC Code", "Element ID", "Data Value"]].rename(
            columns={"Data Value": "level_score"}
        )
        if not lv.empty:
            edges = edges.merge(
                lv,
                left_on=["src", "dst"],
                right_on=["O*NET-SOC Code", "Element ID"],
                how="left",
            ).drop(columns=["O*NET-SOC Code", "Element ID"], errors="ignore")
            edges["level_scale_id"] = edges["level_score"].apply(
                lambda x: "LV" if pd.notna(x) else None
            )

        all_edges.append(edges)

    return pd.concat(all_edges, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_onet(onet_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes_df, edges_df) for all ONET data.

    nodes_df columns: id, type, source, title, description, [domain, job_zone,
                      job_zone_title, typical_education_level, interests,
                      interests_keywords]
    edges_df columns: src, dst, relation, source, requirement_level, scale_id,
                      domain, [level_score, level_scale_id]
    """
    role_nodes = _build_role_nodes(onet_dir)
    element_nodes = _build_element_nodes(onet_dir)
    nodes = pd.concat([role_nodes, element_nodes], ignore_index=True)

    edges = _build_requires_edges(onet_dir)

    # Keep only edges whose src and dst both exist in nodes
    valid_ids = set(nodes["id"].astype(str))
    edges = edges[
        edges["src"].astype(str).isin(valid_ids) & edges["dst"].astype(str).isin(valid_ids)
    ]

    return nodes, edges

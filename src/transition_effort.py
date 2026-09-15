"""Transition Effort Score — multi-factor difficulty estimate for career switches.

Measures how hard it is to move from one role to another based on:
1. Skill gap magnitude (IDF-weighted missing skills)
2. Domain distance (ISCO structural proximity)
3. Empirical support (do people actually make this transition?)
4. Transferability (IDF-weighted shared skills)

Higher TES = more effort required. Range: [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from src.kg_enrichment import compute_skill_idf
from src.skill_gap import role_requirements
from src.tes_calibration import TESCalibration, default_calibration
from src.transition_policy import is_training_transition

if TYPE_CHECKING:
    from src.transition_embedding import RuntimeTransitionSmoother


@dataclass(frozen=True)
class EffortWeights:
    skill_gap: float = 0.35
    domain: float = 0.15
    empirical: float = 0.25
    transferability: float = 0.25

    def __post_init__(self) -> None:
        total = self.skill_gap + self.domain + self.empirical + self.transferability
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class EffortResult:
    score: float
    band: str
    skill_gap_magnitude: float | None
    domain_distance: float | None
    empirical_support: float
    transferability: float | None
    weights: EffortWeights
    calibration: TESCalibration
    estimated_weeks_min: int = 0
    estimated_weeks_max: int = 0

    def to_dict(self) -> dict:
        return {
            "effort_score": round(self.score, 4),
            "effort_band": self.band,
            "estimated_weeks_min": self.estimated_weeks_min,
            "estimated_weeks_max": self.estimated_weeks_max,
            "estimated_weeks_basis": "heuristic",
            "components": {
                "skill_gap_magnitude": _round_or_none(self.skill_gap_magnitude),
                "domain_distance": _round_or_none(self.domain_distance),
                "empirical_support": round(self.empirical_support, 4),
                "transferability": _round_or_none(self.transferability),
            },
            "calibration": self.calibration.to_dict(),
        }


_BAND_WEEKS: dict[str, tuple[int, int]] = {
    "low": (4, 12),
    "moderate": (12, 36),
    "high": (36, 72),
}


def estimate_upskill_weeks(missing_count: int, band: str) -> tuple[int, int]:
    """Estimate min/max weeks to upskill based on missing skill count.

    Uses 5 weeks/skill midpoint with ±40% range. Falls back to band map when
    missing_count is 0 (skill data absent).
    """
    if missing_count > 0:
        raw = missing_count * 5
        return (max(1, int(raw * 0.6)), max(4, int(raw * 1.4)))
    return _BAND_WEEKS.get(band, (4, 12))


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def effort_band(score: float, calibration: TESCalibration | None = None) -> str:
    """Classify effort score into human-readable band."""
    calibration = calibration or default_calibration()
    if score <= calibration.low_max:
        return "low"
    if score <= calibration.moderate_max:
        return "moderate"
    return "high"


def get_idf_map(G: nx.MultiDiGraph) -> dict[str, float]:
    """Get or compute skill IDF map from graph node attributes."""
    idf_map: dict[str, float] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") in ("skill", "element") and "idf" in data:
            idf_map[str(nid)] = float(data["idf"])
    if not idf_map:
        idf_map = compute_skill_idf(G)
    return idf_map


def _idf_sum(skill_ids: set[str], idf_map: dict[str, float]) -> float:
    """Sum of IDF values for a set of skills. Defaults to 1.0 for unknown skills."""
    return sum(idf_map.get(sid, 1.0) for sid in skill_ids)


def skill_gap_magnitude(
    owned_skills: set[str],
    target_required: set[str],
    idf_map: dict[str, float],
) -> float | None:
    """IDF-weighted fraction of skills the user is missing.

    Returns 0.0 if user has all required skills, 1.0 if they have none.
    Returns ``None`` when target requirements are unavailable. Missing evidence
    must not be converted into maximum difficulty.
    """
    if not target_required:
        return None
    if not owned_skills:
        return None
    missing = target_required - owned_skills
    if not missing:
        return 0.0
    total_idf = _idf_sum(target_required, idf_map)
    if total_idf == 0:
        return 0.0
    return _idf_sum(missing, idf_map) / total_idf


def domain_distance(
    source_id: str,
    target_id: str,
    G: nx.MultiDiGraph,
) -> float | None:
    """ISCO Jaccard distance between two roles.

    Returns 0.0 if same ISCO 2-digit group, 1.0 if completely different.
    Returns ``None`` if ISCO codes are unavailable for either role.
    """
    source_isco = str(
        G.nodes.get(source_id, {}).get("isco_group")
        or G.nodes.get(source_id, {}).get("isco_2digit")
        or ""
    )
    target_isco = str(
        G.nodes.get(target_id, {}).get("isco_group")
        or G.nodes.get(target_id, {}).get("isco_2digit")
        or ""
    )

    if not source_isco or not target_isco:
        return None
    if source_isco == target_isco:
        return 0.0

    # ISCO is hierarchical: sharing 3, 2, or 1 leading digits means the roles
    # are progressively closer. The graph may only have a 2-digit fallback.
    common = 0
    for source_digit, target_digit in zip(source_isco, target_isco):
        if source_digit != target_digit:
            break
        common += 1
    depth = min(4, max(len(source_isco), len(target_isco)))
    return 1.0 - min(common, depth) / depth


def empirical_support(
    source_id: str,
    target_id: str,
    G: nx.MultiDiGraph,
    smoother: "RuntimeTransitionSmoother | None" = None,
) -> float | None:
    """Evidence that people actually make this career transition.

    Returns 0.0 (no evidence) to 1.0 (strong observed evidence).
    """
    if not G.has_node(source_id) or not G.has_node(target_id):
        return 0.0
    best_prob = 0.0
    best_count = 0
    for _, target, data in G.out_edges(source_id, data=True):
        if (
            str(target) == str(target_id)
            and is_training_transition(data)
        ):
            prob = float(data.get("probability", 0.0))
            count = int(data.get("count", 0))
            if prob > best_prob or (prob == best_prob and count > best_count):
                best_prob = prob
                best_count = count

    if best_prob > 0:
        return min(1.0, best_prob * math.log2(1 + best_count))

    if smoother is not None:
        try:
            rankings = smoother.rank(source_id, 50)
            for dest in rankings:
                if str(dest.role_id) == str(target_id):
                    return dest.score * 0.5
        except Exception:
            pass

    return 0.0


def transferability(
    source_id: str,
    target_id: str,
    G: nx.MultiDiGraph,
    idf_map: dict[str, float],
    onet_importance_threshold: float = 3.5,
) -> float | None:
    """IDF-weighted fraction of target skills that the source role also requires.

    High transferability = many rare skills transfer from source to target.
    """
    source_skills, _, _ = role_requirements(
        source_id, G, onet_importance_threshold
    )
    target_skills, _, _ = role_requirements(
        target_id, G, onet_importance_threshold
    )

    if not target_skills:
        return None

    shared = source_skills & target_skills
    if not shared:
        return 0.0

    total_idf = _idf_sum(target_skills, idf_map)
    if total_idf == 0:
        return 0.0

    return _idf_sum(shared, idf_map) / total_idf


def transition_effort_score(
    source_id: str,
    target_id: str,
    owned_skills: set[str],
    G: nx.MultiDiGraph,
    idf_map: dict[str, float] | None = None,
    weights: EffortWeights | None = None,
    smoother: "RuntimeTransitionSmoother | None" = None,
    onet_importance_threshold: float = 3.5,
    calibration: TESCalibration | None = None,
) -> EffortResult:
    """Compute the full Transition Effort Score for a source→target pair.

    Args:
        source_id: Current role node ID
        target_id: Target role node ID
        owned_skills: Set of skill node IDs the user owns
        G: The knowledge graph
        idf_map: Pre-computed IDF map (computed if None)
        weights: Component weights (defaults if None)
        smoother: Optional transition smoother for empirical support
    """
    if idf_map is None:
        idf_map = get_idf_map(G)
    calibration = calibration or default_calibration()
    if weights is None:
        weights = EffortWeights(
            skill_gap=calibration.weights["skill_gap"],
            domain=calibration.weights["domain"],
            empirical=calibration.weights["empirical"],
            transferability=calibration.weights["transferability"],
        )

    target_required, _, _ = role_requirements(
        target_id, G, onet_importance_threshold
    )

    sgm = skill_gap_magnitude(owned_skills, target_required, idf_map)
    dd = domain_distance(source_id, target_id, G)
    es = empirical_support(source_id, target_id, G, smoother)
    tf = transferability(
        source_id, target_id, G, idf_map, onet_importance_threshold
    )

    components = (
        (weights.skill_gap, sgm),
        (weights.domain, dd),
        (weights.empirical, 1.0 - es),
        (weights.transferability, None if tf is None else 1.0 - tf),
    )
    available = [(weight, value) for weight, value in components if value is not None]
    if not available:
        raise ValueError("Transition effort has no available evidence components")
    available_weight = sum(weight for weight, _ in available)
    score = sum(weight * float(value) for weight, value in available) / available_weight

    score = max(0.0, min(1.0, score))
    band = effort_band(score, calibration)

    missing_count = len(target_required - owned_skills) if target_required else 0
    weeks_min, weeks_max = estimate_upskill_weeks(missing_count, band)

    return EffortResult(
        score=score,
        band=band,
        skill_gap_magnitude=sgm,
        domain_distance=dd,
        empirical_support=es,
        transferability=tf,
        weights=weights,
        calibration=calibration,
        estimated_weeks_min=weeks_min,
        estimated_weeks_max=weeks_max,
    )


def rank_roles_by_effort(
    candidates: list[dict],
    source_id: str | None,
    owned_skills: set[str],
    G: nx.MultiDiGraph,
    idf_map: dict[str, float] | None = None,
    weights: EffortWeights | None = None,
    smoother: "RuntimeTransitionSmoother | None" = None,
    onet_importance_threshold: float = 3.5,
    calibration: TESCalibration | None = None,
) -> list[dict]:
    """Annotate candidates with TES and sort by effort (low effort first).

    Candidates without a valid source role get effort=None and sort last.
    """
    if idf_map is None:
        idf_map = get_idf_map(G)
    if weights is None:
        selected = calibration or default_calibration()
        weights = EffortWeights(
            skill_gap=selected.weights["skill_gap"],
            domain=selected.weights["domain"],
            empirical=selected.weights["empirical"],
            transferability=selected.weights["transferability"],
        )

    annotated: list[dict] = []
    for candidate in candidates:
        enriched = dict(candidate)
        target_id = str(candidate["id"])

        if source_id and G.has_node(source_id) and G.has_node(target_id):
            result = transition_effort_score(
                source_id,
                target_id,
                owned_skills,
                G,
                idf_map,
                weights,
                smoother,
                onet_importance_threshold,
                calibration,
            )
            enriched["effort"] = result.to_dict()
        else:
            enriched["effort"] = None

        annotated.append(enriched)

    def _sort_key(c: dict) -> tuple:
        effort = c.get("effort")
        if effort is None:
            return (1, 1.0, "")
        return (0, effort["effort_score"], c.get("id", ""))

    annotated.sort(key=_sort_key)
    return annotated

"""Compare semantic-only and transition/skill-gap ranking on fixed queries.

This evaluator uses the configured embedding and reranking services but skips
response generation and course search.  Its purpose is to measure the local,
graph-structural effect of the two novelty contributions on the same semantic
candidate set.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from src.embeddings_index import load_chroma_collection
from src.graph_build import load_graph
from src.inference_pipeline import (
    augment_candidates_with_transitions,
    filter_candidates_to_graph,
    rerank_candidates,
    retrieve_candidates,
)
from src.karrierewege_preprocessing import write_json_atomic
from src.skill_gap import (
    compute_skill_gap,
    rank_roles_by_gap,
    resolve_current_role,
    resolve_user_skills,
)


CASES = (
    {
        "id": "professional_data",
        "query": "I am a data analyst with four years of experience. I know data mining and data models and want to move into data science.",
        "current_role": "data analyst",
        "skills": ["data mining", "data models"],
    },
    {
        "id": "professional_software",
        "query": "I am a software developer experienced in Python and computer programming. I want to progress toward artificial intelligence engineering.",
        "current_role": "software developer",
        "skills": ["Python", "computer programming"],
    },
    {
        "id": "professional_project",
        "query": "I am a project manager skilled in strategic planning and business strategy concepts. I want a technology transformation role.",
        "current_role": "project manager",
        "skills": ["strategic planning", "business strategy concepts"],
    },
    {
        "id": "student_computing",
        "query": "I recently completed computer science and know Python and data mining. I want to enter analytics or AI.",
        "current_role": "",
        "skills": ["Python", "data mining"],
    },
)


def _role_row(candidate: dict, G) -> dict:
    role_id = candidate["id"]
    gap = candidate.get("skill_gap") or {}
    transition = candidate.get("transition") or {}
    return {
        "id": role_id,
        "title": G.nodes[role_id].get("title", role_id),
        "source": G.nodes[role_id].get("source", ""),
        "semantic_score": float(
            candidate.get("rerank_score", candidate.get("score", 0.0)) or 0.0
        ),
        "accessibility": gap.get("accessibility"),
        "gap_evidence": gap.get("gap_evidence"),
        "transition_probability": transition.get("probability"),
        "transition_count": transition.get("count"),
    }


def _accessibility_inversions(rows: list[dict]) -> int:
    values = [row["accessibility"] for row in rows if row["accessibility"] is not None]
    return sum(
        left < right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


def _top_accessibility_mean(rows: list[dict], limit: int = 3) -> float | None:
    values = [
        row["accessibility"]
        for row in rows[:limit]
        if row["accessibility"] is not None
    ]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the career-ranking ablation suite.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/karrierewege/ranking_ablation.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("artifacts/karrierewege/ranking_ablation.csv"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(root)
    graph = load_graph(settings.graph_path)
    collection = load_chroma_collection(settings)
    results = []
    csv_rows = []

    for case in CASES:
        print(f"[ranking_ablation] {case['id']}")
        current_role_id = resolve_current_role(case["current_role"], graph)
        skill_evidence = resolve_user_skills(case["skills"], [case["query"]], graph)

        retrieved = filter_candidates_to_graph(
            retrieve_candidates(case["query"], collection, settings), graph
        )
        if current_role_id:
            retrieved = [
                candidate for candidate in retrieved if candidate["id"] != current_role_id
            ]
        semantic_shortlist = rerank_candidates(case["query"], retrieved, settings)
        baseline_rows = []
        for candidate in semantic_shortlist:
            annotated = dict(candidate)
            annotated["skill_gap"] = compute_skill_gap(
                skill_evidence["skill_ids"], candidate["id"], graph
            )
            baseline_rows.append(_role_row(annotated, graph))

        augmented = augment_candidates_with_transitions(
            retrieved,
            current_role_id,
            graph,
            settings.transition_candidate_limit,
        )
        augmented_shortlist = rerank_candidates(case["query"], augmented, settings)
        transition_only_rows = []
        for candidate in augmented_shortlist:
            annotated = dict(candidate)
            annotated["skill_gap"] = compute_skill_gap(
                skill_evidence["skill_ids"], candidate["id"], graph
            )
            transition_only_rows.append(_role_row(annotated, graph))
        full_ranked = rank_roles_by_gap(
            skill_evidence["skill_ids"],
            augmented_shortlist,
            graph,
            current_role_id=current_role_id,
        )
        full_rows = [_role_row(candidate, graph) for candidate in full_ranked]

        baseline_first = baseline_rows[0]["accessibility"] if baseline_rows else None
        full_first = full_rows[0]["accessibility"] if full_rows else None
        comparable_accessibility = [
            row["accessibility"] for row in full_rows if row["accessibility"] is not None
        ]
        monotonic = all(
            left >= right
            for left, right in zip(comparable_accessibility, comparable_accessibility[1:])
        )
        empirical_shortlist = sum(
            row["transition_probability"] is not None for row in full_rows
        )
        retrieved_ids = {candidate["id"] for candidate in retrieved}
        augmented_ids = {candidate["id"] for candidate in augmented}
        baseline_inversions = _accessibility_inversions(baseline_rows)
        transition_only_inversions = _accessibility_inversions(transition_only_rows)
        full_inversions = _accessibility_inversions(full_rows)

        case_result = {
            "id": case["id"],
            "query": case["query"],
            "resolved_current_role_id": current_role_id,
            "matched_skill_count": len(skill_evidence["skill_ids"]),
            "semantic_only": baseline_rows,
            "transition_only": transition_only_rows,
            "transition_and_gap": full_rows,
            "metrics": {
                "baseline_first_accessibility": baseline_first,
                "full_first_accessibility": full_first,
                "first_role_accessibility_gain": (
                    full_first - baseline_first
                    if full_first is not None and baseline_first is not None
                    else None
                ),
                "accessibility_order_nonincreasing": monotonic,
                "transition_candidates_added": len(augmented_ids - retrieved_ids),
                "transition_supported_shortlist_roles": empirical_shortlist,
                "baseline_accessibility_inversions": baseline_inversions,
                "transition_only_accessibility_inversions": transition_only_inversions,
                "full_accessibility_inversions": full_inversions,
                "accessibility_inversions_removed": transition_only_inversions - full_inversions,
                "baseline_top3_accessibility_mean": _top_accessibility_mean(baseline_rows),
                "full_top3_accessibility_mean": _top_accessibility_mean(full_rows),
            },
        }
        results.append(case_result)
        csv_rows.append({"case": case["id"], **case_result["metrics"]})

    gains = [
        row["metrics"]["first_role_accessibility_gain"]
        for row in results
        if row["metrics"]["first_role_accessibility_gain"] is not None
    ]
    payload = {
        "method": {
            "baseline": "semantic retrieval plus Cohere reranking",
            "transition_only": "empirical candidate expansion plus Cohere reranking",
            "full": "transition candidate expansion plus Cohere reranking plus skill-gap ordering",
            "cases": len(results),
        },
        "summary": {
            "mean_first_role_accessibility_gain": sum(gains) / len(gains) if gains else None,
            "monotonic_cases": sum(
                row["metrics"]["accessibility_order_nonincreasing"] for row in results
            ),
            "cases_with_transition_supported_shortlist": sum(
                row["metrics"]["transition_supported_shortlist_roles"] > 0 for row in results
            ),
            "accessibility_inversions_removed": sum(
                row["metrics"]["accessibility_inversions_removed"] for row in results
            ),
        },
        "results": results,
    }

    output = args.output if args.output.is_absolute() else root / args.output
    csv_output = args.csv_output if args.csv_output.is_absolute() else root / args.csv_output
    write_json_atomic(payload, output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[ranking_ablation] Wrote {output}")
    print(f"[ranking_ablation] Wrote {csv_output}")


if __name__ == "__main__":
    main()

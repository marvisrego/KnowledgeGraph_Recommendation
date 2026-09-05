"""ISCO Ground-Truth Alignment Validation (P1).

Validates and optimises the ONET-ESCO embedding alignment threshold
using ISCO-08 codes as silver-standard ground truth.

Sweeps similarity thresholds from 0.40 to 0.80 and computes:
- Precision: fraction of SIMILAR_TO pairs that share the same ISCO 2-digit code
- Recall: fraction of all possible same-ISCO pairs that are captured at this threshold
- F1: harmonic mean

Produces a precision-recall curve and identifies the optimal threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from src.graph_build import load_graph


def compute_all_pair_similarities(G, settings: Settings) -> list[tuple[str, str, float]]:
    """Compute embedding cosine similarity for all ONET-ESCO role pairs.

    Uses the existing ChromaDB vectors (text-embedding-3-large).
    Returns [(onet_id, esco_id, similarity), ...] for all computable pairs.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_collection("roles_collection")

    # Get all embeddings
    onet_roles = [(str(nid), d) for nid, d in G.nodes(data=True)
                  if d.get("type") == "role" and d.get("source") == "onet"]
    esco_roles = [(str(nid), d) for nid, d in G.nodes(data=True)
                  if d.get("type") == "role" and d.get("source") == "esco"]

    print(f"  Loading embeddings for {len(onet_roles)} ONET + {len(esco_roles)} ESCO roles...")

    # Fetch all embeddings
    all_ids = [nid for nid, _ in onet_roles] + [nid for nid, _ in esco_roles]
    batch_size = 500
    embeddings: dict[str, np.ndarray] = {}

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        result = collection.get(ids=batch_ids, include=["embeddings"])
        if result["embeddings"] is not None:
            for j, eid in enumerate(result["ids"]):
                vec = np.array(result["embeddings"][j], dtype=np.float64)
                if np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
                    embeddings[eid] = vec / np.linalg.norm(vec)

    print(f"  Loaded {len(embeddings)} valid embedding vectors")

    onet_ids = [nid for nid, _ in onet_roles if nid in embeddings]
    esco_ids = [nid for nid, _ in esco_roles if nid in embeddings]

    if not onet_ids or not esco_ids:
        print("  ERROR: No embeddings available for similarity computation")
        return []

    # Build matrices
    onet_matrix = np.array([embeddings[nid] for nid in onet_ids])
    esco_matrix = np.array([embeddings[nid] for nid in esco_ids])

    print(f"  Computing {len(onet_ids)} x {len(esco_ids)} = {len(onet_ids)*len(esco_ids):,} similarities...")

    # Compute all pairwise cosine similarities (dot product since vectors are normalized)
    sim_matrix = onet_matrix @ esco_matrix.T

    # Extract all pairs above minimum threshold (0.35 to avoid huge output)
    pairs = []
    min_threshold = 0.35
    for i in range(len(onet_ids)):
        for j in range(len(esco_ids)):
            sim = float(sim_matrix[i, j])
            if sim >= min_threshold:
                pairs.append((onet_ids[i], esco_ids[j], sim))

    print(f"  Found {len(pairs):,} pairs above {min_threshold}")
    return pairs


def sweep_thresholds(
    pairs: list[tuple[str, str, float]],
    G,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Sweep alignment thresholds and compute precision/recall/F1 at each.

    Ground truth: same ISCO 2-digit code = positive pair.
    """
    if thresholds is None:
        thresholds = [round(0.40 + i * 0.02, 2) for i in range(21)]  # 0.40 to 0.80

    # Build ground truth: all same-ISCO ONET-ESCO pairs
    onet_isco: dict[str, str] = {}
    esco_isco: dict[str, str] = {}
    for nid, d in G.nodes(data=True):
        if d.get("type") != "role":
            continue
        isco = d.get("isco_2digit", "")
        if not isco:
            continue
        if d.get("source") == "onet":
            onet_isco[str(nid)] = isco
        elif d.get("source") == "esco":
            esco_isco[str(nid)] = isco

    # Total possible same-ISCO pairs (ground truth positives)
    from collections import Counter
    onet_by_isco = Counter(onet_isco.values())
    esco_by_isco = Counter(esco_isco.values())
    total_positive_pairs = sum(
        onet_by_isco[code] * esco_by_isco.get(code, 0)
        for code in onet_by_isco
    )
    print(f"  Total same-ISCO ground truth pairs: {total_positive_pairs:,}")

    results = []
    for threshold in thresholds:
        # Pairs above this threshold
        above = [(o, e, s) for o, e, s in pairs if s >= threshold]
        total_predicted = len(above)

        if total_predicted == 0:
            results.append({
                "threshold": threshold,
                "predicted_pairs": 0,
                "true_positives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            })
            continue

        # True positives: predicted pairs that share ISCO code
        true_positives = sum(
            1 for o, e, _ in above
            if onet_isco.get(o) and esco_isco.get(e) and onet_isco[o] == esco_isco[e]
        )

        precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        recall = true_positives / total_positive_pairs if total_positive_pairs > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": threshold,
            "predicted_pairs": total_predicted,
            "true_positives": true_positives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    return results


def find_optimal_threshold(results: list[dict]) -> dict:
    """Find threshold that maximizes F1 score."""
    best = max(results, key=lambda r: r["f1"])
    return best


def run_alignment_analysis(settings: Settings | None = None, output_dir: Path | None = None) -> dict:
    """Full alignment validation pipeline."""
    if settings is None:
        settings = Settings.from_env()
    if output_dir is None:
        output_dir = Path("artifacts/alignment_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ISCO GROUND-TRUTH ALIGNMENT VALIDATION (P1)")
    print("=" * 70)

    G = load_graph(settings.graph_path)

    # Step 1: Compute all similarities
    print("\n[1/3] Computing ONET-ESCO embedding similarities...")
    pairs = compute_all_pair_similarities(G, settings)

    if not pairs:
        print("ERROR: No pairs computed. Cannot proceed.")
        return {}

    # Step 2: Sweep thresholds
    print("\n[2/3] Sweeping thresholds 0.40 - 0.80...")
    results = sweep_thresholds(pairs, G)

    print(f"\n  {'Threshold':>10} | {'Pairs':>8} | {'TP':>6} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for r in results:
        print(f"  {r['threshold']:>10.2f} | {r['predicted_pairs']:>8} | {r['true_positives']:>6} | {r['precision']:>10.4f} | {r['recall']:>8.4f} | {r['f1']:>8.4f}")

    # Step 3: Find optimal
    print("\n[3/3] Finding optimal threshold...")
    optimal = find_optimal_threshold(results)
    current_threshold = settings.similarity_threshold

    print(f"\n  Optimal threshold (max F1): {optimal['threshold']}")
    print(f"  Optimal F1: {optimal['f1']}")
    print(f"  Optimal precision: {optimal['precision']}")
    print(f"  Optimal recall: {optimal['recall']}")
    print(f"  Optimal pairs: {optimal['predicted_pairs']}")
    print(f"\n  Current threshold: {current_threshold}")
    current_result = next((r for r in results if abs(r["threshold"] - current_threshold) < 0.01), None)
    if current_result:
        print(f"  Current F1: {current_result['f1']}")
        print(f"  Current precision: {current_result['precision']}")
        print(f"  Current recall: {current_result['recall']}")

    # Save report
    report = {
        "method": "ISCO-08 2-digit code agreement as silver-standard ground truth",
        "embedding_model": settings.embed_model,
        "onet_roles": sum(1 for _, d in G.nodes(data=True) if d.get("source") == "onet" and d.get("type") == "role"),
        "esco_roles": sum(1 for _, d in G.nodes(data=True) if d.get("source") == "esco" and d.get("type") == "role"),
        "total_pairs_computed": len(pairs),
        "current_threshold": current_threshold,
        "optimal_threshold": optimal["threshold"],
        "optimal_f1": optimal["f1"],
        "sweep_results": results,
    }

    report_path = output_dir / "alignment_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # Save P-R curve data for plotting
    pr_curve_path = output_dir / "precision_recall_curve.csv"
    with open(pr_curve_path, "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,f1,predicted_pairs,true_positives\n")
        for r in results:
            f.write(f"{r['threshold']},{r['precision']},{r['recall']},{r['f1']},{r['predicted_pairs']},{r['true_positives']}\n")
    print(f"  P-R curve data: {pr_curve_path}")

    print("\n" + "=" * 70)
    print("ALIGNMENT VALIDATION COMPLETE")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_alignment_analysis()

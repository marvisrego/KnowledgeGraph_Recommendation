"""Train and evaluate link prediction for career transition edges.

Pipeline:
1. Load graph + embeddings
2. Build training data from train-split TRANSITIONS_TO edges
3. 5-fold cross-validation (report AUC)
4. Train final model on full training set
5. Generate prediction map for evaluation
6. Evaluate vs. held-out validation and test splits
7. Compare with direct baseline and embedding-smoothed results
8. Save model + metrics
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.transition_metrics import evaluate_prediction_map
from src.graph_build import load_graph
from src.transition_policy import is_training_transition
from src.karrierewege_preprocessing import (
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
)
from src.kg_enrichment import compute_skill_idf
from src.link_prediction import (
    FEATURE_NAMES,
    build_neighbour_index,
    build_training_data,
    build_transition_index,
    extract_pair_features,
    link_prediction_map,
    save_model,
    train_link_predictor,
)
from src.text_normalization import normalize_label


def load_embeddings_from_chroma(chroma_dir: Path) -> dict[str, np.ndarray]:
    """Load role embeddings from ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection("roles_collection")
    except Exception:
        try:
            collection = client.get_collection("career_roles")
        except Exception:
            print("  WARNING: No embedding collection found in ChromaDB")
            return {}

    all_data = collection.get(include=["embeddings"])
    embeddings: dict[str, np.ndarray] = {}
    if all_data["embeddings"] is not None:
        for i, emb_id in enumerate(all_data["ids"]):
            emb = all_data["embeddings"][i]
            if emb is not None:
                vec = np.array(emb, dtype=np.float64)
                if np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
                    embeddings[emb_id] = vec
    return embeddings


def evaluate_link_prediction(
    settings: Settings | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Full link prediction evaluation pipeline."""
    if settings is None:
        settings = Settings.from_env()
    if output_dir is None:
        output_dir = Path("artifacts/link_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LINK PREDICTION EVALUATION")
    print("=" * 70)

    # Step 1: Load graph and embeddings
    print("\n[1/7] Loading graph and embeddings...")
    G = load_graph(settings.graph_path)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    embeddings = load_embeddings_from_chroma(settings.chroma_dir)
    print(f"  Embeddings: {len(embeddings)} vectors loaded")

    idf_map = {
        str(nid): float(d["idf"])
        for nid, d in G.nodes(data=True)
        if "idf" in d
    }
    if not idf_map:
        idf_map = compute_skill_idf(G)
    print(f"  IDF: {len(idf_map)} skills")

    # Step 2: Build training data
    print("\n[2/7] Building training data...")
    t0 = time.time()
    X, y, pairs = build_training_data(G, idf_map, embeddings, neg_ratio=5, seed=42)
    t_build = time.time() - t0
    print(f"  Samples: {len(y)} ({y.sum()} positive, {len(y) - y.sum()} negative)")
    print(f"  Features: {X.shape[1]} ({', '.join(FEATURE_NAMES)})")
    print(f"  Build time: {t_build:.1f}s")

    # Step 3: Cross-validation
    print("\n[3/7] 5-fold cross-validation...")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    cv_aps = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = train_link_predictor(X_train, y_train)
        preds = model.predict(X_val)
        auc = roc_auc_score(y_val, preds)
        ap = average_precision_score(y_val, preds)
        cv_aucs.append(auc)
        cv_aps.append(ap)
        print(f"  Fold {fold}: AUC={auc:.4f}, AP={ap:.4f}")

    mean_auc = np.mean(cv_aucs)
    mean_ap = np.mean(cv_aps)
    print(f"  Mean AUC: {mean_auc:.4f} (+/- {np.std(cv_aucs):.4f})")
    print(f"  Mean AP: {mean_ap:.4f} (+/- {np.std(cv_aps):.4f})")

    # Step 4: Train final model
    print("\n[4/7] Training final model on full training set...")
    t0 = time.time()
    final_model = train_link_predictor(X, y)
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    print("  Feature importance (gain):")
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.1f}")

    # Step 5: Generate prediction map (only for held-out source roles)
    print("\n[5/7] Generating prediction map for held-out source roles...")
    split_files = discover_split_files(settings.karrierewege_dir)
    title_index = build_esco_title_index(G)

    # Collect all source roles from held-out splits to predict for
    all_held_out_sources: set[str] = set()
    for split_name in ("validation", "test"):
        if split_name not in split_files:
            continue
        held_out = aggregate_split_transitions(
            split_files[split_name], split_name, title_index,
            chunk_size=settings.transition_chunk_size,
            strict_mapping=True,
            max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
        )
        for (src, _) in held_out.pair_counts.keys():
            all_held_out_sources.add(src)

    # Map normalized titles back to role IDs
    title_to_id = {}
    for nid, d in G.nodes(data=True):
        if d.get("type") == "role" and d.get("source") == "esco":
            norm_title = normalize_label(str(d.get("title", "")))
            if norm_title:
                title_to_id[norm_title] = str(nid)

    source_ids = [title_to_id[t] for t in all_held_out_sources if t in title_to_id]
    # Also include all roles that have transition data (for full coverage)
    transition_sources = [
        str(nid) for nid, d in G.nodes(data=True)
        if d.get("type") == "role" and d.get("source") == "esco"
        and any(is_training_transition(ed) for _, _, ed in G.out_edges(nid, data=True))
    ]
    source_ids = list(set(source_ids + transition_sources))
    print(f"  Predicting for {len(source_ids)} source roles from held-out data...")

    t0 = time.time()
    predictions = link_prediction_map(
        final_model, G, idf_map, embeddings, top_k=50, source_role_ids=source_ids
    )
    t_pred = time.time() - t0
    print(f"  Predictions generated for {len(predictions)} source roles")
    print(f"  Prediction time: {t_pred:.1f}s")

    # Step 6: Evaluate vs held-out splits
    print("\n[6/7] Evaluating against held-out splits...")

    results = {}
    for split_name in ("validation", "test"):
        if split_name not in split_files:
            print(f"  WARNING: {split_name} split not found, skipping.")
            continue
        held_out = aggregate_split_transitions(
            split_files[split_name], split_name, title_index,
            chunk_size=settings.transition_chunk_size,
            strict_mapping=True,
            max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
        )
        metrics = evaluate_prediction_map(predictions, held_out)
        results[split_name] = metrics
        print(f"\n  {split_name.upper()} results:")
        for key, val in sorted(metrics.items()):
            if isinstance(val, float):
                print(f"    {key}: {val:.6f}")
            else:
                print(f"    {key}: {val}")

    # Step 7: Save model and report
    print("\n[7/7] Saving model and metrics...")
    model_path = output_dir / "link_predictor.pkl"
    save_model(final_model, model_path)
    print(f"  Model saved: {model_path}")

    report = {
        "cv_auc_mean": float(mean_auc),
        "cv_auc_std": float(np.std(cv_aucs)),
        "cv_ap_mean": float(mean_ap),
        "cv_ap_std": float(np.std(cv_aps)),
        "training_samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum()),
        "feature_importance": {
            name: float(imp) for name, imp in zip(FEATURE_NAMES, importance)
        },
        "source_roles_predicted": len(predictions),
        "build_time_s": round(t_build, 1),
        "train_time_s": round(t_train, 1),
        "prediction_time_s": round(t_pred, 1),
    }
    for split_name, metrics in results.items():
        report[f"{split_name}_metrics"] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    return report


if __name__ == "__main__":
    evaluate_link_prediction()

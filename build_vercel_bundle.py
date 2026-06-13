from __future__ import annotations

import json
from pathlib import Path

from career_kg_chat import Settings, build_graph_bundle, load_graph_bundle, save_graph_bundle


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    settings = Settings.from_env(root_dir)

    dataset_ready = settings.onet_dir.exists() and settings.postings_path.exists() and settings.jobs_dir.exists() and settings.companies_dir.exists() and settings.mappings_dir.exists()
    if not dataset_ready:
        cached_bundle = load_graph_bundle(settings)
        if cached_bundle is None:
            raise RuntimeError(
                "Raw dataset paths are unavailable and no cached graph bundle exists. Commit artifacts/career_kg_bundle.pkl or provide the dataset during the build step."
            )
        print(json.dumps(cached_bundle.stats, indent=2))
        return

    bundle = build_graph_bundle(settings, postings_limit_override=settings.postings_limit)
    save_graph_bundle(settings, bundle)
    print(json.dumps(bundle.stats, indent=2))


if __name__ == "__main__":
    main()
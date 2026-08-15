from __future__ import annotations

import json
from pathlib import Path

from career_kg_chat import Settings, prepare_runtime, runtime_stats, write_dual_graph_view


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    settings = Settings.from_env(root_dir)

    runtime = prepare_runtime(settings, rebuild=True, postings_limit_override=settings.postings_limit)
    write_dual_graph_view(settings, runtime)
    print(json.dumps(runtime_stats(runtime), indent=2))


if __name__ == "__main__":
    main()
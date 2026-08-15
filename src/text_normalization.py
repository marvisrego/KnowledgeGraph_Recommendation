"""Shared runtime text normalization."""

from __future__ import annotations

import re
import unicodedata


def normalize_label(value: object) -> str:
    """Return a conservative comparison key for an occupation or skill label."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

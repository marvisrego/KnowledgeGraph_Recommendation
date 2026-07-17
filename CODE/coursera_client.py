from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from functools import lru_cache
from html import unescape
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_LIMIT = 5
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
SUPPORTED_PATH_ROOTS = {
    "learn": "course",
    "specializations": "specialization",
    "professional-certificates": "professional_certificate",
}
COURSE_LINK_PATTERN = re.compile(r'href="(/(?:learn|specializations|professional-certificates)/([^"?#/]+))"', re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    return WHITESPACE_PATTERN.sub(" ", (value or "").strip())


def normalize_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def slug_to_title(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part)


def canonicalize_coursera_url(raw_url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlparse(raw_url.strip())
    scheme = parsed.scheme or "https"
    host = parsed.netloc.lower() or "www.coursera.org"
    if not host.endswith("coursera.org"):
        raise ValueError(f"Unsupported Coursera host: {raw_url}")

    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    match = re.match(r"^/(learn|specializations|professional-certificates)/([^/?#]+)", path)
    if not match:
        raise ValueError(f"Unsupported Coursera path: {raw_url}")

    path_root, slug = match.group(1), match.group(2)
    canonical_path = f"/{path_root}/{slug}"
    canonical_url = urllib.parse.urlunparse((scheme, host, canonical_path, "", "", ""))
    return canonical_url, SUPPORTED_PATH_ROOTS[path_root], slug


def course_id_from_slug(content_type: str, slug: str) -> str:
    return f"coursera:{content_type}:{slug}"


def build_search_queries(role_title: str | None, missing_skills: list[str], fallback_query: str) -> list[str]:
    queries: list[str] = []
    role_title = normalize_text(role_title)
    cleaned_skills = [normalize_text(skill) for skill in missing_skills if normalize_text(skill)]
    if role_title and cleaned_skills:
        queries.append(f"{role_title} {' '.join(cleaned_skills[:2])}")
    if role_title:
        queries.append(role_title)
    for skill in cleaned_skills[:3]:
        queries.append(skill)
    if fallback_query:
        queries.append(normalize_text(fallback_query))

    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        key = normalize_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


@lru_cache(maxsize=128)
def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Coursera request failed for {url}: {detail[:220]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Coursera request failed for {url}: {exc}") from exc


@lru_cache(maxsize=128)
def fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    raw_text = fetch_text(url, timeout=timeout)
    return json.loads(raw_text)


def fetch_first_element(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any] | None:
    payload = fetch_json(url, timeout=timeout)
    elements = payload.get("elements", [])
    if isinstance(elements, list) and elements:
        first = elements[0]
        if isinstance(first, dict):
            return first
    return None


def extract_search_scope(html_text: str) -> str:
    for marker in ['id="searchResults"', 'aria-label="Search Results"']:
        index = html_text.find(marker)
        if index >= 0:
            return html_text[index:]
    return html_text


def extract_search_candidates(html_text: str, *, limit: int) -> list[dict[str, str]]:
    scoped_html = extract_search_scope(html_text)
    candidates: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for match in COURSE_LINK_PATTERN.finditer(scoped_html):
        path = match.group(1)
        slug = match.group(2)
        path_root = path.split("/")[1]
        content_type = SUPPORTED_PATH_ROOTS.get(path_root)
        if not content_type or path in seen_paths:
            continue
        seen_paths.add(path)
        candidates.append(
            {
                "path": path,
                "slug": slug,
                "content_type": content_type,
                "canonical_url": f"https://www.coursera.org{path}",
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def fetch_partner_names(partner_ids: list[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> list[str]:
    cleaned_ids = [str(partner_id).strip() for partner_id in partner_ids if str(partner_id).strip()]
    if not cleaned_ids:
        return []
    url = "https://api.coursera.org/api/partners.v1?ids=" + urllib.parse.quote(",".join(cleaned_ids), safe=",")
    payload = fetch_json(url, timeout=timeout)
    names = []
    for element in payload.get("elements", []):
        if isinstance(element, dict):
            name = normalize_text(str(element.get("name") or ""))
            if name and name not in names:
                names.append(name)
    return names


def fetch_instructor_names(instructor_ids: list[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> list[str]:
    cleaned_ids = [str(instructor_id).strip() for instructor_id in instructor_ids if str(instructor_id).strip()]
    if not cleaned_ids:
        return []
    url = "https://api.coursera.org/api/instructors.v1?ids=" + urllib.parse.quote(",".join(cleaned_ids), safe=",")
    payload = fetch_json(url, timeout=timeout)
    names = []
    for element in payload.get("elements", []):
        if isinstance(element, dict):
            full_name = normalize_text(str(element.get("fullName") or ""))
            if full_name and full_name not in names:
                names.append(full_name)
    return names


def shorten(value: str | None, max_length: int = 420) -> str:
    text = normalize_text(unescape(value or ""))
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def hydrate_course_candidate(candidate: dict[str, str], *, search_query: str, timeout: int) -> dict[str, Any] | None:
    slug = candidate["slug"]
    canonical_url = candidate["canonical_url"]
    content_type = candidate["content_type"]

    if content_type == "course":
        course_row = fetch_first_element(
            f"https://api.coursera.org/api/courses.v1?q=slug&slug={urllib.parse.quote(slug)}",
            timeout=timeout,
        ) or {}
        detail_row = fetch_first_element(
            f"https://api.coursera.org/api/onDemandCourses.v1?q=slug&slug={urllib.parse.quote(slug)}",
            timeout=timeout,
        ) or {}
        if not course_row and not detail_row:
            return None
        partner_names = fetch_partner_names(detail_row.get("partnerIds", []), timeout=timeout)
        instructor_names = fetch_instructor_names(detail_row.get("instructorIds", [])[:4], timeout=timeout)
        return {
            "course_id": course_id_from_slug(content_type, slug),
            "platform": "coursera",
            "content_type": content_type,
            "title": normalize_text(str(course_row.get("name") or detail_row.get("name") or slug_to_title(slug))),
            "canonical_url": canonical_url,
            "slug": slug,
            "description": shorten(str(detail_row.get("description") or "")),
            "tagline": "",
            "partner_names": partner_names,
            "instructor_names": instructor_names,
            "estimated_workload": normalize_text(str(detail_row.get("estimatedWorkload") or "")),
            "language_primary": ", ".join(detail_row.get("primaryLanguageCodes", [])[:2]),
            "subtitle_language_count": len(detail_row.get("subtitleLanguageCodes", [])),
            "shareable_certificate": bool(detail_row.get("isVerificationEnabled")),
            "search_query": search_query,
        }

    if content_type == "specialization":
        detail_row = fetch_first_element(
            f"https://api.coursera.org/api/onDemandSpecializations.v1?q=slug&slug={urllib.parse.quote(slug)}",
            timeout=timeout,
        ) or {}
        if not detail_row:
            return None
        return {
            "course_id": course_id_from_slug(content_type, slug),
            "platform": "coursera",
            "content_type": content_type,
            "title": normalize_text(str(detail_row.get("name") or slug_to_title(slug))),
            "canonical_url": canonical_url,
            "slug": slug,
            "description": shorten(str(detail_row.get("description") or "")),
            "tagline": normalize_text(str(detail_row.get("tagline") or "")),
            "partner_names": [],
            "instructor_names": [],
            "estimated_workload": "",
            "language_primary": "",
            "subtitle_language_count": 0,
            "shareable_certificate": True,
            "search_query": search_query,
        }

    if content_type == "professional_certificate":
        return {
            "course_id": course_id_from_slug(content_type, slug),
            "platform": "coursera",
            "content_type": content_type,
            "title": slug_to_title(slug),
            "canonical_url": canonical_url,
            "slug": slug,
            "description": "",
            "tagline": "",
            "partner_names": [],
            "instructor_names": [],
            "estimated_workload": "",
            "language_primary": "",
            "subtitle_language_count": 0,
            "shareable_certificate": True,
            "search_query": search_query,
        }

    return None


def search_course_recommendations(query: str, *, limit: int = DEFAULT_LIMIT, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> list[dict[str, Any]]:
    search_url = "https://www.coursera.org/search?query=" + urllib.parse.quote(query)
    html_text = fetch_text(search_url, timeout=timeout)
    candidates = extract_search_candidates(html_text, limit=max(limit * 3, limit))
    recommendations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for position, candidate in enumerate(candidates, start=1):
        recommendation = hydrate_course_candidate(candidate, search_query=query, timeout=timeout)
        if not recommendation:
            continue
        recommendation["rank"] = position
        recommendation["match_label"] = query
        url = recommendation["canonical_url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        recommendations.append(recommendation)
        if len(recommendations) >= limit:
            break

    return recommendations


def search_course_recommendations_multi(
    queries: list[str],
    *,
    limit: int = DEFAULT_LIMIT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for query in queries:
        for recommendation in search_course_recommendations(query, limit=limit, timeout=timeout):
            url = str(recommendation.get("canonical_url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            recommendations.append(recommendation)
            if len(recommendations) >= limit:
                return recommendations
    return recommendations
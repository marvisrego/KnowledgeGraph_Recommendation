from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from coursera_client import DEFAULT_HEADERS, canonicalize_coursera_url, hydrate_course_candidate


DISCOVERY_URL = "https://accounts.coursera.org/.well-known/openid-configuration"
DEFAULT_COURSE_URL = "https://www.coursera.org/learn/machine-learning"
DEFAULT_ENDPOINTS = [
    ("api_courses_v1_by_slug", "https://api.coursera.org/api/courses.v1?q=slug&slug={slug}"),
    ("api_on_demand_courses_v1_by_slug", "https://api.coursera.org/api/onDemandCourses.v1?q=slug&slug={slug}"),
    ("www_courses_v1_by_slug", "https://www.coursera.org/api/courses.v1?q=slug&slug={slug}"),
    ("www_on_demand_courses_v1_by_slug", "https://www.coursera.org/api/onDemandCourses.v1?q=slug&slug={slug}"),
]
COURSE_FIELDS_TO_CHECK = [
    "title",
    "content_type",
    "canonical_url",
    "description",
    "partner_names",
    "estimated_workload",
    "language_primary",
    "subtitle_language_count",
    "shareable_certificate",
    "tagline",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Coursera OAuth and likely catalog endpoints, then compare them with public course-page detail extraction."
    )
    parser.add_argument("--client-id", help="Coursera OAuth client ID. Overrides env-file values.")
    parser.add_argument("--client-secret", help="Coursera OAuth client secret. Overrides env-file values.")
    parser.add_argument("--env-file", default=".env", help="Primary env file to read before falling back to process env or the template file.")
    parser.add_argument(
        "--template-env-file",
        default=".env.example",
        help="Secondary env file used only if the client credentials are not found in the primary env file.",
    )
    parser.add_argument("--course-url", default=DEFAULT_COURSE_URL, help="Coursera course, specialization, or certificate URL to probe.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--scope", default="", help="Optional OAuth scope string to include in token requests.")
    parser.add_argument(
        "--response-chars",
        type=int,
        default=400,
        help="Maximum number of response characters to keep for each probe result.",
    )
    parser.add_argument("--output-json", help="Optional file path to also write the report as JSON.")
    return parser.parse_args(argv)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    session.headers["Accept"] = "application/json, text/plain;q=0.9, text/html;q=0.8"
    return session


def resolve_secret(
    cli_value: str | None,
    process_keys: list[str],
    env_values: dict[str, str],
    template_values: dict[str, str],
    *,
    label: str,
    env_path: Path,
    template_path: Path,
) -> tuple[str | None, str]:
    if cli_value:
        return cli_value, "cli"
    for key in process_keys:
        value = os.environ.get(key)
        if value:
            return value, f"process_env:{key}"
    for key in process_keys:
        value = env_values.get(key)
        if value:
            return value, f"env_file:{env_path.name}:{key}"
    for key in process_keys:
        value = template_values.get(key)
        if value:
            return value, f"template_env_file:{template_path.name}:{key}"
    return None, f"missing:{label}"


def truncate_text(value: str, max_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    return str(value)


def fetch_discovery(session: requests.Session, timeout: int) -> dict[str, Any]:
    response = session.get(DISCOVERY_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return {
        "issuer": payload.get("issuer"),
        "authorization_endpoint": payload.get("authorization_endpoint"),
        "token_endpoint": payload.get("token_endpoint"),
        "userinfo_endpoint": payload.get("userinfo_endpoint"),
        "grant_types_supported": payload.get("grant_types_supported", []),
        "response_types_supported": payload.get("response_types_supported", []),
    }


def try_token_request(
    session: requests.Session,
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    *,
    scope: str,
    timeout: int,
    response_chars: int,
) -> tuple[list[dict[str, Any]], str | None]:
    token_results: list[dict[str, Any]] = []
    access_token: str | None = None

    request_variants = [
        (
            "basic_auth_client_credentials",
            {
                "auth": (client_id, client_secret),
                "data": {"grant_type": "client_credentials", **({"scope": scope} if scope else {})},
            },
        ),
        (
            "body_client_credentials",
            {
                "data": {
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    **({"scope": scope} if scope else {}),
                }
            },
        ),
    ]

    for mode, kwargs in request_variants:
        try:
            response = session.post(token_endpoint, timeout=timeout, **kwargs)
            content_type = response.headers.get("content-type", "")
            payload: dict[str, Any] | None = None
            if "json" in content_type.lower():
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
            body_preview = truncate_text(response.text, response_chars)
            result = {
                "mode": mode,
                "ok": response.ok,
                "status_code": response.status_code,
                "content_type": content_type,
                "response_excerpt": body_preview,
            }
            if payload is not None:
                result["response_json_keys"] = sorted(payload.keys())
                if response.ok and isinstance(payload.get("access_token"), str):
                    access_token = payload["access_token"]
                    result["access_token_preview"] = mask_secret(access_token)
                    result["token_type"] = payload.get("token_type")
            token_results.append(result)
        except requests.RequestException as exc:
            token_results.append(
                {
                    "mode": mode,
                    "ok": False,
                    "status_code": None,
                    "content_type": None,
                    "response_excerpt": str(exc),
                }
            )
    return token_results, access_token


def probe_course_endpoints(
    session: requests.Session,
    slug: str,
    access_token: str | None,
    *,
    timeout: int,
    response_chars: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    for name, template in DEFAULT_ENDPOINTS:
        url = template.format(slug=slug)
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            content_type = response.headers.get("content-type", "")
            payload: Any = None
            payload_summary: dict[str, Any] = {}
            if "json" in content_type.lower():
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
            if isinstance(payload, dict):
                payload_summary["top_level_keys"] = sorted(payload.keys())
                elements = payload.get("elements")
                if isinstance(elements, list):
                    payload_summary["element_count"] = len(elements)
                    if elements:
                        first_element = elements[0]
                        if isinstance(first_element, dict):
                            payload_summary["first_element_keys"] = sorted(first_element.keys())
            result = {
                "endpoint_name": name,
                "url": url,
                "authenticated": bool(access_token),
                "ok": response.ok,
                "status_code": response.status_code,
                "content_type": content_type,
                "response_excerpt": truncate_text(response.text, response_chars),
                **payload_summary,
            }
            results.append(result)
        except requests.RequestException as exc:
            results.append(
                {
                    "endpoint_name": name,
                    "url": url,
                    "authenticated": bool(access_token),
                    "ok": False,
                    "status_code": None,
                    "content_type": None,
                    "response_excerpt": str(exc),
                }
            )
    return results


def probe_public_course_page(session: requests.Session, course_url: str, *, timeout: int) -> dict[str, Any]:
    canonical_url, content_type, slug = canonicalize_coursera_url(course_url)
    course = hydrate_course_candidate(
        {
            "slug": slug,
            "canonical_url": canonical_url,
            "content_type": content_type,
        },
        search_query=slug,
        timeout=timeout,
    )
    if not course:
        return {"ok": False, "error": "No structured Coursera detail could be loaded for this URL."}
    field_availability = {field: bool(course.get(field)) for field in COURSE_FIELDS_TO_CHECK}
    selected_course = {
        field: course.get(field, "") for field in COURSE_FIELDS_TO_CHECK
    }
    selected_course["slug"] = slug
    selected_course["content_type"] = content_type

    return {
        "ok": True,
        "status_code": 200,
        "selected_course_fields": selected_course,
        "field_availability": field_availability,
        "counts": {
            "partners": len(course.get("partner_names", [])),
            "instructors": len(course.get("instructor_names", [])),
        },
        "samples": {
            "partners": course.get("partner_names", [])[:5],
            "instructors": course.get("instructor_names", [])[:5],
        },
    }


def build_manual_auth_url(authorization_endpoint: str, client_id: str) -> str:
    params = {
        "response_type": "token",
        "client_id": client_id,
        "redirect_uri": "https://example.com/callback",
    }
    return f"{authorization_endpoint}?{urlencode(params)}"


def build_summary(report: dict[str, Any]) -> dict[str, Any]:
    token_results = report.get("token_tests", [])
    endpoint_results = report.get("course_endpoint_tests", [])
    public_probe = report.get("public_course_probe", {})
    can_get_token = any(test.get("ok") and test.get("access_token_preview") for test in token_results)
    successful_endpoints = [result["endpoint_name"] for result in endpoint_results if result.get("ok")]
    public_endpoints = [
        result["endpoint_name"]
        for result in endpoint_results
        if result.get("ok") and not result.get("authenticated")
    ]
    authenticated_endpoints = [
        result["endpoint_name"]
        for result in endpoint_results
        if result.get("ok") and result.get("authenticated")
    ]
    public_fields_ok = public_probe.get("field_availability", {})
    available_public_fields = sorted(field for field, present in public_fields_ok.items() if present)
    missing_public_fields = sorted(field for field, present in public_fields_ok.items() if not present)

    if public_endpoints and not can_get_token:
        recommendation = (
            "The tested course endpoints already returned data without OAuth, so these keys were not required for the probed detail path. "
            "Use this report to confirm whether Coursera gave you a separate authenticated catalog API or whether public endpoints are sufficient."
        )
    elif not can_get_token:
        recommendation = (
            "Authenticated course API access was not proven. Use this report to confirm the exact Coursera grant, scope, and catalog endpoint with Coursera support."
        )
    else:
        recommendation = "Authenticated token exchange succeeded. Validate the successful endpoint list before wiring live recommendations into the app."

    if not successful_endpoints and public_probe.get("ok"):
        recommendation += " Public Coursera detail endpoints still return useful metadata if you need a fallback."

    return {
        "can_get_oauth_token": can_get_token,
        "successful_course_endpoints": successful_endpoints,
        "public_course_endpoints": public_endpoints,
        "authenticated_course_endpoints": authenticated_endpoints,
        "can_extract_public_course_details": bool(public_probe.get("ok")),
        "available_public_fields": available_public_fields,
        "missing_public_fields": missing_public_fields,
        "recommended_next_step": recommendation,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root_dir = Path(__file__).resolve().parent
    env_path = (root_dir / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    template_path = (
        (root_dir / args.template_env_file).resolve()
        if not Path(args.template_env_file).is_absolute()
        else Path(args.template_env_file)
    )
    env_values = load_env_file(env_path)
    template_values = load_env_file(template_path)

    client_id, client_id_source = resolve_secret(
        args.client_id,
        ["COURSERA_CLIENT_ID", "coursera_key"],
        env_values,
        template_values,
        label="client_id",
        env_path=env_path,
        template_path=template_path,
    )
    client_secret, client_secret_source = resolve_secret(
        args.client_secret,
        ["COURSERA_CLIENT_SECRET", "coursera_secret"],
        env_values,
        template_values,
        label="client_secret",
        env_path=env_path,
        template_path=template_path,
    )

    report: dict[str, Any] = {
        "course_url": args.course_url,
        "credential_sources": {
            "client_id": client_id_source,
            "client_secret": client_secret_source,
        },
        "credential_previews": {
            "client_id": mask_secret(client_id),
            "client_secret": mask_secret(client_secret),
        },
    }

    session = build_session()

    try:
        discovery = fetch_discovery(session, args.timeout)
    except requests.RequestException as exc:
        report["oauth_discovery"] = {"ok": False, "error": str(exc)}
        report["summary"] = {
            "can_get_oauth_token": False,
            "successful_course_endpoints": [],
            "can_extract_public_course_details": False,
            "available_public_fields": [],
            "missing_public_fields": COURSE_FIELDS_TO_CHECK,
            "recommended_next_step": "OAuth discovery failed before any course check could run.",
        }
        print(json.dumps(safe_json(report), indent=2))
        return 1

    report["oauth_discovery"] = {"ok": True, **discovery}
    manual_auth_url = build_manual_auth_url(discovery["authorization_endpoint"], client_id or "missing-client-id")
    report["manual_authorization_url"] = manual_auth_url

    access_token: str | None = None
    if client_id and client_secret:
        token_results, access_token = try_token_request(
            session,
            discovery["token_endpoint"],
            client_id,
            client_secret,
            scope=args.scope,
            timeout=args.timeout,
            response_chars=args.response_chars,
        )
    else:
        token_results = [
            {
                "mode": "skipped",
                "ok": False,
                "status_code": None,
                "content_type": None,
                "response_excerpt": "Client credentials were not found. Pass --client-id/--client-secret or populate .env.",
            }
        ]
    report["token_tests"] = token_results

    try:
        _, _, slug = canonicalize_coursera_url(args.course_url)
    except ValueError as exc:
        report["course_endpoint_tests"] = []
        report["public_course_probe"] = {"ok": False, "error": str(exc)}
        report["summary"] = build_summary(report)
        print(json.dumps(safe_json(report), indent=2))
        return 1

    report["course_endpoint_tests"] = probe_course_endpoints(
        session,
        slug,
        access_token,
        timeout=args.timeout,
        response_chars=args.response_chars,
    )

    try:
        report["public_course_probe"] = probe_public_course_page(session, args.course_url, timeout=args.timeout)
    except (requests.RequestException, ValueError) as exc:
        report["public_course_probe"] = {"ok": False, "error": str(exc)}

    report["summary"] = build_summary(report)
    rendered = json.dumps(safe_json(report), indent=2)
    print(rendered)

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = root_dir / output_path
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
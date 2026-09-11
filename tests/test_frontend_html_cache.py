"""Prevent stale frontend HTML after same-size Vercel deployments."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flask import send_from_directory

import career_kg_web


class FrontendHtmlCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / "public_react"
        self.dist.mkdir()
        self.index = self.dist / "index.html"
        self.old_html = b'<script type="module" src="/assets/old.js"></script>'
        self.new_html = b'<script type="module" src="/assets/new.js"></script>'
        self.assertEqual(len(self.old_html), len(self.new_html))
        self._write_html(self.old_html)
        with patch.object(career_kg_web, "__file__", str(self.root / "career_kg_web.py")), \
                patch.object(career_kg_web.Settings, "from_env"):
            self.app = career_kg_web.create_app()
        self.app.testing = True
        self.client = self.app.test_client()

    def _write_html(self, content):
        self.index.write_bytes(content)
        os.utime(self.index, (1540000000, 1540000000))

    def _previous_cache_headers(self):
        # Match the metadata validators a browser cached before the fix.
        with self.app.test_request_context("/"):
            response = send_from_directory(str(self.dist), "index.html")
            try:
                return {
                    "If-None-Match": response.headers["ETag"],
                    "If-Modified-Since": response.headers["Last-Modified"],
                }
            finally:
                response.close()

    def test_changed_same_size_html_never_returns_stale_304(self):
        previous_headers = self._previous_cache_headers()
        self._write_html(self.new_html)
        # Demonstrate the original validator collision under Vercel timestamps.
        self.assertEqual(previous_headers, self._previous_cache_headers())
        for url in ("/", "/index.html"):
            for headers in ({}, previous_headers, {"If-None-Match": previous_headers["If-None-Match"]},
                            {"If-Modified-Since": previous_headers["If-Modified-Since"]}):
                with self.subTest(url=url, headers=headers), self.client.get(url, headers=headers) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.data, self.new_html)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertNotIn("ETag", response.headers)

    def test_spa_fallback_uses_the_same_html_policy(self):
        headers = self._previous_cache_headers()
        self._write_html(self.new_html)
        with self.app.test_request_context("/career-paths", headers=headers):
            # Exercise the existing fallback directly without changing Flask's
            # static-route precedence or adding any new routes.
            response = self.app.view_functions["spa_catch_all"]("career-paths")
            try:
                response.direct_passthrough = False
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_data(), self.new_html)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            finally:
                response.close()

    def test_static_asset_conditional_caching_is_unchanged(self):
        assets = self.dist / "assets"
        assets.mkdir()
        (assets / "new.js").write_text("console.log('loaded')", encoding="utf-8")
        with self.client.get("/assets/new.js") as response:
            self.assertEqual(response.status_code, 200)
            etag = response.headers["ETag"]
            self.assertNotEqual(response.headers["Cache-Control"], "no-store")
        with self.client.get("/assets/new.js", headers={"If-None-Match": etag}) as response:
            self.assertEqual(response.status_code, 304)

    def test_head_and_missing_routes_keep_existing_semantics(self):
        with self.client.options("/index.html") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("GET", response.headers["Allow"])
            self.assertNotEqual(response.headers.get("Cache-Control"), "no-store")
        for url in ("/", "/index.html"):
            with self.client.head(url) as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b"")
                self.assertEqual(response.headers["Cache-Control"], "no-store")
        for url in ("/assets/missing.js", "/api/missing"):
            with self.client.get(url) as response:
                self.assertEqual(response.status_code, 404)
        self.index.unlink()
        self.dist.rmdir()
        with self.client.get("/") as response:
            self.assertEqual(response.status_code, 503)
            self.assertIn("Frontend build is missing", response.json["error"])


if __name__ == "__main__":
    unittest.main()

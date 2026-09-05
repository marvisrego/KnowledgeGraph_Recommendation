import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.embeddings_index import embed_texts


class EmbeddingRetryTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            embed_model_api_key="not-a-real-key",
            embed_model_endpoint="https://example.invalid",
            embed_model="test-model",
            embed_batch_size=100,
        )

    @patch("src.embeddings_index.time.sleep")
    @patch("src.embeddings_index._post_json")
    def test_retries_transient_timeout(self, post_json, sleep):
        post_json.side_effect = [
            RuntimeError("[embeddings_index] HTTP 408"),
            {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
        ]

        result = embed_texts(["role"], self.settings)

        self.assertEqual(result, [[1.0, 0.0]])
        self.assertEqual(post_json.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("src.embeddings_index.time.sleep")
    @patch("src.embeddings_index._post_json")
    def test_does_not_retry_invalid_request(self, post_json, sleep):
        post_json.side_effect = RuntimeError("[embeddings_index] HTTP 400")

        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            embed_texts(["role"], self.settings)

        self.assertEqual(post_json.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

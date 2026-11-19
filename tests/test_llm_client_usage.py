import os
import unittest
from unittest.mock import patch

from src.utils.llm_client import (
    OpenAICompatibleClient,
    create_private_client,
    default_max_tokens_field,
)


class OpenAICompatibleClientUsageTests(unittest.TestCase):
    def make_client(self):
        return OpenAICompatibleClient(
            base_url="http://127.0.0.1:9999/v1",
            model_name="test-model",
            api_key=None,
            max_retries=0,
        )

    def test_records_chat_completions_usage(self):
        client = self.make_client()
        client._record_reported_usage(
            {"usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}}
        )
        stats = client.performance_stats()
        self.assertEqual(stats["responses_with_usage"], 1)
        self.assertEqual(stats["reported_prompt_tokens"], 120)
        self.assertEqual(stats["reported_completion_tokens"], 30)
        self.assertEqual(stats["reported_total_tokens"], 150)

    def test_records_responses_style_usage_and_ignores_missing_usage(self):
        client = self.make_client()
        client._record_reported_usage({"usage": {"input_tokens": 20, "output_tokens": 7}})
        client._record_reported_usage({"choices": []})
        stats = client.performance_stats()
        self.assertEqual(stats["responses_with_usage"], 1)
        self.assertEqual(stats["reported_prompt_tokens"], 20)
        self.assertEqual(stats["reported_completion_tokens"], 7)
        self.assertEqual(stats["reported_total_tokens"], 27)

    def test_gpt_55_and_gpt_6_use_completion_tokens_by_default(self):
        self.assertEqual(
            default_max_tokens_field("gpt-5.5"), "max_completion_tokens"
        )
        self.assertEqual(
            default_max_tokens_field("gpt-6-astra"), "max_completion_tokens"
        )

    def test_openai_base_uses_standard_key_and_omits_temperature(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "your key here"},
            clear=True,
        ):
            client = create_private_client(
                {
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-5.6-terra",
                    "api_key_env": "PRIVATE_API_KEY",
                }
            )
        self.assertEqual(client.endpoint, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(client.max_tokens_field, "max_completion_tokens")
        self.assertFalse(client.include_temperature)
        self.assertEqual(client.session.headers["Authorization"], "Bearer your key here")

    def test_openai_payload_uses_only_supported_default_parameters(self):
        client = OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            model_name="gpt-5.6-terra",
            max_tokens_field="max_completion_tokens",
            include_temperature=False,
            max_retries=0,
        )
        with patch.object(client, "_post_json", return_value={"choices": [{"message": {"content": "ok"}}]}) as post:
            self.assertEqual(client.generate("hello"), "ok")
        payload = post.call_args.args[0]
        self.assertEqual(payload["max_completion_tokens"], 2000)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)


if __name__ == "__main__":
    unittest.main()

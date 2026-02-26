"""Tests for rlm.utils.token_utils — token counting and context-limit lookups."""

import sys
from unittest.mock import MagicMock, patch

from rlm.utils.token_utils import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
    _count_tokens_tiktoken,
    count_tokens,
    get_context_limit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(*contents: str) -> list[dict]:
    """Return a list of user messages with the given string contents."""
    return [{"role": "user", "content": c} for c in contents]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_context_limit_is_128k(self):
        assert DEFAULT_CONTEXT_LIMIT == 128_000

    def test_chars_per_token_estimate(self):
        assert CHARS_PER_TOKEN_ESTIMATE == 4

    def test_model_context_limits_is_dict_of_ints(self):
        assert isinstance(MODEL_CONTEXT_LIMITS, dict)
        for key, value in MODEL_CONTEXT_LIMITS.items():
            assert isinstance(key, str), f"key {key!r} is not a str"
            assert isinstance(value, int), f"value for {key!r} is not an int"
            assert value > 0, f"context limit for {key!r} must be positive"


# ---------------------------------------------------------------------------
# get_context_limit
# ---------------------------------------------------------------------------


class TestGetContextLimit:
    # --- empty / sentinel inputs ---

    def test_empty_string_returns_default(self):
        assert get_context_limit("") == DEFAULT_CONTEXT_LIMIT

    def test_none_like_falsy_string_returns_default(self):
        # The implementation checks `if not model_name` — an empty string is falsy
        assert get_context_limit("") == DEFAULT_CONTEXT_LIMIT

    def test_unknown_sentinel_returns_default(self):
        assert get_context_limit("unknown") == DEFAULT_CONTEXT_LIMIT

    def test_completely_unknown_model_returns_default(self):
        assert get_context_limit("totally-made-up-model-xyz") == DEFAULT_CONTEXT_LIMIT

    # --- exact-key matches ---

    def test_exact_key_gpt4o(self):
        assert get_context_limit("gpt-4o") == 128_000

    def test_exact_key_gpt4(self):
        assert get_context_limit("gpt-4") == 8_192

    def test_exact_key_gpt4_32k(self):
        assert get_context_limit("gpt-4-32k") == 32_768

    def test_exact_key_claude_3_opus(self):
        assert get_context_limit("claude-3-opus") == 200_000

    def test_exact_key_gemini_1_5_pro(self):
        assert get_context_limit("gemini-1.5-pro") == 1_000_000

    def test_exact_key_o1(self):
        assert get_context_limit("o1") == 200_000

    def test_exact_key_gpt5(self):
        assert get_context_limit("gpt-5") == 272_000

    def test_exact_key_sonnet_alias(self):
        assert get_context_limit("sonnet") == 200_000

    def test_exact_key_haiku_alias(self):
        assert get_context_limit("haiku") == 200_000

    def test_exact_key_opus_alias(self):
        assert get_context_limit("opus") == 200_000

    # --- substring / prefix matches (longest wins) ---

    def test_substring_match_openai_prefixed_gpt4o(self):
        # "gpt-4o" is contained in "@openai/gpt-4o"
        assert get_context_limit("@openai/gpt-4o") == 128_000

    def test_substring_match_gpt4o_mini_beats_gpt4o(self):
        # Both "gpt-4o" and "gpt-4o-mini" match; "gpt-4o-mini" is longer → wins
        assert get_context_limit("provider/gpt-4o-mini-latest") == 128_000

    def test_substring_match_gpt4_turbo_beats_gpt4(self):
        # "gpt-4-turbo" (11 chars) > "gpt-4" (5 chars)
        assert get_context_limit("openai/gpt-4-turbo-2024") == 128_000

    def test_substring_match_gpt4_32k_beats_gpt4(self):
        # "gpt-4-32k" (9) > "gpt-4" (5)
        assert get_context_limit("my-deployment-gpt-4-32k") == 32_768

    def test_substring_match_gpt5_nano_beats_gpt5(self):
        # "gpt-5-nano" (10) > "gpt-5" (5)
        assert get_context_limit("provider/gpt-5-nano") == 272_000

    def test_substring_match_claude_35_sonnet(self):
        assert get_context_limit("anthropic/claude-3-5-sonnet-20241022") == 200_000

    def test_substring_match_gemini_flash(self):
        assert get_context_limit("google/gemini-2.0-flash-exp") == 1_000_000

    def test_substring_match_o1_mini_beats_o1(self):
        # "o1-mini" (7) > "o1" (2); limit differs: o1-mini=128k, o1=200k
        assert get_context_limit("provider/o1-mini") == 128_000

    def test_substring_match_o1_preview_beats_o1(self):
        assert get_context_limit("o1-preview-2024") == 128_000

    def test_substring_match_qwen3_max_beats_qwen3(self):
        # "qwen3-max" (9) > "qwen3" (5)
        assert get_context_limit("api/qwen3-max-v2") == 256_000

    def test_substring_match_kimi_k2_thinking_beats_kimi_k2(self):
        # "kimi-k2-thinking" (16) > "kimi-k2" (7)
        assert get_context_limit("kimi-k2-thinking-latest") == 256_000

    # --- no false-positive substring: "o1" must not match "o10" unless intended ---

    def test_no_match_for_unrelated_model(self):
        assert get_context_limit("llama-3-70b") == DEFAULT_CONTEXT_LIMIT

    # --- return type ---

    def test_return_type_is_int(self):
        assert isinstance(get_context_limit("gpt-4o"), int)
        assert isinstance(get_context_limit("unknown-model-xyz"), int)


# ---------------------------------------------------------------------------
# _count_tokens_tiktoken  (internal helper)
# ---------------------------------------------------------------------------


class TestCountTokensTiktoken:
    """Test the private tiktoken helper with both mocked-available and mocked-missing paths."""

    def _make_mock_enc(self, token_length: int = 1) -> MagicMock:
        """Return a mock encoder whose encode() returns a list of length token_length per call."""
        enc = MagicMock()
        enc.encode = MagicMock(side_effect=lambda text: [0] * len(text))
        return enc

    def test_returns_none_when_tiktoken_not_installed(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            result = _count_tokens_tiktoken(_make_messages("hello"), "gpt-4o")
        assert result is None

    def test_counts_string_content(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        # encode returns one token per character for determinism
        mock_enc.encode = MagicMock(side_effect=lambda text: list(range(len(text))))
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            messages = [{"role": "user", "content": "hi"}]  # "hi" → 2 tokens
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        # 3 (overhead) + 2 (chars in "hi") = 5
        assert result == 5

    def test_falls_back_to_cl100k_when_model_unknown(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda text: list(range(len(text))))
        # encoding_for_model raises, get_encoding succeeds
        mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown model")
        mock_tiktoken.get_encoding.return_value = mock_enc

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            messages = [{"role": "user", "content": "abc"}]  # 3 chars → 3 tokens
            result = _count_tokens_tiktoken(messages, "mystery-model")

        assert result == 3 + 3  # 3 overhead + 3 content tokens

    def test_returns_none_when_both_encodings_fail(self):
        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.side_effect = Exception("boom")
        mock_tiktoken.get_encoding.side_effect = Exception("also boom")

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(_make_messages("test"), "gpt-4o")

        assert result is None

    def test_counts_list_content_with_text_parts(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda text: list(range(len(text))))
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ab"},  # 2 tokens
                    {"type": "image_url", "url": "…"},  # non-text → skipped
                ],
            }
        ]

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        # 3 (overhead) + 2 (text part "ab") = 5
        assert result == 5

    def test_counts_non_string_content_via_str_coercion(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda text: list(range(len(text))))
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        # content is an integer — should be str()-ified
        messages = [{"role": "user", "content": 42}]  # str(42) == "42" → 2 chars
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        assert result == 3 + 2  # overhead + len("42")

    def test_adds_name_token_when_name_present(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])  # no content tokens
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        messages = [{"role": "user", "content": "", "name": "alice"}]
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        # 3 (overhead) + 0 (content) + 1 (name) = 4
        assert result == 4

    def test_none_content_skips_encoding(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        messages = [{"role": "user", "content": None}]
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        # None should be skipped entirely: only overhead counts
        assert result == 3

    def test_empty_string_content_skips_encoding(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        messages = [{"role": "user", "content": ""}]
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        # encode("") → [] → 0 tokens; total = 3 overhead
        assert result == 3

    def test_multiple_messages_accumulate(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        # 1 token per character
        mock_enc.encode = MagicMock(side_effect=lambda text: list(range(len(text))))
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        messages = [
            {"role": "user", "content": "ab"},  # 3 overhead + 2 content
            {"role": "assistant", "content": "cde"},  # 3 overhead + 3 content
        ]
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = _count_tokens_tiktoken(messages, "gpt-4o")

        assert result == (3 + 2) + (3 + 3)  # = 11


# ---------------------------------------------------------------------------
# count_tokens  (public API)
# ---------------------------------------------------------------------------


class TestCountTokens:
    # --- empty input ---

    def test_empty_message_list_returns_zero(self):
        assert count_tokens([], "gpt-4o") == 0

    # --- "unknown" model always uses char-based fallback ---

    def test_unknown_model_uses_char_estimate(self):
        # 8 chars / 4 = 2 tokens (exact division)
        messages = [{"role": "user", "content": "abcdefgh"}]
        result = count_tokens(messages, "unknown")
        assert result == 2

    def test_empty_model_name_uses_char_estimate(self):
        messages = [{"role": "user", "content": "abcd"}]
        result = count_tokens(messages, "")
        assert result == 1

    # --- fallback arithmetic (tiktoken absent) ---

    def test_char_fallback_exact_division(self):
        # Force tiktoken to be absent for this test
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [{"role": "user", "content": "abcd"}]  # 4 chars / 4 = 1
            result = count_tokens(messages, "gpt-4o")
        assert result == 1

    def test_char_fallback_ceiling_division(self):
        # 5 chars / 4 → ceil = 2
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [{"role": "user", "content": "abcde"}]
            result = count_tokens(messages, "gpt-4o")
        assert result == 2

    def test_char_fallback_single_char(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [{"role": "user", "content": "x"}]
            result = count_tokens(messages, "gpt-4o")
        assert result == 1

    def test_char_fallback_accumulates_multiple_messages(self):
        # 4 + 4 = 8 chars; 8 / 4 = 2
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [
                {"role": "user", "content": "abcd"},
                {"role": "assistant", "content": "efgh"},
            ]
            result = count_tokens(messages, "gpt-4o")
        assert result == 2

    def test_char_fallback_missing_content_key_treated_as_empty(self):
        # A message dict with no "content" key at all
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [{"role": "user"}]
            result = count_tokens(messages, "gpt-4o")
        assert result == 0

    def test_char_fallback_none_content_treated_as_empty(self):
        # content=None → "" via `or ""`
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [{"role": "user", "content": None}]
            result = count_tokens(messages, "gpt-4o")
        assert result == 0

    def test_char_fallback_list_content_stringified(self):
        # list content → str([...]) length
        with patch.dict(sys.modules, {"tiktoken": None}):
            content_list = [{"type": "text", "text": "hi"}]
            messages = [{"role": "user", "content": content_list}]
            expected_chars = len(str(content_list))
            expected_tokens = (
                expected_chars + CHARS_PER_TOKEN_ESTIMATE - 1
            ) // CHARS_PER_TOKEN_ESTIMATE
            result = count_tokens(messages, "gpt-4o")
        assert result == expected_tokens

    # --- tiktoken-backed path ---

    def test_uses_tiktoken_when_available(self):
        mock_tiktoken = MagicMock()
        mock_enc = MagicMock()
        # encode returns exactly 10 tokens regardless of text
        mock_enc.encode = MagicMock(return_value=list(range(10)))
        mock_tiktoken.encoding_for_model.return_value = mock_enc

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            messages = [{"role": "user", "content": "anything"}]
            result = count_tokens(messages, "gpt-4o")

        # 3 overhead + 10 content = 13
        assert result == 13
        mock_enc.encode.assert_called_once()

    def test_falls_back_to_chars_when_tiktoken_returns_none(self):
        # Simulate _count_tokens_tiktoken returning None (both encoders fail)
        with patch("rlm.utils.token_utils._count_tokens_tiktoken", return_value=None):
            messages = [{"role": "user", "content": "abcd"}]  # 4 chars → 1 token
            result = count_tokens(messages, "gpt-4o")
        assert result == 1

    # --- return type ---

    def test_return_type_is_int(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            result = count_tokens([{"role": "user", "content": "test"}], "gpt-4o")
        assert isinstance(result, int)

    def test_return_is_non_negative(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            assert count_tokens([], "gpt-4o") >= 0
            assert count_tokens([{"role": "user", "content": ""}], "gpt-4o") >= 0

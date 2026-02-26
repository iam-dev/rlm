"""Reusable mock LM for testing."""

from collections import defaultdict

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class MockLM(BaseLM):
    """Mock LM with controllable response sequences and usage tracking.

    By default, echoes prompts. Call set_responses() for scripted behavior.
    """

    def __init__(self, model_name: str = "mock-model"):
        super().__init__(model_name=model_name)
        self._responses: list[str] = []
        self._call_index: int = 0
        self.call_log: list[str] = []

        # Usage tracking (matches real client interface)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)

    def set_responses(self, responses: list[str]) -> None:
        """Set a sequence of responses to return."""
        self._responses = responses
        self._call_index = 0

    def completion(self, prompt, model: str | None = None) -> str:
        """Return next scripted response or echo the prompt."""
        prompt_str = str(prompt)[:200] if not isinstance(prompt, str) else prompt
        self.call_log.append(prompt_str)

        model = model or self.model_name
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += len(prompt_str)
        self.last_prompt_tokens = len(prompt_str)

        if self._responses and self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
        else:
            response = f"Mock response to: {prompt_str[:50]}"

        self.model_output_tokens[model] += len(response)
        self.last_completion_tokens = len(response)
        return response

    async def acompletion(self, prompt, model: str | None = None) -> str:
        """Async completion — delegates to synchronous completion."""
        return self.completion(prompt, model)

    def get_usage_summary(self) -> UsageSummary:
        """Return aggregated usage across all calls."""
        model_summaries: dict[str, ModelUsageSummary] = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Return usage for the most recent call."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )

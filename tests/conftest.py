"""Shared test fixtures for RLM test suite."""

import pytest

from tests.mock_lm import MockLM


@pytest.fixture
def mock_lm():
    """Create a fresh MockLM instance."""
    return MockLM()


@pytest.fixture
def mock_lm_sequence():
    """Create a MockLM with controllable response sequence.

    Usage:
        def test_example(mock_lm_sequence):
            lm = mock_lm_sequence(["response1", "response2"])
            assert lm.completion("anything") == "response1"
            assert lm.completion("anything") == "response2"
    """

    def _factory(responses: list[str]) -> MockLM:
        lm = MockLM()
        lm.set_responses(responses)
        return lm

    return _factory

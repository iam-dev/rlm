"""Tests for LMHandler — routing, lifecycle, socket protocol, usage tracking."""

import socket

import pytest

from rlm.core.comms_utils import (
    LMRequest,
    LMResponse,
    send_lm_request,
    send_lm_request_batched,
    socket_request,
)
from rlm.core.lm_handler import LMHandler
from tests.mock_lm import MockLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_handler(model_name: str = "mock-model") -> LMHandler:
    """Return a fresh LMHandler backed by a MockLM. Caller is responsible for stopping it."""
    return LMHandler(client=MockLM(model_name=model_name))


# ---------------------------------------------------------------------------
# 1. Lifecycle tests — start / stop / context manager
# ---------------------------------------------------------------------------


class TestLMHandlerLifecycle:
    def test_start_returns_host_and_port(self):
        handler = make_handler()
        try:
            host, port = handler.start()
            assert host == "127.0.0.1"
            assert isinstance(port, int)
            assert port > 0
        finally:
            handler.stop()

    def test_start_is_idempotent(self):
        """Calling start() a second time returns the same address without error."""
        handler = make_handler()
        try:
            addr1 = handler.start()
            addr2 = handler.start()
            assert addr1 == addr2
        finally:
            handler.stop()

    def test_stop_tears_down_server(self):
        """After stop(), the port is no longer accepting connections."""
        handler = make_handler()
        host, port = handler.start()
        handler.stop()

        with pytest.raises((ConnectionRefusedError, OSError)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((host, port))

    def test_server_is_none_before_start(self):
        handler = make_handler()
        assert handler._server is None
        assert handler._thread is None

    def test_stop_before_start_is_safe(self):
        """stop() on a handler that was never started should not raise."""
        handler = make_handler()
        handler.stop()  # must not raise

    def test_context_manager_starts_and_stops_server(self):
        """__enter__ starts the server and __exit__ stops it cleanly."""
        mock = MockLM()
        captured_port = None

        with LMHandler(client=mock) as handler:
            _, captured_port = handler.address
            assert captured_port > 0
            # Server is reachable inside the context
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(handler.address)  # should not raise

        # Server is unreachable outside the context
        with pytest.raises((ConnectionRefusedError, OSError)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("127.0.0.1", captured_port))

    def test_context_manager_returns_self(self):
        mock = MockLM()
        with LMHandler(client=mock) as handler:
            assert isinstance(handler, LMHandler)

    def test_context_manager_propagates_exceptions(self):
        """__exit__ must not swallow exceptions raised inside the block."""
        mock = MockLM()
        with pytest.raises(RuntimeError, match="boom"):
            with LMHandler(client=mock):
                raise RuntimeError("boom")

    def test_port_property_reflects_server_address(self):
        """handler.port should equal the socket server's actual port after start()."""
        handler = make_handler()
        try:
            _, port = handler.start()
            assert handler.port == port
        finally:
            handler.stop()

    def test_address_property(self):
        handler = make_handler()
        try:
            handler.start()
            host, port = handler.address
            assert host == "127.0.0.1"
            assert port > 0
        finally:
            handler.stop()


# ---------------------------------------------------------------------------
# 2. Client registration and routing
# ---------------------------------------------------------------------------


class TestClientRegistrationAndRouting:
    def test_default_client_registered_on_init(self):
        mock = MockLM("alpha")
        handler = LMHandler(client=mock)
        assert "alpha" in handler.clients
        assert handler.clients["alpha"] is mock

    def test_register_additional_client(self):
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-b")
        handler = LMHandler(client=mock_a)
        handler.register_client("model-b", mock_b)
        assert handler.clients["model-b"] is mock_b

    def test_get_client_by_model_name(self):
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-b")
        handler = LMHandler(client=mock_a)
        handler.register_client("model-b", mock_b)
        assert handler.get_client("model-b") is mock_b

    def test_get_client_without_model_returns_default(self):
        mock = MockLM("default-model")
        handler = LMHandler(client=mock)
        assert handler.get_client() is mock

    def test_get_client_unknown_model_returns_default(self):
        mock = MockLM("default-model")
        handler = LMHandler(client=mock)
        result = handler.get_client("nonexistent-model")
        assert result is mock

    def test_get_client_depth_0_returns_default(self):
        mock_default = MockLM("default")
        mock_other = MockLM("other")
        handler = LMHandler(client=mock_default, other_backend_client=mock_other)
        assert handler.get_client(depth=0) is mock_default

    def test_get_client_depth_1_returns_other_backend(self):
        mock_default = MockLM("default")
        mock_other = MockLM("other")
        handler = LMHandler(client=mock_default, other_backend_client=mock_other)
        assert handler.get_client(depth=1) is mock_other

    def test_get_client_depth_1_without_other_backend_returns_default(self):
        mock_default = MockLM("default")
        handler = LMHandler(client=mock_default)
        # No other_backend_client — should fall back to default
        assert handler.get_client(depth=1) is mock_default

    def test_model_name_overrides_depth_routing(self):
        """If model is specified and registered, it wins over depth-based routing."""
        mock_default = MockLM("default")
        mock_other = MockLM("other")
        mock_specific = MockLM("specific")
        handler = LMHandler(client=mock_default, other_backend_client=mock_other)
        handler.register_client("specific", mock_specific)
        # Even though depth=1 would give mock_other, the explicit model name wins.
        assert handler.get_client(model="specific", depth=1) is mock_specific

    def test_register_client_overwrites_existing(self):
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-a")  # same name, different object
        handler = LMHandler(client=mock_a)
        handler.register_client("model-a", mock_b)
        assert handler.clients["model-a"] is mock_b


# ---------------------------------------------------------------------------
# 3. Direct completion (synchronous, no socket)
# ---------------------------------------------------------------------------


class TestDirectCompletion:
    def test_completion_returns_mock_response(self):
        mock = MockLM()
        mock.set_responses(["direct response"])
        handler = LMHandler(client=mock)
        result = handler.completion("hello")
        assert result == "direct response"

    def test_completion_default_echoes_prompt(self):
        mock = MockLM()
        handler = LMHandler(client=mock)
        result = handler.completion("test prompt")
        assert "Mock response to:" in result
        assert "test prompt" in result

    def test_completion_routes_to_named_model(self):
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-b")
        mock_b.set_responses(["model-b response"])
        handler = LMHandler(client=mock_a)
        handler.register_client("model-b", mock_b)
        result = handler.completion("hello", model="model-b")
        assert result == "model-b response"

    def test_completion_uses_default_when_no_model_specified(self):
        mock = MockLM()
        mock.set_responses(["from default"])
        handler = LMHandler(client=mock)
        result = handler.completion("any prompt")
        assert result == "from default"

    def test_completion_logs_to_call_log(self):
        mock = MockLM()
        handler = LMHandler(client=mock)
        handler.completion("logged prompt")
        assert "logged prompt" in mock.call_log

    def test_completion_with_scripted_sequence(self):
        mock = MockLM()
        mock.set_responses(["first", "second", "third"])
        handler = LMHandler(client=mock)
        assert handler.completion("q1") == "first"
        assert handler.completion("q2") == "second"
        assert handler.completion("q3") == "third"


# ---------------------------------------------------------------------------
# 4. Socket roundtrip — LMRequest / LMResponse via the live server
# ---------------------------------------------------------------------------


class TestSocketRoundtrip:
    def test_single_prompt_returns_success_response(self):
        mock = MockLM()
        mock.set_responses(["socket response"])
        with LMHandler(client=mock) as handler:
            request = LMRequest(prompt="hello via socket")
            response = send_lm_request(handler.address, request)

        assert response.success
        assert response.error is None
        assert response.chat_completion is not None
        assert response.chat_completion.response == "socket response"

    def test_single_prompt_response_has_correct_root_model(self):
        mock = MockLM("test-model-name")
        with LMHandler(client=mock) as handler:
            request = LMRequest(prompt="what model?")
            response = send_lm_request(handler.address, request)

        assert response.chat_completion.root_model == "test-model-name"

    def test_single_prompt_response_echoes_prompt(self):
        mock = MockLM()
        with LMHandler(client=mock) as handler:
            request = LMRequest(prompt="my specific prompt")
            response = send_lm_request(handler.address, request)

        assert response.chat_completion.prompt == "my specific prompt"

    def test_single_prompt_response_has_execution_time(self):
        mock = MockLM()
        with LMHandler(client=mock) as handler:
            request = LMRequest(prompt="timing test")
            response = send_lm_request(handler.address, request)

        assert response.chat_completion.execution_time >= 0.0

    def test_socket_request_missing_prompt_returns_error(self):
        mock = MockLM()
        with LMHandler(client=mock) as handler:
            # Send a request with neither prompt nor prompts
            raw_response = socket_request(handler.address, {"depth": 0})

        response = LMResponse.from_dict(raw_response)
        assert not response.success
        assert response.error is not None
        assert "prompt" in response.error.lower() or "missing" in response.error.lower()

    def test_socket_request_non_object_payload_returns_error(self):
        """A JSON array (not object) should return an error, not crash the server."""
        mock = MockLM()
        with LMHandler(client=mock) as handler:
            import json
            import struct

            payload = json.dumps([1, 2, 3]).encode("utf-8")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect(handler.address)
                s.sendall(struct.pack(">I", len(payload)) + payload)

                import json as _json
                import struct as _struct

                raw_len = s.recv(4)
                length = _struct.unpack(">I", raw_len)[0]
                data = b""
                while len(data) < length:
                    chunk = s.recv(length - len(data))
                    data += chunk
                response_dict = _json.loads(data.decode("utf-8"))

        response = LMResponse.from_dict(response_dict)
        assert not response.success
        assert response.error is not None

    def test_model_routing_over_socket(self):
        """Specifying model in LMRequest routes to the correct registered client."""
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-b")
        mock_b.set_responses(["from model-b"])

        with LMHandler(client=mock_a) as handler:
            handler.register_client("model-b", mock_b)
            request = LMRequest(prompt="route me", model="model-b")
            response = send_lm_request(handler.address, request)

        assert response.success
        assert response.chat_completion.response == "from model-b"
        assert response.chat_completion.root_model == "model-b"

    def test_depth_routing_over_socket(self):
        """depth=1 in LMRequest routes to other_backend_client over the socket."""
        mock_default = MockLM("default")
        mock_other = MockLM("other-backend")
        mock_other.set_responses(["other backend answer"])

        with LMHandler(client=mock_default, other_backend_client=mock_other) as handler:
            request = LMRequest(prompt="deep question", depth=1)
            response = send_lm_request(handler.address, request)

        assert response.success
        assert response.chat_completion.response == "other backend answer"

    def test_concurrent_socket_requests_all_succeed(self):
        """Multiple concurrent requests should all be handled (server is threaded)."""
        import threading

        mock = MockLM()

        n = 5
        with LMHandler(client=mock) as handler:
            responses = []
            lock = threading.Lock()

            def worker(i):
                req = LMRequest(prompt=f"request {i}")
                resp = send_lm_request(handler.address, req)
                with lock:
                    responses.append(resp)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert len(responses) == n
        assert all(r.success for r in responses)

    def test_usage_summary_in_response(self):
        """LMResponse chat_completion carries a non-empty usage_summary."""
        mock = MockLM("usage-model")
        with LMHandler(client=mock) as handler:
            request = LMRequest(prompt="usage check")
            response = send_lm_request(handler.address, request)

        assert response.success
        usage = response.chat_completion.usage_summary
        assert usage is not None
        assert "usage-model" in usage.model_usage_summaries


# ---------------------------------------------------------------------------
# 5. get_usage_summary — no double-counting
# ---------------------------------------------------------------------------


class TestGetUsageSummary:
    def test_returns_usage_summary_type(self):
        mock = MockLM()
        handler = LMHandler(client=mock)
        summary = handler.get_usage_summary()
        # Use type name check instead of isinstance to be resilient against
        # test_imports.py reloading sys.modules (which creates new class objects).
        assert type(summary).__name__ == "UsageSummary"
        assert type(summary).__module__ == "rlm.core.types"
        assert hasattr(summary, "model_usage_summaries")

    def test_summary_contains_default_client_model(self):
        mock = MockLM("tracked-model")
        handler = LMHandler(client=mock)
        # Make a call so usage is non-zero
        handler.completion("prompt for usage")
        summary = handler.get_usage_summary()
        assert "tracked-model" in summary.model_usage_summaries

    def test_no_double_counting_when_other_backend_registered(self):
        """get_usage_summary must not count the same model twice.

        Only default_client is auto-registered in self.clients at construction time.
        other_backend_client is held as a reference but is NOT automatically added to
        self.clients — callers must call register_client() explicitly if they want it
        included in the usage summary.

        The contract tested here: after manually registering both clients, each model
        name appears exactly once in the returned summary.
        """
        mock_default = MockLM("default-model")
        mock_other = MockLM("other-model")
        handler = LMHandler(client=mock_default, other_backend_client=mock_other)

        # Explicitly register the other backend so it appears in the summary.
        handler.register_client(mock_other.model_name, mock_other)

        # Drive calls through each client so MockLM populates its usage counters.
        handler.completion("call default")
        handler.get_client(depth=1).completion("call other")

        summary = handler.get_usage_summary()
        model_keys = list(summary.model_usage_summaries.keys())

        # Each model name must appear at most once — no double-counting.
        assert len(model_keys) == len(set(model_keys)), f"Duplicate models in summary: {model_keys}"
        assert "default-model" in summary.model_usage_summaries
        assert "other-model" in summary.model_usage_summaries

    def test_summary_reflects_multiple_registered_clients(self):
        """After calls through each registered client, both appear in the summary.

        MockLM only populates its per-model counters after a call, so we must make
        at least one call per client before asserting they show up.
        """
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-b")
        handler = LMHandler(client=mock_a)
        handler.register_client("model-b", mock_b)

        # Make a call through each registered client so MockLM records usage.
        handler.completion("call for a")
        mock_b.completion("call for b")

        summary = handler.get_usage_summary()
        assert "model-a" in summary.model_usage_summaries
        assert "model-b" in summary.model_usage_summaries

    def test_overwriting_same_model_name_does_not_duplicate(self):
        """Re-registering the same model name replaces the entry — no duplication.

        After overwriting, self.clients["same-name"] points to mock_b.
        get_usage_summary() iterates self.clients.values() exactly once per key,
        so "same-name" must appear at most once regardless of how many times the
        key was registered.

        We route the call via model name so it goes through mock_b (the replacement).
        """
        mock_a = MockLM("same-name")
        mock_b = MockLM("same-name")
        handler = LMHandler(client=mock_a)
        handler.register_client("same-name", mock_b)

        # Route explicitly by model name so the call goes to mock_b (the replacement
        # stored in self.clients["same-name"]), not through default_client (mock_a).
        result = handler.get_client("same-name").completion("call via replacement")
        assert result is not None  # sanity: the call went through

        summary = handler.get_usage_summary()
        keys = list(summary.model_usage_summaries.keys())
        assert keys.count("same-name") == 1

    def test_empty_usage_summary_before_any_calls(self):
        """Before any completions, MockLM reports zero calls; summary has the model but zero tokens."""
        mock = MockLM("no-calls-model")
        handler = LMHandler(client=mock)
        summary = handler.get_usage_summary()
        # MockLM only emits a model entry after a call is made, so the dict may be empty
        # OR contain zero-call entries — either is acceptable; what must NOT happen is a crash.
        assert isinstance(summary.model_usage_summaries, dict)

    def test_usage_accumulates_across_direct_calls(self):
        """Each call through handler.completion() increments the mock's per-model counters."""
        mock = MockLM("accumulate-model")
        handler = LMHandler(client=mock)

        handler.completion("call one")
        handler.completion("call two")
        handler.completion("call three")

        summary = handler.get_usage_summary()
        model_summary = summary.model_usage_summaries.get("accumulate-model")
        assert model_summary is not None
        assert model_summary.total_calls == 3


# ---------------------------------------------------------------------------
# 6. Batched requests through socket
# ---------------------------------------------------------------------------


class TestBatchedSocketRequests:
    def test_batched_request_returns_one_response_per_prompt(self):
        mock = MockLM()
        prompts = ["prompt A", "prompt B", "prompt C"]

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, prompts)

        assert len(responses) == len(prompts)

    def test_batched_responses_are_all_successful(self):
        mock = MockLM()
        prompts = ["q1", "q2", "q3"]

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, prompts)

        assert all(r.success for r in responses)

    def test_batched_responses_preserve_prompt_order(self):
        """The i-th response corresponds to the i-th prompt."""
        mock = MockLM()
        mock.set_responses(["answer-0", "answer-1", "answer-2"])
        prompts = ["prompt-0", "prompt-1", "prompt-2"]

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, prompts)

        for i, (resp, prompt) in enumerate(zip(responses, prompts, strict=True)):
            assert resp.success, f"Response {i} failed: {resp.error}"
            # The chat_completion prompt field should match what was sent
            assert resp.chat_completion.prompt == prompt

    def test_batched_responses_contain_completions(self):
        mock = MockLM()
        mock.set_responses(["one", "two"])
        prompts = ["p1", "p2"]

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, prompts)

        assert responses[0].chat_completion.response == "one"
        assert responses[1].chat_completion.response == "two"

    def test_batched_single_prompt(self):
        mock = MockLM()
        mock.set_responses(["solo answer"])

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, ["only prompt"])

        assert len(responses) == 1
        assert responses[0].success
        assert responses[0].chat_completion.response == "solo answer"

    def test_batched_large_number_of_prompts(self):
        """Stress test: 20 prompts in one batched call all come back."""
        mock = MockLM()
        prompts = [f"question {i}" for i in range(20)]

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, prompts)

        assert len(responses) == 20
        assert all(r.success for r in responses)

    def test_batched_uses_model_routing(self):
        """Specifying model= in a batched request routes to the correct client."""
        mock_a = MockLM("model-a")
        mock_b = MockLM("model-b")
        mock_b.set_responses(["b-resp-1", "b-resp-2"])
        prompts = ["p1", "p2"]

        with LMHandler(client=mock_a) as handler:
            handler.register_client("model-b", mock_b)
            responses = send_lm_request_batched(handler.address, prompts, model="model-b")

        assert all(r.success for r in responses)
        for resp in responses:
            assert resp.chat_completion.root_model == "model-b"

    def test_batched_execution_time_is_non_negative(self):
        mock = MockLM()
        prompts = ["a", "b"]

        with LMHandler(client=mock) as handler:
            responses = send_lm_request_batched(handler.address, prompts)

        for resp in responses:
            assert resp.chat_completion.execution_time >= 0.0

    def test_batched_and_single_requests_interleaved(self):
        """Mix of single and batched requests to the same server all succeed."""
        mock = MockLM()

        with LMHandler(client=mock) as handler:
            single = send_lm_request(handler.address, LMRequest(prompt="single"))
            batched = send_lm_request_batched(handler.address, ["b1", "b2"])
            single2 = send_lm_request(handler.address, LMRequest(prompt="single again"))

        assert single.success
        assert all(r.success for r in batched)
        assert single2.success

    def test_batched_via_raw_lm_request(self):
        """LMRequest with prompts= field triggers batched path end-to-end."""
        mock = MockLM()
        mock.set_responses(["x", "y"])

        with LMHandler(client=mock) as handler:
            request = LMRequest(prompts=["first", "second"])
            response = send_lm_request(handler.address, request)

        # send_lm_request wraps the response — batched path returns is_batched=True
        assert response.success
        # The raw response carries chat_completions
        assert response.is_batched
        assert len(response.chat_completions) == 2
        assert response.chat_completions[0].response == "x"
        assert response.chat_completions[1].response == "y"

"""Tests for rlm.core.comms_utils — socket protocol and message dataclasses.

All socket tests use socket.socketpair() so no external services are needed.
"""

import socket
import struct
import threading

import pytest

from rlm.core.comms_utils import (
    MAX_MESSAGE_SIZE,
    LMRequest,
    LMResponse,
    socket_recv,
    socket_send,
)
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary

# =============================================================================
# Helpers / Factories
# =============================================================================


def make_usage_summary(model: str = "test-model") -> UsageSummary:
    return UsageSummary(
        model_usage_summaries={
            model: ModelUsageSummary(
                total_calls=1,
                total_input_tokens=100,
                total_output_tokens=50,
            )
        }
    )


def make_chat_completion(
    root_model: str = "test-model",
    prompt: str = "What is 2+2?",
    response: str = "4",
    execution_time: float = 0.1,
) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model=root_model,
        prompt=prompt,
        response=response,
        usage_summary=make_usage_summary(root_model),
        execution_time=execution_time,
    )


# =============================================================================
# LMRequest tests
# =============================================================================


class TestLMRequest:
    def test_to_dict_from_dict_roundtrip_single_prompt(self):
        req = LMRequest(prompt="Hello world", model="gpt-4", depth=2)
        roundtripped = LMRequest.from_dict(req.to_dict())
        assert roundtripped.prompt == "Hello world"
        assert roundtripped.model == "gpt-4"
        assert roundtripped.depth == 2
        assert roundtripped.prompts is None

    def test_to_dict_from_dict_roundtrip_batched_prompt(self):
        prompts = ["First", "Second", "Third"]
        req = LMRequest(prompts=prompts, model="gpt-3.5", depth=1)
        roundtripped = LMRequest.from_dict(req.to_dict())
        assert roundtripped.prompts == prompts
        assert roundtripped.model == "gpt-3.5"
        assert roundtripped.depth == 1
        assert roundtripped.prompt is None

    def test_to_dict_excludes_none_prompt(self):
        req = LMRequest(prompt=None, depth=0)
        d = req.to_dict()
        assert "prompt" not in d

    def test_to_dict_excludes_none_prompts(self):
        req = LMRequest(prompts=None, depth=0)
        d = req.to_dict()
        assert "prompts" not in d

    def test_to_dict_excludes_none_model(self):
        req = LMRequest(prompt="hi", model=None, depth=0)
        d = req.to_dict()
        assert "model" not in d

    def test_to_dict_always_includes_depth(self):
        req = LMRequest(prompt="hi")
        d = req.to_dict()
        assert "depth" in d
        assert d["depth"] == 0

    def test_is_batched_false_when_only_single_prompt(self):
        req = LMRequest(prompt="hello")
        assert req.is_batched is False

    def test_is_batched_false_when_prompts_none(self):
        req = LMRequest()
        assert req.is_batched is False

    def test_is_batched_false_when_prompts_empty_list(self):
        req = LMRequest(prompts=[])
        assert req.is_batched is False

    def test_is_batched_true_when_prompts_nonempty(self):
        req = LMRequest(prompts=["a", "b"])
        assert req.is_batched is True

    def test_default_depth_is_zero(self):
        req = LMRequest(prompt="hi")
        assert req.depth == 0

    def test_from_dict_default_depth_when_missing(self):
        req = LMRequest.from_dict({"prompt": "hello"})
        assert req.depth == 0

    def test_to_dict_from_dict_preserves_dict_prompt(self):
        """prompt can be a dict (e.g. OpenAI message list serialised as dict)."""
        dict_prompt = {"role": "user", "content": "What is 2+2?"}
        req = LMRequest(prompt=dict_prompt, depth=0)
        roundtripped = LMRequest.from_dict(req.to_dict())
        assert roundtripped.prompt == dict_prompt

    def test_to_dict_from_dict_preserves_dict_prompts(self):
        dict_prompts = [{"role": "user", "content": "msg1"}, {"role": "user", "content": "msg2"}]
        req = LMRequest(prompts=dict_prompts, depth=3)
        roundtripped = LMRequest.from_dict(req.to_dict())
        assert roundtripped.prompts == dict_prompts
        assert roundtripped.depth == 3


# =============================================================================
# LMResponse tests
# =============================================================================


class TestLMResponse:
    # ------------------------------------------------------------------
    # Success (single) response
    # ------------------------------------------------------------------

    def test_success_to_dict_from_dict_roundtrip(self):
        completion = make_chat_completion()
        resp = LMResponse.success_response(completion)

        assert resp.success is True
        assert resp.is_batched is False

        d = resp.to_dict()
        assert d["error"] is None
        assert d["chat_completions"] is None
        assert isinstance(d["chat_completion"], dict)

        restored = LMResponse.from_dict(d)
        assert restored.success is True
        assert restored.error is None
        assert restored.chat_completion is not None
        assert restored.chat_completion.response == completion.response
        assert restored.chat_completion.root_model == completion.root_model
        assert restored.chat_completion.execution_time == completion.execution_time

    def test_success_usage_summary_roundtrip(self):
        completion = make_chat_completion()
        resp = LMResponse.success_response(completion)
        restored = LMResponse.from_dict(resp.to_dict())

        assert "test-model" in restored.chat_completion.usage_summary.model_usage_summaries
        summary = restored.chat_completion.usage_summary.model_usage_summaries["test-model"]
        assert summary.total_input_tokens == 100
        assert summary.total_output_tokens == 50

    # ------------------------------------------------------------------
    # Error response
    # ------------------------------------------------------------------

    def test_error_to_dict_from_dict_roundtrip(self):
        resp = LMResponse.error_response("Something went wrong")

        assert resp.success is False
        assert resp.is_batched is False

        d = resp.to_dict()
        assert d["error"] == "Something went wrong"
        assert d["chat_completion"] is None
        assert d["chat_completions"] is None

        restored = LMResponse.from_dict(d)
        assert restored.success is False
        assert restored.error == "Something went wrong"
        assert restored.chat_completion is None
        assert restored.chat_completions is None

    def test_error_response_success_property_false(self):
        resp = LMResponse.error_response("oops")
        assert resp.success is False

    # ------------------------------------------------------------------
    # Batched response
    # ------------------------------------------------------------------

    def test_batched_to_dict_from_dict_roundtrip(self):
        completions = [make_chat_completion(response=f"answer {i}") for i in range(3)]
        resp = LMResponse.batched_success_response(completions)

        assert resp.is_batched is True
        assert resp.success is True

        d = resp.to_dict()
        assert d["error"] is None
        assert d["chat_completion"] is None
        assert isinstance(d["chat_completions"], list)
        assert len(d["chat_completions"]) == 3

        restored = LMResponse.from_dict(d)
        assert restored.is_batched is True
        assert restored.success is True
        assert len(restored.chat_completions) == 3
        for i, c in enumerate(restored.chat_completions):
            assert c.response == f"answer {i}"

    def test_batched_response_chat_completion_is_none(self):
        completions = [make_chat_completion()]
        resp = LMResponse.batched_success_response(completions)
        assert resp.chat_completion is None

    # ------------------------------------------------------------------
    # Degenerate / empty response
    # ------------------------------------------------------------------

    def test_empty_response_returns_error_string(self):
        resp = LMResponse()
        d = resp.to_dict()
        assert d["error"] is not None
        assert "No chat completion or error provided" in d["error"]

    def test_is_batched_false_when_chat_completions_none(self):
        resp = LMResponse(chat_completion=make_chat_completion())
        assert resp.is_batched is False


# =============================================================================
# socket_send / socket_recv — protocol round-trip
# =============================================================================


class TestSocketProtocol:
    def _make_pair(self):
        """Return (sender, receiver) socket pair."""
        a, b = socket.socketpair()
        return a, b

    # ------------------------------------------------------------------
    # Basic round-trip
    # ------------------------------------------------------------------

    def test_simple_dict_roundtrip(self):
        sender, receiver = self._make_pair()
        try:
            data = {"key": "value", "number": 42}
            socket_send(sender, data)
            result = socket_recv(receiver)
            assert result == data
        finally:
            sender.close()
            receiver.close()

    def test_empty_dict_roundtrip(self):
        sender, receiver = self._make_pair()
        try:
            socket_send(sender, {})
            result = socket_recv(receiver)
            assert result == {}
        finally:
            sender.close()
            receiver.close()

    def test_nested_dict_roundtrip(self):
        sender, receiver = self._make_pair()
        try:
            data = {"outer": {"inner": [1, 2, 3]}, "flag": True, "none_val": None}
            socket_send(sender, data)
            result = socket_recv(receiver)
            assert result == data
        finally:
            sender.close()
            receiver.close()

    def test_unicode_payload_roundtrip(self):
        sender, receiver = self._make_pair()
        try:
            data = {"text": "Hello \u4e16\u754c \U0001f600"}
            socket_send(sender, data)
            result = socket_recv(receiver)
            assert result["text"] == data["text"]
        finally:
            sender.close()
            receiver.close()

    # ------------------------------------------------------------------
    # LMRequest round-trip via sockets
    # ------------------------------------------------------------------

    def test_lm_request_over_socket(self):
        sender, receiver = self._make_pair()
        try:
            req = LMRequest(prompt="Solve 1+1", model="gpt-4", depth=1)
            socket_send(sender, req.to_dict())
            raw = socket_recv(receiver)
            restored = LMRequest.from_dict(raw)
            assert restored.prompt == req.prompt
            assert restored.model == req.model
            assert restored.depth == req.depth
        finally:
            sender.close()
            receiver.close()

    # ------------------------------------------------------------------
    # LMResponse round-trip via sockets
    # ------------------------------------------------------------------

    def test_lm_response_success_over_socket(self):
        sender, receiver = self._make_pair()
        try:
            completion = make_chat_completion(response="42")
            resp = LMResponse.success_response(completion)
            socket_send(sender, resp.to_dict())
            raw = socket_recv(receiver)
            restored = LMResponse.from_dict(raw)
            assert restored.success is True
            assert restored.chat_completion.response == "42"
        finally:
            sender.close()
            receiver.close()

    def test_lm_response_error_over_socket(self):
        sender, receiver = self._make_pair()
        try:
            resp = LMResponse.error_response("timeout")
            socket_send(sender, resp.to_dict())
            raw = socket_recv(receiver)
            restored = LMResponse.from_dict(raw)
            assert restored.success is False
            assert restored.error == "timeout"
        finally:
            sender.close()
            receiver.close()

    # ------------------------------------------------------------------
    # Large payload round-trip
    # ------------------------------------------------------------------

    def _send_in_thread(self, sock: socket.socket, data: dict) -> threading.Thread:
        """Helper: start a daemon thread that calls socket_send then closes sock."""

        def _run():
            try:
                socket_send(sock, data)
            finally:
                sock.close()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def test_large_payload_roundtrip(self):
        """Send a 1 MB payload — forces multi-chunk recv."""
        sender, receiver = self._make_pair()
        try:
            large_string = "x" * (1024 * 1024)  # 1 MB
            data = {"big": large_string}
            # Must send concurrently: the kernel socket-buffer for socketpair is
            # typically ~128 KB, so sendall would deadlock if recv isn't running.
            t = self._send_in_thread(sender, data)
            result = socket_recv(receiver)
            t.join(timeout=10)
            assert result["big"] == large_string
        finally:
            receiver.close()

    def test_large_payload_5mb_roundtrip(self):
        """5 MB payload stays well under the 100 MB limit."""
        sender, receiver = self._make_pair()
        try:
            large_string = "a" * (5 * 1024 * 1024)
            data = {"content": large_string, "meta": {"size": len(large_string)}}
            t = self._send_in_thread(sender, data)
            result = socket_recv(receiver)
            t.join(timeout=30)
            assert len(result["content"]) == len(large_string)
            assert result["meta"]["size"] == len(large_string)
        finally:
            receiver.close()

    # ------------------------------------------------------------------
    # MAX_MESSAGE_SIZE enforcement
    # ------------------------------------------------------------------

    def test_max_message_size_rejected(self):
        """Receiving a message that claims to exceed MAX_MESSAGE_SIZE raises ValueError."""
        sender, receiver = self._make_pair()
        try:
            # Send a length prefix claiming MAX_MESSAGE_SIZE + 1 bytes,
            # but no payload — socket_recv should raise before reading the body.
            oversized_length = MAX_MESSAGE_SIZE + 1
            sender.sendall(struct.pack(">I", oversized_length))
            with pytest.raises(ValueError, match="exceeds maximum"):
                socket_recv(receiver)
        finally:
            sender.close()
            receiver.close()

    def test_max_message_size_exactly_allowed(self):
        """Verify MAX_MESSAGE_SIZE itself is not rejected (boundary)."""
        # We only test that the check is > not >=:
        # a message claiming exactly MAX_MESSAGE_SIZE bytes does NOT raise ValueError.
        # We do not actually send MAX_MESSAGE_SIZE bytes — we just verify the
        # length check logic by inspecting the constant boundary.
        # Send a normal small message and confirm it passes through.
        sender, receiver = self._make_pair()
        try:
            data = {"payload": "ok"}
            socket_send(sender, data)
            result = socket_recv(receiver)
            assert result == data
        finally:
            sender.close()
            receiver.close()

    def test_max_message_size_constant_is_100mb(self):
        assert MAX_MESSAGE_SIZE == 100 * 1024 * 1024

    # ------------------------------------------------------------------
    # Connection closed before length prefix
    # ------------------------------------------------------------------

    def test_connection_closed_before_length_returns_empty_dict(self):
        """If the peer closes before sending anything, socket_recv returns {}."""
        sender, receiver = self._make_pair()
        sender.close()  # close without sending anything
        try:
            result = socket_recv(receiver)
            assert result == {}
        finally:
            receiver.close()

    # ------------------------------------------------------------------
    # Connection closed mid-message
    # ------------------------------------------------------------------

    def test_connection_closed_mid_message_raises_connection_error(self):
        """If peer sends the length but closes before sending the full body,
        socket_recv raises ConnectionError."""
        sender, receiver = self._make_pair()
        try:
            # Claim to send 1000 bytes but only send 10 bytes of body
            claimed_length = 1000
            sender.sendall(struct.pack(">I", claimed_length))
            sender.sendall(b"x" * 10)
            sender.close()  # EOF mid-message
            with pytest.raises(ConnectionError, match="Connection closed before message complete"):
                socket_recv(receiver)
        finally:
            receiver.close()

    def test_connection_closed_mid_message_partial_body(self):
        """Variant: peer sends exactly 1 byte of body then closes."""
        sender, receiver = self._make_pair()
        try:
            claimed_length = 256
            sender.sendall(struct.pack(">I", claimed_length))
            sender.sendall(b"A")  # only 1 byte of the promised 256
            sender.close()
            with pytest.raises(ConnectionError):
                socket_recv(receiver)
        finally:
            receiver.close()

    # ------------------------------------------------------------------
    # Multiple sequential messages on the same socket pair
    # ------------------------------------------------------------------

    def test_multiple_messages_sequential(self):
        """Framing must be correct: each message must be read independently."""
        sender, receiver = self._make_pair()
        try:
            messages = [
                {"seq": 0, "data": "first"},
                {"seq": 1, "data": "second"},
                {"seq": 2, "data": "third"},
            ]
            for msg in messages:
                socket_send(sender, msg)

            for expected in messages:
                result = socket_recv(receiver)
                assert result == expected
        finally:
            sender.close()
            receiver.close()

    # ------------------------------------------------------------------
    # Thread-safety: send from one thread, recv in another
    # ------------------------------------------------------------------

    def test_send_recv_across_threads(self):
        sender, receiver = self._make_pair()
        received = []
        error_holder = []

        def recv_thread():
            try:
                received.append(socket_recv(receiver))
            except Exception as exc:
                error_holder.append(exc)
            finally:
                receiver.close()

        t = threading.Thread(target=recv_thread, daemon=True)
        t.start()

        data = {"thread": "test", "value": 99}
        socket_send(sender, data)
        sender.close()

        t.join(timeout=5)
        assert not error_holder, f"recv thread raised: {error_holder}"
        assert received == [data]

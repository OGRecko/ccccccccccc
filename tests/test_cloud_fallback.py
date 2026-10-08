"""The cloud fallback must be off, honest, and unable to act.

The requirement is one line - "cloud fallback must be OFF by default" - but the
promise behind it is bigger: nothing leaves this machine unless the user asks
for it, and when they do ask, the cloud cannot start running tools behind their
back. So these tests check the promise, not the line:

* **off (the shipped default)** - the cloud client is never even *constructed*
  (a bomb class proves no code path reaches it), a failing local model stays
  local, and `cloud_used_last_turn` stays false;
* **on without a key** - still local, with a warning; no client is built;
* **on with a key** - the fallback runs when the local model fails, and
  `use_only_if_local_down: false` does not silently disable it (it used to);
* **cannot act** - the client strips tool schemas unless `allow_tools: true`;
* **the key stays out of every log**, even after a fallback turn.

Everything here is offline: the "cloud" is a fake session that records the
request body instead of sending it.
"""

from __future__ import annotations

import json
from typing import Any

import core.brain as brain_module
from core.brain import Brain, OpenAICompatibleClient
from core.config import Config
from tests.mock_ollama import MockOllama

KEY_ENV = "GARVIS_CLOUD_API_KEY"
CLOUD_TEXT = "Cloud here. Local model was down."


# ---------------------------------------------------------------------------
# fakes: a cloud that never leaves the process, and a bomb that never fires
# ---------------------------------------------------------------------------
class _FakeResponse:
    """Just enough of a requests.Response for the streaming path."""

    status_code = 200

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self, decode_unicode: bool = True):
        payload = {"choices": [{"delta": {"content": CLOUD_TEXT}, "finish_reason": None}]}
        yield f"data: {json.dumps(payload)}"
        yield "data: [DONE]"


class _FakeSession:
    """Records the request bodies the client would have sent."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def post(self, url: str, json: dict | None = None, **kwargs: Any) -> _FakeResponse:
        self.payloads.append(json or {})
        return _FakeResponse()


class _BombClient:
    """Explodes if anything tries to build a cloud client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a cloud client was constructed while the fallback is off")


def _offline_cloud_client() -> OpenAICompatibleClient:
    """A real client whose HTTP layer is the fake session above."""
    client = OpenAICompatibleClient(
        base_url="https://cloud.invalid/v1", model="test-cloud", api_key="not-a-real-key",
    )
    client.session = _FakeSession()
    return client


def _log_text(cfg: Config) -> str:
    """Everything on disk that is a log for this session."""
    text = ""
    for key in ("logging.file", "logging.activity_log", "logging.activity_jsonl"):
        path = cfg.resolve_path(cfg.get(key))
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="replace")
    return text


# ---------------------------------------------------------------------------
# off by default
# ---------------------------------------------------------------------------
def test_the_shipped_config_ships_with_the_fallback_off(cfg: Config) -> None:
    assert cfg.get("brain.cloud_fallback.enabled") is False
    assert cfg.cloud_fallback_enabled is False


def test_no_cloud_client_is_ever_built_while_it_is_off(cfg: Config, activity, monkeypatch) -> None:
    """Not "it is not called" - it does not exist. A bomb proves the point."""
    monkeypatch.setattr(brain_module, "OpenAICompatibleClient", _BombClient)
    monkeypatch.setenv(KEY_ENV, "a-key-that-must-not-be-used")

    brain = Brain(cfg)  # must not explode

    assert brain._cloud is None, "a cloud client was built although the fallback is off"
    assert brain.cloud_used_last_turn is False


def test_a_local_failure_stays_local_when_it_is_off(cfg: Config, activity,
                                                    mock_ollama: MockOllama, monkeypatch) -> None:
    monkeypatch.setattr(brain_module, "OpenAICompatibleClient", _BombClient)
    cfg.set("brain.host", mock_ollama.url)
    brain = Brain(cfg)
    mock_ollama.fail_requests(3)  # the local model "dies"

    result = brain.respond("hello")

    assert result.error, "a dead local model must be reported, not papered over"
    assert result.provider == "ollama"
    assert brain.cloud_used_last_turn is False
    assert "cloud" not in (result.reply or "").lower()


# ---------------------------------------------------------------------------
# enabled, but not usable
# ---------------------------------------------------------------------------
def test_enabling_without_the_key_stays_local(cfg: Config, activity, monkeypatch) -> None:
    """Turning it on is not enough - the key has to be there too."""
    cfg.set("brain.cloud_fallback.enabled", True)
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr(brain_module, "OpenAICompatibleClient", _BombClient)

    brain = Brain(cfg)  # again: must not explode

    assert brain._cloud is None
    assert any("not set" in w for w in cfg.validate()), cfg.validate()


def test_the_missing_key_is_reported_by_validate(cfg: Config) -> None:
    cfg.set("brain.cloud_fallback.enabled", True)
    warnings = cfg.validate()
    assert any("Cloud fallback is enabled" in w for w in warnings), warnings


def test_enabling_with_the_key_builds_a_locked_down_client(cfg: Config, activity,
                                                           monkeypatch) -> None:
    built: list[dict[str, Any]] = []

    class _Recording(OpenAICompatibleClient):
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)
            super().__init__(**kwargs)

    cfg.set("brain.cloud_fallback.enabled", True)
    monkeypatch.setenv(KEY_ENV, "not-a-real-key")
    monkeypatch.setattr(brain_module, "OpenAICompatibleClient", _Recording)

    brain = Brain(cfg)

    assert brain._cloud is not None and built
    assert built[0]["allow_tools"] is False, "the fallback must not be allowed to act by default"
    assert built[0]["model"] == cfg.get("brain.cloud_fallback.model")
    assert built[0]["api_key"] == "not-a-real-key"


# ---------------------------------------------------------------------------
# the fallback actually works (and the knob cannot silently kill it)
# ---------------------------------------------------------------------------
def _fallback_brain(cfg: Config, mock_ollama: MockOllama, monkeypatch, *,
                    use_only_if_local_down: bool | None = None) -> Brain:
    cfg.set("brain.host", mock_ollama.url)
    cfg.set("brain.cloud_fallback.enabled", True)
    if use_only_if_local_down is not None:
        cfg.set("brain.cloud_fallback.use_only_if_local_down", use_only_if_local_down)
    monkeypatch.setenv(KEY_ENV, "not-a-real-key")
    brain = Brain(cfg)
    brain._cloud = _offline_cloud_client()  # never hits the network
    return brain


def test_a_local_failure_falls_back_to_the_cloud(cfg: Config, activity, mock_ollama: MockOllama,
                                                 monkeypatch) -> None:
    brain = _fallback_brain(cfg, mock_ollama, monkeypatch)
    mock_ollama.fail_requests(3)

    result = brain.respond("hello")

    assert result.provider == "cloud", result.error
    assert CLOUD_TEXT in result.reply
    assert brain.cloud_used_last_turn is True


def test_use_only_if_local_down_false_does_not_silently_disable_the_fallback(
    cfg: Config, activity, mock_ollama: MockOllama, monkeypatch,
) -> None:
    """The regression: this setting used to switch the fallback off entirely.

    `_try_cloud` is only reached when the local model failed, so "only if local
    is down" holds either way. Setting the knob to false asks for cloud-first
    answers - which this build does not implement - so the fallback must keep
    working and the user must be *told* the extra behaviour is missing.
    """
    brain = _fallback_brain(cfg, mock_ollama, monkeypatch, use_only_if_local_down=False)
    mock_ollama.fail_requests(3)

    result = brain.respond("hello")

    assert result.provider == "cloud", "the knob silently disabled the fallback"
    assert CLOUD_TEXT in result.reply


def test_the_false_setting_is_warned_about_not_ignored(cfg: Config, activity, monkeypatch) -> None:
    cfg.set("brain.cloud_fallback.enabled", True)
    cfg.set("brain.cloud_fallback.use_only_if_local_down", False)
    monkeypatch.setenv(KEY_ENV, "not-a-real-key")

    warnings = cfg.validate()
    assert any("cloud-first" in w for w in warnings), warnings

    Brain(cfg)  # the warning must be logged too, for people who never run --check
    assert "cloud-first" in _log_text(cfg), "nothing in the log explains the ignored setting"


def test_the_setting_defaults_to_true_in_the_shipped_config(cfg: Config) -> None:
    assert cfg.get("brain.cloud_fallback.use_only_if_local_down") is True


# ---------------------------------------------------------------------------
# can talk, cannot act
# ---------------------------------------------------------------------------
def test_the_client_drops_tool_schemas_unless_allowed() -> None:
    schema = [{"type": "function", "function": {"name": "files.delete", "parameters": {}}}]
    messages = [{"role": "user", "content": "delete my files"}]

    client = _offline_cloud_client()
    list(client.chat_stream(messages, tools=schema))
    assert "tools" not in client.session.payloads[0], (
        "a cloud that cannot act was handed the tool schemas anyway"
    )

    allowed = _offline_cloud_client()
    allowed.allow_tools = True
    list(allowed.chat_stream(messages, tools=schema))
    assert "tools" in allowed.session.payloads[0], (
        "allow_tools: true must actually pass the tools through"
    )


def test_a_fallback_turn_cannot_reach_the_gate(cfg: Config, activity, mock_ollama: MockOllama,
                                               monkeypatch) -> None:
    """Even if a cloud model *asks* for a tool, the gate is what decides.

    The client strips the schemas by default (above). This is the belt to that
    braces: a tool call that arrives anyway goes through the same gate as any
    other, and the brain is not allowed to run it directly.
    """
    calls: list[tuple[str, dict]] = []

    class _Gate:
        def execute(self, name: str, args: dict) -> Any:
            calls.append((name, args))
            from core.permissions import GateOutcome, BLOCKED

            return GateOutcome(ok=False, tool=name, tier="red", decision=BLOCKED,
                               content="BLOCKED: denied by the gate", display="denied")

    brain = _fallback_brain(cfg, mock_ollama, monkeypatch)
    brain.gate = _Gate()
    # The cloud "answers" with a tool call rather than text: the loop must route
    # it through the gate (which refuses) instead of executing anything itself.
    session = _ToolCallSession()
    brain._cloud = _offline_cloud_client()
    brain._cloud.session = session
    mock_ollama.fail_requests(3)

    result = brain.respond("hello")

    assert calls == [("files.delete", {"path": "important.txt"})], (
        "the tool call did not go through the permission gate exactly once"
    )
    assert result.provider == "cloud"
    # The refusal must be fed back as the tool result, so the model is told the
    # truth instead of being left to assume its command worked.
    second_request = json.dumps(session.payloads[-1])
    assert "BLOCKED" in second_request, (
        "the gate's refusal was not passed back to the model"
    )


class _ToolCallSession:
    """Emits an OpenAI tool call, as a hostile/injected cloud model might.

    The first response asks for the tool; after the gate refuses, the second
    answers in words - what a model sees-and-continues would do.
    """

    headers: dict[str, str] = {}

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, url: str, json: dict | None = None, **kwargs: Any) -> _FakeResponse:
        self.payloads.append(json or {})
        return _ToolCallResponse(first=len(self.payloads) == 1)


class _ToolCallResponse(_FakeResponse):
    def __init__(self, first: bool = True) -> None:
        self._first = first

    def iter_lines(self, decode_unicode: bool = True):
        if self._first:
            delta = {"tool_calls": [{"function": {"name": "files.delete",
                                                  "arguments": '{"path": "important.txt"}'}}]}
            finish = "tool_calls"
        else:
            delta = {"content": "I could not do that: the gate refused."}
            finish = "stop"
        yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]})}"
        yield "data: [DONE]"


# ---------------------------------------------------------------------------
# the key stays a key
# ---------------------------------------------------------------------------
def test_the_cloud_key_never_reaches_a_log(cfg: Config, activity, mock_ollama: MockOllama,
                                           monkeypatch) -> None:
    canary = "CANARY-CLOUDKEY-9f3a2b"
    monkeypatch.setenv(KEY_ENV, canary)
    brain = _fallback_brain(cfg, mock_ollama, monkeypatch)
    brain._cloud = _offline_cloud_client()
    brain._cloud._api_key = canary
    assert brain._cloud._api_key == canary, "the test is vacuous if the key is not held"
    mock_ollama.fail_requests(3)

    brain.respond("hello")

    text = _log_text(cfg)
    assert canary not in text, "the cloud API key was written to a log"
    assert "CLOUDKEY" not in text

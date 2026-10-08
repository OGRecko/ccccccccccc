"""The permission gate: GREEN / YELLOW / RED, enforced in code.

This module is the only path from the model to the machine. The brain never
calls a tool function directly - it calls :meth:`PermissionGate.execute`, which:

1. **Classifies** the call (declared tier, name rules, red keywords, path/URL
   allowlists, command patterns, per-tool guards). Classification can only
   *promote*, never demote: a tool declared RED stays RED no matter what a rule
   or the model says.
2. **Refuses** anything unconditionally forbidden (secrets, denied globs, bank
   sites, blocked shell patterns, unknown tools). No confirmation can override
   these - they are simply not available.
3. **Confirms** YELLOW (quick yes) and RED (repeat the exact action, then say
   "confirm"), through a :class:`Confirmer` chain (UI click, voice, console).
4. **Runs** the tool with a timeout and one retry, then verifies the result.
5. **Logs** every decision to the activity log, with secrets redacted.

Why the model cannot bypass this
--------------------------------
* ``Brain`` holds the registry only to *describe* tools; execution goes through
  the gate. There is no tool that executes another tool.
* Tools are looked up by exact name in the gate's own registry - the model
  cannot invent a name or pass a callable.
* Args are validated against config allowlists *after* the model produces them,
  using resolved real paths, so ``~/Documents/../../etc/passwd`` style tricks and
  symlinks fail.
* The gate's decision is not advisory: a denial is returned to the model as a
  tool result, and the tool never runs.
* Config files, the gate itself, and the kill switch are outside every allowlist
  and are additionally in ``files.denied_globs``, so the model cannot edit its
  own permissions even if a write is approved.
"""

from __future__ import annotations

import fnmatch
import os
import queue
import re
import shlex
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from core import safety

GREEN = "green"
YELLOW = "yellow"
RED = "red"
_TIER_RANK = {GREEN: 0, YELLOW: 1, RED: 2}

# Decisions recorded in the log.
VERIFY_FAILED_PREFIX = "VERIFY FAILED:"
#: Tools whose claim cannot be judged without knowing the state beforehand.
_NEEDS_BEFORE = ("files.delete", "files.move", "files.copy")

#: Added to a state-changing tool's output when no check was possible at all.
UNVERIFIED_NOTE = (
    "[not independently verified: this tool reports no state that could be re-checked "
    "afterwards, so 'it worked' here means the tool said so]"
)
FLAGGED = "flagged"        # ran, then the result was not there -> reported as failed
ALLOWED = "allowed"        # GREEN: ran without asking
CONFIRMED = "confirmed"    # user approved a YELLOW/RED action
DENIED = "denied"          # user said no, or the confirmation timed out
BLOCKED = "blocked"        # never allowed, no confirmation possible
ERROR = "error"            # tool raised
TIMEOUT = "timeout"        # tool exceeded its timeout
ABORTED = "aborted"        # too many consecutive failures; stopped
STOPPED = "stopped"        # the kill switch is engaged; nothing runs until resume


def promote(tier: str, to: str) -> str:
    """Return the more restrictive of two tiers."""
    if tier not in _TIER_RANK:
        return RED
    if to not in _TIER_RANK:
        return tier
    return to if _TIER_RANK[to] > _TIER_RANK[tier] else tier


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------
# The *rule* for RED lives in the gate, not in the confirmers: a confirmer is
# only a channel (keyboard, voice, tray button). Any channel can say "yes"; only
# the gate decides whether that is enough for the tier it classified.
@dataclass
class ConfirmRequest:
    """What the user is being asked to approve."""

    tier: str
    tool: str
    summary: str          # one line, human readable
    exact: str            # the exact action, scriptable form
    reasons: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    mode: str = "spoken"  # spoken | click | either | both
    stage: str = "single"  # single | repeat | confirm
    require_phrase: str = "confirm"
    #: The same action described in words, for channels that listen instead of
    #: read ("delete the file voice-demo.txt"). Spoken repeats are checked
    #: against this; typing at the console still uses ``exact``.
    challenge_spoken: str = ""
    #: True when the call carries a secret: channels that would speak the
    #: challenge aloud must abstain (see VoiceConfirmer).
    sensitive: bool = False

    @property
    def challenge(self) -> str:
        """The string the user must reproduce for stage='repeat'."""
        return self.exact

    def match_repeat(self, heard: str) -> tuple[bool, str]:
        """Check a repeat for this request. Returns (matched, why_not).

        Two accepted forms, both strict about *what* is being approved:

        1. the machine form typed exactly (``files.delete path=voice-demo.txt``),
           which is what the console shows;
        2. the spoken form (``delete the file voice demo dot txt``), matched on
           significant words only, so speech-to-text noise like "dot" or an
           added "the file" does not block a legitimate confirmation.

        A negation anywhere in the repeat ("do not delete ...") never matches,
        and a repeat that drops the target does not match either.
        """
        if not heard or not str(heard).strip():
            return False, "nothing was repeated"
        if _norm(heard) == _norm(self.exact):
            return True, ""
        if self.challenge_spoken and _norm(heard) == _norm(self.challenge_spoken):
            return True, ""

        heard_tokens = _significant_tokens(heard)
        if not heard_tokens:
            return False, "nothing recognisable was repeated"
        if _is_negated(heard):
            return False, "the repeat contained a negation"

        best_gap = 1.0
        for candidate in filter(None, (self.challenge_spoken, self.exact)):
            want = _significant_tokens(candidate)
            if not want:
                continue
            missing = [token for token in want if token not in heard_tokens]
            gap = len(missing) / len(want)
            best_gap = min(best_gap, gap)
            if not missing:
                return True, ""
        detail = f" (missing: {', '.join(sorted(set(want) - set(heard_tokens)))[:60]})" if best_gap < 1 else ""
        return False, "the repeated action did not match" + detail


@dataclass
class ConfirmAnswer:
    approved: bool
    method: str = "none"        # console | voice | ui | scripted
    text: str = ""              # what the user actually said/typed
    timed_out: bool = False
    #: True when this channel could not handle the request (wrong medium,
    #: e.g. a secret challenge on voice). The chain then tries the next one
    #: instead of treating it as a refusal.
    unavailable: bool = False
    note: str = ""

    @property
    def spoken_text(self) -> str:
        """Alias kept for readability at call sites."""
        return self.text


class Confirmer(Protocol):
    """A channel that can ask the human a question. Has no policy of its own."""

    name: str

    def confirm(self, request: ConfirmRequest) -> ConfirmAnswer:  # pragma: no cover
        ...


class ConfirmerChain:
    """Asks several channels in order; the first real answer wins."""

    def __init__(self, confirmers: Iterable[Confirmer]) -> None:
        self.confirmers = [c for c in confirmers if c is not None]

    def __bool__(self) -> bool:
        return bool(self.confirmers)

    def __len__(self) -> int:
        return len(self.confirmers)

    def confirm(self, request: ConfirmRequest) -> ConfirmAnswer:
        if not self.confirmers:
            return ConfirmAnswer(False, note="no confirmer available (denied by default)")
        last = ConfirmAnswer(False, note="no confirmer answered")
        abstained: list[str] = []
        for confirmer in self.confirmers:
            if not getattr(confirmer, "available", True):
                continue
            try:
                answer = confirmer.confirm(request)
            except Exception as exc:  # a broken channel must never approve anything
                last = ConfirmAnswer(False, note=f"{confirmer.name} failed: {exc}")
                continue
            if answer.unavailable:
                # Remember *why* so the denial can explain itself to the user.
                if answer.note:
                    abstained.append(f"{confirmer.name}: {answer.note}")
            if answer.approved:
                answer.method = answer.method or confirmer.name
                return answer
            last = answer
            if answer.timed_out or answer.unavailable:
                continue  # this channel could not answer; try the next one
            if answer.method and answer.method != "none":
                return answer  # a clear "no" ends the chain
        if abstained:
            last.note = "; ".join([last.note] + abstained)
        return last


def _norm(text: str) -> str:
    """Normalise a spoken/typed confirmation for comparison."""
    return " ".join(str(text or "").lower().split()).strip(" .,!?\"'").strip()


#: Words speech-to-text adds or drops around punctuation and filenames. They
#: carry no meaning for *which action* is being approved, so they are ignored
#: when comparing a spoken repeat against the challenge. Extensions are NOT
#: in here: "notes.txt" and "notes.md" must stay different.
_SPEECH_NOISE = {
    "dot", "dash", "hyphen", "underscore", "slash", "equals", "period",
    "the", "a", "an", "to", "of", "for", "and", "then", "that", "this",
    "please", "garvis", "file", "folder", "path", "command", "ok", "okay",
}
_NEGATIONS = {"not", "dont", "don", "no", "never", "cancel", "abort", "stop"}


def _significant_tokens(text: str) -> set[str]:
    """Meaningful words of a phrase, with speech noise and punctuation removed."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return {token for token in cleaned.split() if token and token not in _SPEECH_NOISE}


def _is_negated(text: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    tokens = set(cleaned.split())
    if tokens & _NEGATIONS:
        return True
    return bool(re.search(r"\bdo\s+not\b|\bdon'?t\b|\bn't\b", str(text or ""), re.IGNORECASE))


class ConsoleConfirmer:
    """Reads the answer from the terminal.

    Two modes, deliberately:

    * **Interactive** (stdin is a TTY): blocks and waits as long as it takes.
      The user is sitting right there; a timeout would only risk a stale "yes"
      approving the *next* request. Ctrl+C denies.
    * **Non-interactive** (piped input, tests, cron): waits at most
      ``timeout_s`` on a worker thread, then denies and marks itself desynced
      for the rest of the run, because an abandoned ``input()`` thread would
      otherwise swallow a later keystroke.
    """

    name = "console"

    def __init__(
        self,
        accept_words: Iterable[str] | None = None,
        deny_words: Iterable[str] | None = None,
        interactive: bool | None = None,
    ) -> None:
        self.accept = {_norm(w) for w in (accept_words or ["yes", "y", "yeah", "yep", "ok", "okay", "go ahead", "do it"])}
        self.deny = {_norm(w) for w in (deny_words or ["no", "n", "nope", "stop", "cancel", "abort"])}
        self._desynced = False
        if interactive is None:
            try:
                interactive = os.isatty(0)
            except (ValueError, OSError):  # pragma: no cover
                interactive = False
        self.interactive = interactive

    def confirm(self, request: ConfirmRequest) -> ConfirmAnswer:
        if self._desynced:
            return ConfirmAnswer(False, method=self.name, note="console prompt timed out earlier; denying")

        exact = request.exact or request.summary
        print("\n" + "-" * 72)
        print(f"  {request.tier.upper()} ACTION NEEDS YOUR APPROVAL")
        print(f"  {request.summary}")
        for reason in request.reasons:
            print(f"    - why: {reason}")
        wait_note = "Ctrl+C to deny" if self.interactive else f"timeout {request.timeout_s:.0f}s"

        if request.stage == "repeat":
            print("\n  Step 1 of 2 - repeat this to show it is really you:")
            print(f"      {exact}")
            if request.challenge_spoken and _norm(request.challenge_spoken) != _norm(exact):
                print(f"      (or say: \"{request.challenge_spoken}\")")
            print(f"  ({wait_note})")
            print("-" * 72)
            text = self._ask("repeat> ", request.timeout_s)
        elif request.stage == "confirm":
            print(f"\n  Step 2 of 2 - type the word: {request.require_phrase}")
            print(f"  ({wait_note})")
            print("-" * 72)
            text = self._ask(f"{request.require_phrase}> ", request.timeout_s)
        else:
            print(f"\n  Approve? [y/N] ({wait_note})")
            print("-" * 72)
            text = self._ask("confirm> ", request.timeout_s)

        if text is None:
            return self._timeout(request)

        cleaned = _norm(text)
        if request.stage in ("repeat", "confirm"):
            # The gate compares the wording; here we only report what was said.
            if cleaned in self.deny or not cleaned:
                return ConfirmAnswer(False, method=self.name, text=text, note="declined")
            return ConfirmAnswer(True, method=self.name, text=text)

        if cleaned in self.accept:
            return ConfirmAnswer(True, method=self.name, text=text)
        if cleaned in self.deny or not cleaned:
            return ConfirmAnswer(False, method=self.name, text=text, note="declined")
        return ConfirmAnswer(False, method=self.name, text=text, note=f"'{text.strip()}' is not an approval")

    def _ask(self, prompt: str, timeout_s: float) -> str | None:
        """Read one line. None means timeout, Ctrl+C or EOF - all count as deny."""
        if self.interactive:
            try:
                return input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                return None

        result: queue.Queue[str | None] = queue.Queue(maxsize=2)

        def reader() -> None:
            try:
                result.put(input(prompt), block=False)
            except (EOFError, KeyboardInterrupt):
                try:
                    result.put(None, block=False)
                except queue.Full:
                    pass
            except Exception:
                pass

        thread = threading.Thread(target=reader, daemon=True, name="garvis-confirm-input")
        thread.start()
        try:
            return result.get(timeout=max(1.0, timeout_s))
        except queue.Empty:
            return None

    def _timeout(self, request: ConfirmRequest) -> ConfirmAnswer:
        if not self.interactive:
            self._desynced = True
        print("  No answer. Denied.")
        return ConfirmAnswer(False, method=self.name, timed_out=True, note="timeout")


class AutoDenyConfirmer:
    """Denies everything: unattended runs and tests where nothing may approve."""

    name = "auto-deny"

    def confirm(self, request: ConfirmRequest) -> ConfirmAnswer:
        return ConfirmAnswer(False, method=self.name, note="auto-deny mode: confirmation impossible")


class ScriptedConfirmer:
    """Scripted answers, for tests and the self-test routine.

    ``texts`` are what the simulated user "says", in order. The gate still
    applies its own rules, so a script that only says "yes" cannot approve a RED
    action - which is exactly what the tests assert.

    ``approve_all=True`` is the "cooperative user" shortcut: it answers with the
    correct challenge / confirm word for each stage.
    """

    name = "scripted"

    def __init__(
        self,
        texts: Iterable[str] | None = None,
        approve_all: bool = False,
        default_text: str | None = None,
    ) -> None:
        self.texts = list(texts or [])
        self.approve_all = approve_all
        self.default_text = default_text
        self.requests: list[ConfirmRequest] = []

    def confirm(self, request: ConfirmRequest) -> ConfirmAnswer:
        self.requests.append(request)
        if self.approve_all:
            if request.stage == "repeat":
                return ConfirmAnswer(True, method=self.name, text=request.challenge)
            if request.stage == "confirm":
                return ConfirmAnswer(True, method=self.name, text=request.require_phrase)
            return ConfirmAnswer(True, method=self.name, text="yes")
        if self.texts:
            return ConfirmAnswer(True, method=self.name, text=self.texts.pop(0))
        if self.default_text is not None:
            return ConfirmAnswer(True, method=self.name, text=self.default_text)
        return ConfirmAnswer(False, method=self.name, note="scripted: nothing left to say")


# ---------------------------------------------------------------------------
# Classification results
# ---------------------------------------------------------------------------
@dataclass
class Classification:
    tier: str
    reasons: list[str] = field(default_factory=list)
    blocked: bool = False          # no confirmation can approve this
    block_reason: str = ""
    #: True when the call looks like it carries a secret (a red keyword matched
    #: in the arguments). The gate then scrubs every arg value before logging.
    sensitive: bool = False

    def explain(self) -> str:
        bits = [f"tier={self.tier}"]
        if self.blocked:
            bits.append(f"blocked: {self.block_reason}")
        if self.reasons:
            bits.append("; ".join(self.reasons))
        return " | ".join(bits)


@dataclass
class GateOutcome:
    """What the gate did, and what the model is allowed to see."""

    ok: bool
    tool: str
    tier: str
    decision: str
    content: str = ""              # raw content for the model
    display: str | None = None
    duration_ms: float = 0.0
    attempts: int = 0
    verified: str | None = None
    for_model_content: str = ""    # fenced content actually sent to the model
    error: str | None = None
    #: True when the tool actually executed. A failure from a tool that ran can
    #: quote the outside world (a Playwright error quoting page HTML, a command's
    #: stderr), so it is fenced like any other third-party text. Refusals built by
    #: the gate itself (denied/blocked/aborted) stay unfenced: that text is ours.
    ran: bool = False

    def __post_init__(self) -> None:
        if not self.for_model_content:
            self.refresh_for_model()

    def refresh_for_model(self) -> None:
        """Rebuild what the model (and the UI event) sees from the current state."""
        if self.ok:
            self.for_model_content = safety.wrap_tool_result(self.tool, self.content)
            return
        detail = self.display or self.content or "no detail"
        if self.ran:
            # The tool ran, so this text may have come from a page, a command or
            # another program. Fence it: the model must not be able to mistake
            # "an error that mentions instructions" for the user talking to it.
            detail = safety.wrap_tool_result(self.tool, detail)
        self.for_model_content = (
            f"{self.decision.upper()}: {detail}\n"
            f"(tool={self.tool}, tier={self.tier})"
        )

    def mark_unverified(self) -> None:
        """Say plainly that nothing outside the tool could be re-checked."""
        self.content = (
            f"{self.content}\n{UNVERIFIED_NOTE}" if self.content else UNVERIFIED_NOTE
        )
        self.refresh_for_model()

    def apply_verification(self, note: str) -> None:
        """Fold a verification note into the outcome, honestly.

        Evidence ("notes.txt exists, 412 bytes") is added to the content so the
        model knows its work was checked. A failure ("VERIFY FAILED: ...") turns
        the outcome into a failure: the action was reported as done, the world
        says otherwise, and the user must hear that - not a success message.
        """
        self.verified = note
        if note.startswith(VERIFY_FAILED_PREFIX):
            self.ok = False
            self.error = note
            self.display = f"{note} (the tool said it worked)"
            self.content = (
                f"{note}\nThe tool reported success, but the result could not be found, "
                f"so this is being reported as a failure."
            )
            self.decision = FLAGGED
        else:
            self.content = f"{self.content}\n[checked] {note}" if self.content else f"[checked] {note}"
        self.refresh_for_model()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
class PermissionGate:
    """Classifies, confirms, executes, verifies and logs every tool call."""

    def __init__(
        self,
        cfg: Any,
        activity: Any = None,
        services: dict[str, Any] | None = None,
        log: Any = None,
        confirmer: Confirmer | None = None,
        registry: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.activity = activity
        self.services = services if services is not None else {}
        self.log = log
        self.registry = registry if registry is not None else self.services.get("registry")
        if self.registry is None:
            raise RuntimeError("PermissionGate needs a tool registry (pass registry= or services={'registry': ...})")

        self.default_tier = str(cfg.get("permissions.default_tier", RED)).lower()
        self.rules = self._compile_rules(cfg.get("permissions.rules", []) or [])
        self.red_keywords = [str(k).lower() for k in (cfg.get("permissions.red_keywords", []) or [])]
        self.yellow_cfg = cfg.section("permissions").get("yellow_confirm", {}) or {}
        self.red_cfg = cfg.section("permissions").get("red_confirm", {}) or {}
        self.denied_globs = [str(g) for g in (cfg.get("files.denied_globs", []) or [])]
        self._bare_protected = set(_BARE_PROTECTED_PATTERNS) | _bare_protected_patterns(self.denied_globs)
        self.allowed_read = cfg.allowed_folders("read")
        self.allowed_write = cfg.allowed_folders("write")
        self.follow_symlinks = bool(cfg.get("files.follow_symlinks", False))
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in (cfg.get("shell.blocked_patterns", []) or [])]
        self.sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir", "sandbox"))
        self.default_timeout = float(cfg.get("safety.tool_timeout_s", 60))
        self.max_retries = max(0, int(cfg.get("safety.tool_retries", 1)))
        self.max_consecutive_errors = int(cfg.get("safety.max_consecutive_errors", 3))
        self.note_unverified = bool(cfg.get("permissions.note_unverified", True))
        self.confirmer = confirmer if confirmer is not None else self.build_confirmer(cfg)

        self._consecutive_errors = 0
        self._lock = threading.RLock()
        self.stats = {ALLOWED: 0, CONFIRMED: 0, DENIED: 0, BLOCKED: 0, ERROR: 0,
                      TIMEOUT: 0, ABORTED: 0, STOPPED: 0}

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def _compile_rules(raw: Iterable[Any]) -> list[tuple[str, re.Pattern[str]]]:
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for rule in raw:
            if not isinstance(rule, dict):
                continue
            tier = str(rule.get("tier", "")).lower()
            pattern = str(rule.get("match", ""))
            if tier in _TIER_RANK and pattern:
                try:
                    compiled.append((tier, re.compile(pattern)))
                except re.error:
                    continue
        return compiled

    @staticmethod
    def build_confirmer(cfg: Any) -> Confirmer:
        """Console confirmer by default; later stages prepend voice/UI channels.

        ``safety.require_confirmation: false`` in config swaps in the auto-deny
        confirmer for unattended runs (nothing YELLOW/RED can ever be approved).
        """
        if str(cfg.get("permissions.yellow_confirm.mode", "spoken")).lower() == "none":
            return AutoDenyConfirmer()
        return ConfirmerChain([ConsoleConfirmer(
            accept_words=[str(cfg.get("permissions.yellow_confirm.phrase", "yes"))]
            + [str(w) for w in (cfg.get("permissions.yellow_confirm.also_accept", []) or [])],
        )])

    def add_confirmer(self, confirmer: Confirmer, first: bool = True) -> None:
        """Register another channel (voice, UI). Later stages call this."""
        existing = list(getattr(self.confirmer, "confirmers", [])) or (
            [self.confirmer] if self.confirmer else []
        )
        if isinstance(self.confirmer, ConfirmerChain):
            self.confirmer.confirmers.insert(0 if first else len(self.confirmer.confirmers), confirmer)
        else:
            self.confirmer = ConfirmerChain(
                [confirmer, self.confirmer] if first else [self.confirmer, confirmer]
            )
        self.log_info("added %s confirmation channel", confirmer.name)

    def _log_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Arguments as they may be written down: declared secrets are hidden.

        ``secret_args`` (e.g. the text about to be typed into a web field) and
        ``log_omit`` are replaced by "[hidden]". Nothing else is touched, so the
        activity log stays useful for "what did you do today".
        """
        tool = self.registry.get(tool_name) if self.registry is not None else None
        hidden = set(getattr(tool, "log_omit", ()) or ()) | set(getattr(tool, "secret_args", ()) or ())
        if not hidden or not args:
            return args
        return {key: ("[hidden]" if key in hidden else value) for key, value in args.items()}

    # -- logging shims -----------------------------------------------------
    def log_info(self, message: str, *args: Any) -> None:
        if self.log:
            getattr(self.log, "info", lambda *_: None)(message, *args)

    def log_debug(self, message: str, *args: Any) -> None:
        if self.log:
            getattr(self.log, "debug", lambda *_: None)(message, *args)

    def log_warning(self, message: str, *args: Any) -> None:
        if self.log:
            getattr(self.log, "warning", lambda *_: None)(message, *args)

    def _record(self, **kwargs: Any) -> None:
        if self.activity is None:
            return
        try:
            self.activity.permission(**kwargs)
        except TypeError:
            self.activity.event("permission", str(kwargs))

    # -- classification ----------------------------------------------------
    def classify(self, tool_name: str, args: dict[str, Any]) -> Classification:
        """Decide the tier for one call. Never demotes, only promotes."""
        tool = self.registry.get(tool_name) if self.registry is not None else None
        if tool is None:
            return Classification(RED, blocked=True, block_reason=f"unknown tool '{tool_name}'")

        tier = tool.tier if tool.tier in _TIER_RANK else self.default_tier
        reasons: list[str] = []

        for rule_tier, pattern in self.rules:
            if pattern.search(tool_name):
                if _TIER_RANK[rule_tier] > _TIER_RANK[tier]:
                    tier = rule_tier
                    reasons.append(f"config rule promoted to {rule_tier}")
                break

        # A red keyword anywhere (tool name or arguments) forces RED. Only a
        # keyword found in the *arguments* marks the call as secret-bearing:
        # "files.delete(path=...)" is RED but the path is not a secret.
        sensitive = False
        name_haystack = tool_name.lower()
        args_haystack = _flatten(args).lower()
        for keyword in self.red_keywords:
            if not keyword:
                continue
            in_name = keyword in name_haystack
            in_args = keyword in args_haystack
            if in_name or in_args:
                tier = promote(tier, RED)
                reasons.append(f"red keyword '{keyword}'" + (" (in arguments)" if in_args else " (in tool name)"))
                sensitive = in_args
                break

        # Paths, URLs and commands declared by the tool.
        path_check = self._check_paths(tool, args)
        if path_check:
            tier = promote(tier, path_check.tier)
            if path_check.blocked:
                return Classification(
                    path_check.tier, path_check.reasons, True, path_check.block_reason, sensitive=sensitive
                )
            reasons.extend(path_check.reasons)

        url_check = self._check_urls(tool, args)
        if url_check:
            tier = promote(tier, url_check.tier)
            if url_check.blocked:
                return Classification(
                    url_check.tier, url_check.reasons, True, url_check.block_reason, sensitive=sensitive
                )
            reasons.extend(url_check.reasons)

        command_check = self._check_command(tool, args)
        if command_check:
            tier = promote(tier, command_check.tier)
            if command_check.blocked:
                return Classification(
                    command_check.tier, command_check.reasons, True, command_check.block_reason, sensitive=sensitive
                )
            reasons.extend(command_check.reasons)

        # Sending content to a third party is never GREEN.
        for arg_name in tool.content_args:
            value = args.get(arg_name)
            if value:
                reasons.append(f"'{arg_name}' sends content to a third party")
                tier = promote(tier, YELLOW)

        if tool.guard is not None:
            try:
                guard_tier, guard_reason = tool.guard(args)
            except Exception as exc:  # a broken guard must fail closed
                return Classification(
                    RED, reasons, blocked=True,
                    block_reason=f"tool guard failed: {exc}", sensitive=sensitive,
                )
            if guard_reason:
                reasons.append(guard_reason)
            if guard_tier == "blocked":
                return Classification(
                    RED, reasons, True, guard_reason or "blocked by tool guard", sensitive=sensitive
                )
            if guard_tier:
                tier = promote(tier, guard_tier)

        return Classification(tier, reasons, sensitive=sensitive)

    def _check_paths(self, tool: Any, args: dict[str, Any]) -> Classification | None:
        paths: list[str] = []
        for arg_name in getattr(tool, "path_args", ()) or ():
            value = args.get(arg_name)
            if isinstance(value, (list, tuple)):
                paths.extend(str(v) for v in value if v)
            elif value:
                paths.append(str(value))
        if not paths:
            return None

        writes = _TIER_RANK[tool.tier] >= _TIER_RANK[YELLOW] or tool.name.split(".")[-1] in (
            "write", "append", "move", "copy", "delete", "mkdir", "create",
        )
        allowed = self.allowed_write if writes else self.allowed_read
        tier = tool.tier
        reasons: list[str] = []
        for raw in paths:
            resolved = self.resolve_for(tool, raw)
            if self.matches_denied_glob(resolved):
                return Classification(
                    RED,
                    reasons,
                    True,
                    f"path matches a protected pattern (secrets/config): {resolved}",
                )
            if not self.within(resolved, allowed):
                where = "writable" if writes else "readable"
                allowed_text = ", ".join(str(p) for p in allowed) or "(none configured)"
                return Classification(
                    RED,
                    reasons,
                    True,
                    f"path is outside the {where} folders ({allowed_text}): {resolved}",
                )
        if tool.tier == GREEN and writes:
            tier = promote(tier, YELLOW)
            reasons.append("writes to an allowed folder")
        return Classification(tier, reasons)

    def _check_urls(self, tool: Any, args: dict[str, Any]) -> Classification | None:
        urls: list[str] = []
        for arg_name in getattr(tool, "url_args", ()) or ():
            value = args.get(arg_name)
            if value:
                urls.append(str(value))
        if not urls:
            return None
        allowed_sites = self.cfg.allowed_sites()
        blocked_sites = self.cfg.blocked_sites()
        tier = tool.tier
        reasons: list[str] = []
        for url in urls:
            domain = domain_of(url)
            if not domain:
                return Classification(RED, reasons, True, f"could not parse a host from '{url}'")
            if any(domain == b or domain.endswith("." + b) for b in blocked_sites):
                return Classification(
                    RED, reasons, True,
                    f"{domain} is on the never-automate list (banking/payments). "
                    f"Open it yourself; GARVIS will not touch it.",
                )
            if not any(domain == a or domain.endswith("." + a) for a in allowed_sites):
                return Classification(
                    RED, reasons, True,
                    f"{domain} is not in browser.allowed_sites. Add it to config.yaml if you want "
                    f"GARVIS to go there.",
                )
            if domain not in reasons:
                reasons.append(f"site allowed: {domain}")
        return Classification(tier, reasons)

    def _check_command(self, tool: Any, args: dict[str, Any]) -> Classification | None:
        """Validate a command line before it can run.

        Checks, in order: blocked patterns, hidden substitutions, then the
        allowlist for the first token of *every* segment (so ``echo ok; rm -rf x``
        cannot pass on the strength of ``echo``). Redirection targets must be
        inside a write-allowed folder.
        """
        commands: list[str] = []
        for arg_name in getattr(tool, "command_args", ()) or ():
            value = args.get(arg_name)
            if value:
                commands.append(str(value))
        if not commands:
            return None

        tier = tool.tier
        reasons: list[str] = []
        allowlist = [a.lower() for a in self.cfg.shell_allowlist()]
        require_allowlist = bool(self.cfg.get("shell.require_allowlist", True))

        for command in commands:
            for pattern in self.blocked_patterns:
                if pattern.search(command):
                    return Classification(
                        RED, reasons, True,
                        f"command matches a blocked pattern ({pattern.pattern}); this never runs.",
                    )

            protected = self._protected_token(command)
            if protected:
                return Classification(RED, reasons, True, protected)

            if not bool(self.cfg.get("shell.allow_substitution", False)):
                for token in ("$(", "`", "${"):
                    if token in command:
                        return Classification(
                            RED, reasons, True,
                            f"command contains '{token}' (command substitution), which is disabled: "
                            f"it could hide a command that is not on the allowlist. "
                            f"Set shell.allow_substitution: true if you really want it.",
                        )

            if not bool(self.cfg.get("shell.allow_redirects", False)):
                for target in _redirect_targets(command):
                    resolved = self.resolve_user_path(target)
                    if not self.within(resolved, self.allowed_write):
                        return Classification(
                            RED, reasons, True,
                            f"output redirection to '{target}' is outside the writable folders "
                            f"({resolved}); disabled unless shell.allow_redirects: true.",
                        )

            segments = split_command_segments(command)
            if not segments:
                return Classification(RED, reasons, True, "empty command")
            for segment in segments:
                first = _first_token(segment)
                if not first:
                    continue
                # Strip a leading VAR=value assignment: the real command is next.
                while "=" in first and not first.startswith("-") and "/" not in first:
                    parts = segment.split(None, 1)
                    segment = parts[1] if len(parts) > 1 else ""
                    first = _first_token(segment)
                    if not first:
                        break
                if not first:
                    continue
                if require_allowlist:
                    if not allowlist:
                        return Classification(RED, reasons, True, "shell allowlist is empty; running nothing")
                    base = os.path.basename(first).lower()
                    if base.endswith(".exe"):
                        base = base[:-4]
                    if base not in allowlist and first.lower() not in allowlist:
                        return Classification(
                            RED, reasons, True,
                            f"'{first}' is not in shell.allowlist. Add it in config.yaml if you want it "
                            f"runnable, or run it yourself.",
                        )
                reasons.append(f"runs a command: {first}")
            tier = promote(tier, YELLOW)

        return Classification(tier, reasons)

    # -- path helpers ------------------------------------------------------
    def base_for(self, tool: Any = None) -> Path | None:
        """Where a tool resolves relative paths; None = the shell's own rule.

        A tool states this with ``Tool.path_base``. It matters: `files.write`
        puts "notes.txt" in the sandbox, while `shell.run` runs it in
        shell.default_cwd. Checking (and verifying) the wrong one means the
        gate inspects a different file than the tool touched.
        """
        if getattr(tool, "path_base", "") == "sandbox":
            return self._sandbox_dir()
        return None

    def _sandbox_dir(self) -> Path:
        return self.cfg.resolve_path(self.cfg.get("files.sandbox_dir", "sandbox"))

    def resolve_for(self, tool: Any, raw: str) -> Path:
        return self.resolve_user_path(raw, base=self.base_for(tool))

    def resolve_user_path(self, raw: str, base: Path | str | None = None) -> Path:
        text = str(raw).strip().strip('"').strip("'")
        path = Path(os.path.expandvars(os.path.expanduser(text)))
        if not path.is_absolute():
            if base is not None:
                path = Path(base) / path
            else:
                # Relative paths resolve inside the sandbox by default; when the
                # shell has its own working folder, that is the shell's rule.
                folder = self._sandbox_dir()
                shell_cwd = self.cfg.get("shell.default_cwd")
                if shell_cwd:
                    folder = self.cfg.resolve_path(shell_cwd)
                path = folder / path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = Path(os.path.abspath(str(path)))
        if not self.follow_symlinks:
            try:
                resolved = Path(os.path.realpath(resolved))
            except OSError:
                pass
        return resolved

    def within(self, path: Path, folders: Iterable[Path]) -> bool:
        for folder in folders:
            try:
                real_folder = Path(os.path.realpath(folder)) if not self.follow_symlinks else folder
            except OSError:
                real_folder = folder
            if path == real_folder or real_folder in path.parents:
                return True
        return False

    def matches_denied_glob(self, path: Path) -> bool:
        text = str(path)
        name = path.name
        for pattern in self.denied_globs:
            if fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(name, pattern):
                return True
            # "**/x" also means "anywhere/x"; fnmatch needs the full string.
            simplified = pattern.replace("**/", "")
            if fnmatch.fnmatch(name, simplified) or fnmatch.fnmatch(text, f"*{simplified}"):
                return True
        return False

    def _protected_token(self, command: str) -> str | None:
        """Refuse a command whose *arguments* name a protected file.

        The file tools have every path checked against files.denied_globs; the
        shell checked only the executable, so `cat ~/.ssh/id_rsa` walked past the
        allowlists and every protected pattern in one step. Two rules close that:

        * a **path-shaped** argument (`~/.ssh/id_rsa`, `./config.yaml`,
          `/home/x/.aws/credentials`, the value side of `--file=...`) is run
          through the same matcher the file tools use;
        * a **file name** from the built-in list (`id_rsa`, `.env`, `config.yaml`,
          `Login Data`, `Cookies`, ...) or matching a built-in/config extension
          pattern (`*.key`, `*.pem`) is refused wherever it appears.

        A bare word is only treated as a file name when it is on one of those
        lists - `grep -rn password src/` is prose and keeps working. What this is
        *not* is containment: an allowlisted interpreter (`python3 -c`,
        `node -e`) can reach anything the user's account can, whatever token
        checks say. The README says so, config.yaml says so beside the allowlist,
        and --check warns about it, because pretending otherwise would be worse
        than the gap.
        """
        for token in _command_tokens(command):
            text = token.strip().strip("\"'")
            if not text:
                continue
            if "=" in text:
                # --file=/path, --output=x, and VAR=value prefixes: judge the value.
                text = text.split("=", 1)[1].strip().strip("\"'")
            if not text or text.startswith("-"):
                continue

            path_shape = text.startswith(("/", "~", ".")) or "/" in text or "\\" in text
            if path_shape and (self.matches_denied_glob(Path(text))
                               or self.matches_denied_glob(self.resolve_user_path(text))):
                return self._protected_reason(text, "a protected pattern")

            name = os.path.basename(text)
            if name and (name in _BARE_PROTECTED_TOKENS
                         or any(fnmatch.fnmatch(name, pattern) for pattern in self._bare_protected)):
                return self._protected_reason(name, "a protected file name")
        return None

    @staticmethod
    def _protected_reason(what: str, kind: str) -> str:
        return (
            f"the command names {what!r}, which is {kind} (keys, browser cookie stores, "
            f"GARVIS's own files). Reading those is off limits to every tool, including the "
            f"shell; this never runs."
        )

    # -- the breaker -------------------------------------------------------
    def breaker_tripped(self) -> bool:
        """True when repeated failures have stopped tool use."""
        return self._consecutive_errors >= self.max_consecutive_errors

    def reset_breaker(self, reason: str = "the user asked to carry on") -> bool:
        """Clear the consecutive-failure breaker. Returns True if it was set.

        Only a *user* action calls this (main.py, at the start of a message the
        user typed or said). The model cannot: no tool reaches this method, and
        clearing it automatically on a model-side retry would defeat the breaker
        - which exists to stop a model that keeps failing from hammering away.

        This is also the only way out. The counter used to zero itself on a
        successful call, but a tripped breaker refuses every call before it can
        run, so nothing could ever succeed: three transient failures bricked the
        gate for the rest of the session while telling the user it was "waiting"
        for something they had no way to do.
        """
        was_tripped = self._consecutive_errors > 0
        self._consecutive_errors = 0
        if was_tripped:
            self.log_warning("failure breaker cleared (%s)", reason)
            self._record(tool="(breaker)", args={}, tier=GREEN, decision="reset",
                         note=f"cleared after {self.max_consecutive_errors} consecutive failures: {reason}")
            if self.activity is not None:
                try:
                    self.activity.event("system", f"failure breaker cleared: {reason}",
                                        extra={"decision": "reset"})
                except Exception:
                    pass
        return was_tripped

    # -- execution ---------------------------------------------------------
    def execute(self, tool_name: str, args: dict[str, Any] | None = None) -> GateOutcome:
        """The single entry point for running a tool."""
        args = dict(args or {})
        started = time.perf_counter()

        with self._lock:
            switch = self.services.get("killswitch")
            if switch is not None and getattr(switch, "frozen", False):
                # "stop everything" has to mean *everything*, and this is the one
                # door every tool call goes through. handle_line already refuses
                # to send a frozen session's text to the model, so this is the
                # second lock on the same door: a UI button, a worker thread or a
                # future caller cannot quietly run a tool after a stop.
                outcome = GateOutcome(
                    ok=False, tool=tool_name, tier=RED, decision=STOPPED,
                    content="Stopped: the kill switch is engaged, so nothing runs. Tell the "
                            "user the stop is still on and that saying 'resume' ends it - do "
                            "not retry, and do not look for another route.",
                    display="stopped by the kill switch",
                )
                outcome.duration_ms = (time.perf_counter() - started) * 1000
                self._record(tool=tool_name, args=self._log_args(tool_name, args), tier=RED,
                             decision=STOPPED, note="the kill switch is engaged")
                self.stats[STOPPED] = self.stats.get(STOPPED, 0) + 1
                self._notify("I am stopped. Say 'resume' when you want me to carry on.")
                return outcome

            if self._consecutive_errors >= self.max_consecutive_errors:
                outcome = GateOutcome(
                    ok=False, tool=tool_name, tier=RED, decision=ABORTED,
                    content=f"Too many consecutive failures ({self._consecutive_errors}). "
                            f"Stopping here; nothing else will run this turn. Tell the user what "
                            f"failed and that anything they say next clears this.",
                    display="aborted after repeated failures",
                )
                outcome.duration_ms = (time.perf_counter() - started) * 1000
                self._record(tool=tool_name, args=self._log_args(tool_name, args), tier=RED,
                             decision=ABORTED, note="consecutive failure limit reached")
                self.stats[ABORTED] += 1
                self._notify("I have failed several times in a row, so I stopped. "
                             "Anything you say next lets me try again.")
                return outcome

        tool = self.registry.get(tool_name)
        if tool is None:
            outcome = self._deny(tool_name, args, RED, BLOCKED, f"unknown tool '{tool_name}'", started)
            return outcome

        classification = self.classify(tool_name, args)
        tier = classification.tier
        sensitive = classification.sensitive
        self.log_debug("classify %s -> %s", tool_name, classification.explain())

        if classification.blocked:
            return self._deny(
                tool_name, args, tier, BLOCKED,
                f"{classification.block_reason}"
                + (" (" + "; ".join(classification.reasons) + ")" if classification.reasons else ""),
                started,
                sensitive=sensitive,
            )

        # Confirmation, if the tier requires it.
        if tier == YELLOW:
            answer = self._confirm(tool, args, tier, classification.reasons, sensitive=sensitive)
            if not answer.approved:
                return self._deny(
                    tool_name, args, tier, DENIED,
                    f"the user did not approve this {tier.upper()} action"
                    + (f" ({answer.note})" if answer.note else "")
                    + (". Do not retry it; ask again only if the situation changed." if answer.timed_out is False else
                       ". The request timed out; ask again later if needed."),
                    started,
                    sensitive=sensitive,
                )
        elif tier == RED:
            answer = self._confirm(tool, args, tier, classification.reasons, sensitive=sensitive)
            if not answer.approved:
                return self._deny(
                    tool_name, args, tier, DENIED,
                    f"the user did not complete the {tier.upper()} confirmation"
                    + (f" ({answer.note})" if answer.note else "")
                    + ". Do not retry, do not look for another route. Report and stop.",
                    started,
                    sensitive=sensitive,
                )

        # Run it.
        outcome = self._run_tool(tool, args, tier, started)
        outcome.tier = tier
        outcome.duration_ms = (time.perf_counter() - started) * 1000

        with self._lock:
            if outcome.ok:
                self._consecutive_errors = 0
            elif outcome.decision in (ERROR, TIMEOUT):
                self._consecutive_errors += 1

        self.stats[outcome.decision] = self.stats.get(outcome.decision, 0) + 1
        # GREEN runs get a permission record here; YELLOW/RED already have their
        # asked/confirmed/denied records from the confirmation step.
        if tier == GREEN:
            self._record(
                tool=tool_name, args=self._log_args(tool_name, args), tier=tier,
                decision=outcome.decision,
                note=(outcome.verified or (outcome.error or "")[:200] or ""),
                sensitive=sensitive,
            )
        if self.activity is not None:
            try:
                self.activity.tool_call(
                    tool_name, self._log_args(tool_name, args),
                    result=(outcome.content or "")[:1500],
                    ok=outcome.ok,
                    duration_ms=outcome.duration_ms,
                    tier=tier,
                    sensitive=sensitive,
                )
            except Exception:
                pass

        if not outcome.ok:
            self._notify(f"{tool_name} failed: {(outcome.error or '')[:200]}")
        return outcome

    def _run_tool(self, tool: Any, args: dict[str, Any], tier: str, started: float) -> GateOutcome:
        timeout = float(tool.timeout_s or self.default_timeout)
        before = self._snapshot_state(tool, args)
        attempts = 0
        last_error: str | None = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = self._call_with_timeout(tool, args, timeout)
            except TimeoutError:
                outcome = GateOutcome(
                    ok=False, tool=tool.name, tier=tier, decision=TIMEOUT,
                    content=f"'{tool.name}' timed out after {timeout:.0f}s and was abandoned.",
                    display=f"timeout after {timeout:.0f}s",
                    error=f"timeout after {timeout:.0f}s",
                    ran=True,   # it started; partial output may exist
                )
                outcome.attempts = attempts
                outcome.duration_ms = (time.perf_counter() - started) * 1000
                return outcome
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.log_warning("%s attempt %d failed: %s", tool.name, attempts, last_error)
                if attempts <= self.max_retries:
                    time.sleep(0.4)  # one retry, then report honestly
                    continue
                outcome = GateOutcome(
                    ok=False, tool=tool.name, tier=tier, decision=ERROR,
                    content=f"'{tool.name}' failed: {last_error}",
                    display=last_error,
                    error=last_error,
                    ran=True,   # the exception text can quote whatever the tool talked to
                )
                outcome.attempts = attempts
                return outcome

            # Success path (a ToolResult with ok=False is still a "ran fine, failed" case).
            outcome = GateOutcome(
                ok=bool(result.ok),
                tool=tool.name,
                tier=tier,
                decision=CONFIRMED if tier in (YELLOW, RED) else ALLOWED,
                content=result.content,
                display=result.display,
                error=result.error,
                ran=True,
            )
            outcome.attempts = attempts
            if not result.ok:
                self.log_warning("%s returned a failure: %s", tool.name, result.error or result.content[:200])
            if result.ok and not tool.readonly:
                note = self._verify(tool, args, result, before)
                if note:
                    outcome.apply_verification(note)
                    if not outcome.ok:
                        self.log_warning("verification failed for %s: %s", tool.name, note)
                elif self.note_unverified:
                    outcome.mark_unverified()
            return outcome
        # Unreachable, but keeps the type checker honest.
        return GateOutcome(ok=False, tool=tool.name, tier=tier, decision=ERROR,
                           content=f"'{tool.name}' failed: {last_error}", error=last_error)

    def _call_with_timeout(self, tool: Any, args: dict[str, Any], timeout: float) -> Any:
        """Run a tool with a hard deadline.

        A hung tool cannot be killed in Python, so the worker is a daemon thread
        that we abandon: the caller is freed immediately and the app stays
        responsive. The log records the abandonment so a stuck tool is visible.
        """
        box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                box.put((True, tool.run(args)))
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                box.put((False, exc))

        thread = threading.Thread(target=worker, daemon=True, name=f"garvis-tool-{tool.name}")
        thread.start()
        try:
            ok, value = box.get(timeout=max(0.1, timeout))
        except queue.Empty:
            self.log_warning(
                "tool %s exceeded %.0fs; abandoning the worker thread", tool.name, timeout
            )
            raise TimeoutError(f"{tool.name} timed out after {timeout:.0f}s") from None
        if not ok:
            raise value
        return value

    # -- verification (requirement 6) --------------------------------------
    #
    # What "verified" means here, exactly: after a successful state-changing
    # call, check the end state and say what was found. Whatever the check
    # reports goes into the answer the model and the user see, and a check that
    # comes back negative turns the call into a failure - a tool must not be
    # able to claim it moved a file that is not there.
    #
    # What it does not mean: it is not a proof that the change was the right
    # one, and it cannot see inside another program. Files are checked by
    # re-stating them; browser actions get a screenshot through
    # ``services['verifier']`` (``core.browser.after_action_note``, registered in
    # main.py), and screen-control tools take their own screenshot in stage 7.
    def _verify(self, tool: Any, args: dict[str, Any], result: Any = None,
                before: dict[str, bool] | None = None) -> str | None:
        """Check that a state-changing action actually did what it claimed."""
        failure = self._verify_expected_state(tool, args, before or {})
        if failure:
            return failure
        notes = self._verify_notes(tool, args)
        if not notes:
            # Nothing to re-read on disk: this is where a hook can look instead
            # (browser actions, through services['verifier']).
            verifier = self.services.get("verifier")
            if verifier is not None:
                try:
                    extra = verifier(tool.name, args)
                    if extra:
                        notes.append(str(extra))
                except Exception as exc:  # verification problems must not crash the turn
                    notes.append(f"verification hook failed: {exc}")
        return "; ".join(notes) if notes else None

    def _paths(self, tool: Any, args: dict[str, Any]) -> list[Path]:
        """The paths a tool's own declaration says it touched."""
        paths: list[Path] = []
        for arg_name in getattr(tool, "path_args", ()) or ():
            raw = args.get(arg_name)
            if not raw:
                continue
            try:
                paths.append(self.resolve_for(tool, str(raw)))
            except Exception:
                continue
        return paths

    def _snapshot_state(self, tool: Any, args: dict[str, Any]) -> dict[str, bool]:
        """What the world looks like *before* the call.

        Destructive verbs need this: "the file is gone" proves nothing if it was
        never there, and a tool that says it deleted something nonexistent has
        told you nothing about the world.
        """
        if tool.name not in _NEEDS_BEFORE:
            return {}
        return {str(path): path.exists() for path in self._paths(tool, args)}

    def _verify_expected_state(self, tool: Any, args: dict[str, Any],
                              before: dict[str, bool]) -> str | None:
        """Compare the world with what the tool just claimed. None = as expected."""
        name = tool.name
        paths = self._paths(tool, args)

        def missing(path: Path, verb: str) -> str:
            return f"{VERIFY_FAILED_PREFIX} {path} is not there after the {verb}"

        if name.startswith(("files.write", "files.append")) or name == "files.write":
            for path in paths:
                if not path.exists():
                    return missing(path, "write")
        elif name == "files.mkdir":
            for path in paths:
                if not path.is_dir():
                    return missing(path, "mkdir")
        elif name == "files.copy":
            if len(paths) >= 2:
                source, destination = paths[0], paths[-1]
                if not before.get(str(source), False):
                    return (f"{VERIFY_FAILED_PREFIX} {source} was not there before the copy, "
                            f"so there was nothing to copy")
                if destination.is_dir():
                    destination = destination / source.name  # copies into folders, like the tool does
                if not destination.exists():
                    return missing(destination, "copy")
        elif name == "files.move":
            if len(paths) >= 2:
                source, destination = paths[0], paths[-1]
                if not before.get(str(source), False):
                    return (f"{VERIFY_FAILED_PREFIX} {source} was not there before the move, "
                            f"so there was nothing to move")
                if not destination.exists():
                    return missing(destination, "move")
                if source.exists():
                    same = False
                    try:
                        same = destination.samefile(source)  # case-only rename on a lenient filesystem
                    except OSError:
                        pass
                    if not same:
                        return (f"{VERIFY_FAILED_PREFIX} {source} is still there after the move "
                                f"to {destination}")
        elif name == "files.delete":
            for path in paths:
                if path.exists():
                    return f"{VERIFY_FAILED_PREFIX} {path} is still there after the delete"
                if not before.get(str(path), False):
                    return (f"{VERIFY_FAILED_PREFIX} {path} was not there before the delete, "
                            f"so nothing was deleted")
        return None

    def _verify_notes(self, tool: Any, args: dict[str, Any]) -> list[str]:
        """Positive evidence, so the log and the model can see the check happened."""
        name = tool.name
        paths = self._paths(tool, args)
        notes: list[str] = []
        if name.startswith(("files.write", "files.append")):
            for path in paths:
                try:
                    notes.append(f"{path.name} exists ({path.stat().st_size} bytes)")
                except OSError:
                    pass
        elif name == "files.mkdir":
            for path in paths:
                if path.is_dir():
                    notes.append(f"{path.name} exists")
        elif name == "files.copy":
            if len(paths) >= 2:
                destination = paths[-1] / paths[0].name if paths[-1].is_dir() else paths[-1]
                try:
                    notes.append(f"{destination.name} exists ({destination.stat().st_size} bytes)")
                except OSError:
                    pass
        elif name == "files.move":
            if len(paths) >= 2:
                notes.append(f"{paths[-1].name} exists; {paths[0].name} is gone")
        elif name == "files.delete":
            for path in paths:
                notes.append(f"{path.name} is gone")
        return notes

    # -- helpers -----------------------------------------------------------
    def _confirm(
        self, tool: Any, args: dict[str, Any], tier: str, reasons: list[str], sensitive: bool = False
    ) -> ConfirmAnswer:
        """Run the confirmation protocol for this tier.

        YELLOW: one question, any "yes" from any channel.
        RED: two stages - repeat the exact action, then say the confirm word.
        The protocol is implemented HERE so that no confirmation channel can
        shortcut it.
        """
        # For a call flagged as secret-bearing, the summary and the RED challenge
        # use masked arguments: the user must still repeat the *action*, but the
        # password never appears on screen, in the log, or in the speaker.
        display_args = mask_secrets(args) if sensitive else args
        summary = f"{tool.name}: {tool.arg_summary(display_args)}"
        exact = _exact_action(tool, display_args)
        if tier == RED:
            return self._confirm_red(tool, args, summary, exact, reasons, sensitive=sensitive)
        # Yellow: ask out loud in words, not in machine syntax. The console (and
        # the UI) still show the exact arguments for a precise decision.
        human = _spoken_challenge(tool, display_args)
        self._spoken_summary = human

        cfg = self.yellow_cfg
        request = ConfirmRequest(
            tier=tier,
            tool=tool.name,
            summary=summary,
            exact=exact,
            reasons=reasons,
            timeout_s=float(cfg.get("timeout_s", 20)),
            mode=str(cfg.get("mode", "spoken")),
            stage="single",
        )
        self._record(
            tool=tool.name, args=self._log_args(tool.name, args), tier=tier, decision="asked",
            note=" | ".join(reasons) if reasons else "", sensitive=sensitive,
        )
        self._notify(f"About to {human}. Say yes to approve.")
        if self.confirmer is None:
            return ConfirmAnswer(False, note="no confirmation channel is configured")
        return self.confirmer.confirm(request)

    def _confirm_red(
        self, tool: Any, args: dict[str, Any], summary: str, exact: str, reasons: list[str],
        sensitive: bool = False,
    ) -> ConfirmAnswer:
        """RED: the user must reproduce the exact action, then say the magic word."""
        if self.confirmer is None:
            return ConfirmAnswer(False, note="no confirmation channel is configured")

        cfg = self.red_cfg
        timeout_s = float(cfg.get("timeout_s", 45))
        require_phrase = str(cfg.get("require_phrase", "confirm"))
        require_repeat = bool(cfg.get("require_repeat", True))
        base_reasons = list(reasons) + [
            "RED actions are irreversible, cost money, or change security/admin settings"
        ]

        method = "none"
        if require_repeat:
            step1 = ConfirmRequest(
                tier=RED, tool=tool.name, summary=summary, exact=exact,
                reasons=base_reasons, timeout_s=timeout_s,
                mode=str(cfg.get("mode", "both")), stage="repeat", require_phrase=require_phrase,
                sensitive=sensitive,
            )
            step1.challenge_spoken = _spoken_challenge(tool, args)
            self._record(tool=tool.name, args=self._log_args(tool.name, args), tier=RED, decision="asked",
                         note="RED step 1/2: repeat the exact action", sensitive=sensitive)
            self._notify(
                f"This is a red action: {step1.challenge_spoken}. "
                f"Step one, repeat that back to me word for word."
            )
            answer1 = self.confirmer.confirm(step1)
            method = answer1.method or "none"
            if not answer1.approved:
                # Carry the channel's own explanation through, so the user is told
                # *why* (e.g. "this action carries a secret; confirm on screen").
                detail = f" ({answer1.note})" if answer1.note else ""
                return ConfirmAnswer(
                    False, method=method, text=answer1.text, timed_out=answer1.timed_out,
                    note=f"RED step 1: no answer / refused{detail}",
                )
            matched, why_not = step1.match_repeat(answer1.text)
            if not matched:
                self._record(tool=tool.name, args=self._log_args(tool.name, args), tier=RED, decision=DENIED,
                             note=f"RED step 1: repeated wording did not match ({why_not})",
                             sensitive=sensitive)
                return ConfirmAnswer(
                    False, method=method, text=answer1.text,
                    note=f"RED step 1: {why_not} (heard: '{answer1.text[:60]}')",
                )

        step2 = ConfirmRequest(
            tier=RED, tool=tool.name, summary=summary, exact=exact,
            reasons=base_reasons, timeout_s=timeout_s,
            mode=str(cfg.get("mode", "both")), stage="confirm", require_phrase=require_phrase,
            sensitive=sensitive,
        )
        self._record(tool=tool.name, args=self._log_args(tool.name, args), tier=RED, decision="asked",
                     note="RED step 2/2: say the confirmation word", sensitive=sensitive)
        self._notify(
            f"Step two: say the word {require_phrase} if you really want this."
            if require_repeat else
            f"To approve, say the word {require_phrase}."
        )
        answer2 = self.confirmer.confirm(step2)
        method = f"{method}+{answer2.method}" if method not in ("none", answer2.method) else answer2.method
        if not answer2.approved:
            detail = f" ({answer2.note})" if answer2.note else ""
            return ConfirmAnswer(
                False, method=method, text=answer2.text, timed_out=answer2.timed_out,
                note=f"RED step 2: no answer / refused{detail}",
            )
        if _norm(answer2.text) != _norm(require_phrase):
            self._record(tool=tool.name, args=self._log_args(tool.name, args), tier=RED, decision=DENIED,
                         note="RED step 2: confirmation word missing", sensitive=sensitive)
            return ConfirmAnswer(
                False, method=method, text=answer2.text,
                note=f"RED step 2: the word '{require_phrase}' was not given",
            )

        self._record(tool=tool.name, args=self._log_args(tool.name, args), tier=RED, decision=CONFIRMED,
                     note="exact action repeated and confirmation word given", sensitive=sensitive)
        return ConfirmAnswer(True, method=method, text=answer2.text,
                             note="exact action repeated + confirmation word given")

    def _deny(
        self, tool: str, args: dict[str, Any], tier: str, decision: str, reason: str,
        started: float, sensitive: bool = False,
    ) -> GateOutcome:
        outcome = GateOutcome(
            ok=False, tool=tool, tier=tier, decision=decision,
            content=reason,
            display=reason,
        )
        outcome.duration_ms = (time.perf_counter() - started) * 1000
        self.stats[decision] = self.stats.get(decision, 0) + 1
        self._record(tool=tool, args=self._log_args(tool, args), tier=tier, decision=decision,
                     note=reason, sensitive=sensitive)
        self.log_warning("DENIED %s (%s): %s", tool, tier, reason)
        if bool(self.cfg.get("permissions.spoken_denials", True)) and decision in (DENIED,):
            self._notify("That needs your approval, so I stopped.")
        if self.activity is not None:
            try:
                self.activity.tool_call(tool, self._log_args(tool, args),
                                        result=f"{decision}: {reason}", ok=False,
                                        duration_ms=outcome.duration_ms, tier=tier,
                                        sensitive=sensitive)
            except Exception:
                pass
        return outcome

    def _notify(self, text: str) -> None:
        """Say something out loud (or print it) via services['notifier']."""
        notifier = self.services.get("notifier")
        if notifier is None:
            return
        try:
            notifier(text)
        except Exception:
            self.log_debug("notifier failed", exc_info=True)

    def summary(self) -> dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "consecutive_errors": self._consecutive_errors,
            "confirmers": [c.name for c in getattr(self.confirmer, "confirmers", [])] or [
                getattr(self.confirmer, "name", "none")
            ],
            "allowed_read": [str(p) for p in self.allowed_read],
            "allowed_write": [str(p) for p in self.allowed_write],
            "red_keywords": len(self.red_keywords),
            "blocked_patterns": len(self.blocked_patterns),
        }


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _flatten(value: Any, depth: int = 0) -> str:
    if depth > 4:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten(v, depth + 1)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v, depth + 1) for v in value)
    return "" if value is None else str(value)


def split_command_segments(command: str) -> list[str]:
    """Split a command line into its simple commands, respecting quotes.

    ``echo a; rm -rf x | tee log`` -> ["echo a", "rm -rf x", "tee log"].
    Splitting is the point: each segment must pass the allowlist on its own.
    """
    segments: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if quote:
            buffer.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            buffer.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            buffer.append(char)
            buffer.append(command[index + 1])
            index += 2
            continue
        if char in ";&|\n":
            if "".join(buffer).strip():
                segments.append("".join(buffer).strip())
            buffer = []
            while index < length and command[index] in ";&|\n":
                index += 1
            continue
        buffer.append(char)
        index += 1
    if "".join(buffer).strip():
        segments.append("".join(buffer).strip())
    return segments


def _redirect_targets(command: str) -> list[str]:
    """Paths a command writes to with ``>`` or ``>>`` (quote-aware-ish)."""
    targets: list[str] = []
    tokens = command.replace(">>", " > ").replace(">", " > ").split()
    for index, token in enumerate(tokens):
        if token == ">" and index + 1 < len(tokens):
            candidate = tokens[index + 1].strip("\"'")
            if candidate and not candidate.startswith("&"):
                targets.append(candidate)
    return targets


def _first_token(command: str) -> str:
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        parts = command.split()
    return parts[0] if parts else ""


def _spoken_challenge(tool: Any, args: dict[str, Any]) -> str:
    """Describe an action in words, for the spoken RED repeat-back check."""
    spoken_action = getattr(tool, "spoken_action", None)
    if callable(spoken_action):
        try:
            phrase = str(spoken_action(args) or "").strip()
            if phrase:
                return phrase
        except Exception:
            pass
    verb = tool.name.split(".")[-1].replace("_", " ")
    values = [
        str(value) for value in args.values()
        if isinstance(value, (str, int, float)) and str(value).strip()
    ][:2]
    return " ".join([verb] + values).strip()


def mask_secrets(args: dict[str, Any], mask: str = "***") -> dict[str, Any]:
    """Replace every argument value with a mask, keeping the shape of the call.

    Used when the gate has flagged a call as secret-bearing: the user still sees
    which tool and which arguments, but no value is displayed, logged or spoken.
    """
    return {key: (mask if not isinstance(value, (dict, list, tuple)) else mask) for key, value in args.items()}


#: Key formats that stay protected even if files.denied_globs is emptied: these
#: are the files a leak would be unrecoverable for. Extension patterns *from*
#: config.yaml are added to this set (see _bare_protected_patterns).
_BARE_PROTECTED_PATTERNS = ("*.pem", "*.key", "*.kdbx", "*.p12", "*.pfx", "*.ppk")

#: Bare file names that are off limits to a command even without a path around
#: them. Keys, credential stores and GARVIS's own files - the same promise as
#: files.denied_globs, applied to arguments that carry no folder.
_BARE_PROTECTED_TOKENS = frozenset({
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",                          # ssh keys
    ".netrc", ".htpasswd", ".git-credentials", ".env", ".pgpass",          # credential files
    "Login Data", "Cookies", "cookies.sqlite", "credentials.json",         # browser stores
    "config.yaml", "config.local.yaml", "permissions.py", "killswitch.py",
    "system_prompt.md",                                                    # GARVIS's own files
})


def _command_tokens(command: str) -> list[str]:
    """Split a command line into tokens, tolerating an unbalanced quote."""
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _bare_protected_patterns(denied_globs: Iterable[str]) -> set[str]:
    """Extension patterns ("*.key") from files.denied_globs, for bare names.

    Derived rather than duplicated, so adding a pattern to config.yaml protects
    the shell too - the two cannot drift apart.
    """
    patterns: set[str] = set()
    for glob in denied_globs:
        base = str(glob).replace("**/", "").strip()
        if base.startswith("*.") and "*" not in base[2:] and "/" not in base[2:]:
            patterns.add(base)
    return patterns


def _exact_action(tool: Any, args: dict[str, Any]) -> str:
    """The exact, scriptable form of the action the user must repeat for RED."""
    if getattr(tool, "command_args", ()):
        for arg_name in tool.command_args:
            if args.get(arg_name):
                return str(args[arg_name])
    if getattr(tool, "path_args", ()):
        pieces = [tool.name]
        for key, value in args.items():
            if key == "confirm":
                continue
            pieces.append(f"{key}={value}")
        return " ".join(pieces)
    return tool.arg_summary(args)


def domain_of(url: str) -> str:
    """Extract a bare hostname from a URL or a bare domain."""
    text = str(url).strip().lower()
    if not text:
        return ""
    text = re.sub(r"^[a-z]+://", "", text)
    text = text.split("/")[0].split("?")[0].split("#")[0]
    if "@" in text:
        text = text.rsplit("@", 1)[1]
    if text.startswith("["):  # ipv6
        return text.split("]")[0].lstrip("[")
    text = text.split(":")[0]
    return text.strip(".")


def build_gate(cfg: Any, registry: Any, activity: Any = None, services: dict[str, Any] | None = None,
               log: Any = None, confirmer: Confirmer | None = None) -> PermissionGate:
    """Factory used by main.py, the self-test, and stage-3 tests."""
    return PermissionGate(
        cfg=cfg, activity=activity, services=services, log=log, confirmer=confirmer, registry=registry
    )

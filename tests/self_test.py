"""GARVIS self-test: check every part before you trust it with your machine.

    python tests/self_test.py             # everything (model, mic, speaker, browser, screen)
    python tests/self_test.py --quick     # no model, no hardware, no network: config,
                                          # sandbox, permissions, state, memory, tools
    python tests/self_test.py --json      # machine-readable report
    python tests/self_test.py --speak     # also speak a test sentence out loud
    python tests/self_test.py --stress    # include the slow paths (whisper load, capture)
    python main.py --self-test            # the same thing, from main.py

Statuses:
    PASS  works, checked by actually doing it (not by asking "is it installed?")
    WARN  optional part missing or switched off; GARVIS still works, less capable
    FAIL  something required is broken; the reason and the fix are printed
    SKIP  not checked in this mode (--quick, or deliberately disabled in config)

The permission and password checks are the important ones: they prove, live,
that the gate refuses what it must refuse, including when the confirmation
comes from a script that would say yes to anything.

Exit code: 0 = nothing failed, 1 = at least one failure, 2 = could not even
start (config or import error).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
STATUS_ORDER = {FAIL: 0, WARN: 1, PASS: 2, SKIP: 3}
LABEL = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}


@dataclass
class Result:
    """One check's outcome: what was looked at, and what to do about it."""

    area: str
    status: str
    detail: str
    fix: str = ""
    seconds: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


class Ctx:
    """Everything the checks share: the real config, the real services."""

    def __init__(self, cfg: Any, *, quick: bool = False, speak: bool = False,
                 stress: bool = False, workdir: Path | None = None) -> None:
        self.cfg = cfg
        self.quick = quick
        self.speak = speak
        self.stress = stress
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="garvis-self-test-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.app: Any = None                    # built lazily by services()
        self.notices: list[str] = []            # what the gate said out loud

    def services(self) -> Any:
        """Build the real GARVIS once, entirely inside the self-test workdir.

        State, logs, sandbox, screenshots and memory all point at a temp folder,
        so running the self-test never touches a real session, a real file or a
        real task record. The services themselves are the real ones.
        """
        if self.app is None:
            import copy

            import main as main_module

            # A copy, so a self-test (which disables the ui and redirects every
            # path into the workdir) cannot change what the checks report.
            cfg = copy.deepcopy(self.cfg)
            cfg.set("ui.enabled", False)
            cfg.set("state.dir", str(self.workdir / "state"))
            cfg.set("state.auto_task", False)
            cfg.set("logging.activity_log", str(self.workdir / "activity_log.txt"))
            cfg.set("logging.activity_jsonl", str(self.workdir / "activity_log.jsonl"))
            cfg.set("logging.file", str(self.workdir / "garvis.log"))
            cfg.set("files.sandbox_dir", str(self.workdir / "sandbox"))
            # The allowlists have to move with the sandbox. If they kept pointing
            # at the real folders, the gate would approve a relative path that
            # resolves elsewhere and the tools would write outside what the gate
            # checked - so the self-test works entirely inside its workdir.
            cfg.set("files.allowed_read", [str(self.workdir / "sandbox"), str(self.workdir / "memory")])
            cfg.set("files.allowed_write", [str(self.workdir / "sandbox"), str(self.workdir / "memory")])
            cfg.set("screen.shots_dir", str(self.workdir / "screen_shots"))
            cfg.set("screen.frames_dir", str(self.workdir / "screen_frames"))
            cfg.set("browser.screenshots_dir", str(self.workdir / "browser_shots"))
            self.app = main_module.Garvis(cfg, main_module.build_parser().parse_args(["--no-ui"]))
            # The gate "speaks" its prompts through this hook; collect them
            # instead, so the test's own output stays readable and the messages
            # can be asserted on.
            self.app.services["notifier"] = self.notices.append
            # A self-test must never wait for a human at the keyboard. Every
            # approval is scripted - and the script still has to satisfy the RED
            # rules, which is what the permissions check verifies.
            if self.app.gate is not None:
                from core.permissions import ScriptedConfirmer  # noqa: PLC0415

                self.app.gate.confirmer = ScriptedConfirmer(approve_all=True)
        return self.app

    def service(self, name: str) -> Any:
        return self.services().services.get(name)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ok(area: str, detail: str, **evidence: Any) -> Result:
    return Result(area, PASS, detail, evidence=evidence)


def _warn(area: str, detail: str, fix: str = "", **evidence: Any) -> Result:
    return Result(area, WARN, detail, fix=fix, evidence=evidence)


def _fail(area: str, detail: str, fix: str = "", **evidence: Any) -> Result:
    return Result(area, FAIL, detail, fix=fix, evidence=evidence)


def _skip(area: str, detail: str) -> Result:
    return Result(area, SKIP, detail)


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _module_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "?"


def _call(gate: Any, tool: str, **args: Any) -> Any:
    """Run a tool through the real permission gate."""
    return gate.execute(tool, args)


# ---------------------------------------------------------------------------
# 1. this machine
# ---------------------------------------------------------------------------
def check_environment(ctx: Ctx) -> Result:
    python = f"{platform.python_version()} ({platform.system()} {platform.release()}, {platform.machine()})"
    if sys.version_info < (3, 10):
        return _fail("environment", f"Python {python} is too old", "install Python 3.10 or newer")
    return _ok("environment", f"Python {python}, project root {PROJECT_ROOT}")


REQUIRED_PACKAGES = (("yaml", "pyyaml"), ("requests", "requests"))
OPTIONAL_PACKAGES = (
    ("numpy", "numpy", "speech and screen frames"),
    ("playwright", "playwright", "browser control"),
    ("PIL", "pillow", "screenshots, tray icon"),
    ("pystray", "pystray", "tray icon"),
    ("sounddevice", "sounddevice", "microphone + speaker"),
    ("faster_whisper", "faster-whisper", "speech-to-text"),
    ("openwakeword", "openwakeword", "wake word (a spoken 'Garvis' works without it)"),
    ("keyboard", "keyboard", "the Ctrl+Alt+Esc hotkey"),
    ("mss", "mss", "screen capture"),
    ("pyautogui", "pyautogui", "mouse/keyboard control (off by default)"),
)


def check_packages(ctx: Ctx) -> Result:
    missing_required = [pip for mod, pip in REQUIRED_PACKAGES if not _has_module(mod)]
    if missing_required:
        return _fail(
            "packages",
            "missing required package(s): " + ", ".join(missing_required),
            f"pip install {' '.join(missing_required)}",
        )
    missing_optional = [
        f"{pip} ({why})" for mod, pip, why in OPTIONAL_PACKAGES if not _has_module(mod)
    ]
    if missing_optional:
        return _warn(
            "packages",
            "required packages present; optional ones missing: " + ", ".join(missing_optional),
            "pip install -r requirements.txt   (or install just what you need)",
            missing=missing_optional,
        )
    return _ok("packages", "all required and optional packages are installed")


def check_config(ctx: Ctx) -> Result:
    cfg = ctx.cfg
    problems: list[str] = []
    if getattr(cfg, "warnings", None):
        problems.extend(cfg.warnings)
    if not cfg.allowed_folders("read"):
        problems.append("files.allowed_read is empty: GARVIS cannot read anything")
    if not cfg.allowed_folders("write"):
        problems.append("files.allowed_write is empty: GARVIS cannot write anything")
    if not cfg.allowed_sites():
        problems.append("browser.allowed_sites is empty: the browser is fully blocked")
    if cfg.cloud_fallback_enabled:
        problems.append(
            "cloud fallback is ENABLED: prompts may leave this machine "
            "(safety.allow_cloud_when_offline)"
        )
    detail = (
        f"{cfg.config_path.name}: model {cfg.model}, personality {cfg.personality}, "
        f"read={len(cfg.allowed_folders('read'))} write={len(cfg.allowed_folders('write'))} "
        f"sites={len(cfg.allowed_sites())}, cloud fallback "
        f"{'ON' if cfg.cloud_fallback_enabled else 'off'}"
    )
    if problems:
        return _warn("config", detail + " | " + "; ".join(problems),
                     "fix the entries in config.yaml; `python main.py --check` repeats them",
                     problems=problems)
    return _ok("config", detail)


# ---------------------------------------------------------------------------
# 2. logging, and the promise never to write passwords
# ---------------------------------------------------------------------------
def check_logging(ctx: Ctx) -> Result:
    app = ctx.services()
    activity = app.activity
    marker = f"self-test {int(time.time())}"
    secret = "password=hunter2-do-not-log"
    activity.event("system", f"{marker} {secret}")

    # Ask the logger where it writes: in a self-test that is the temp workdir.
    log_path = Path(getattr(activity, "activity_path", None)
                    or ctx.cfg.resolve_path(str(ctx.cfg.get("logging.activity_log"))))
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if marker not in text:
        return _fail(
            "logging",
            f"the activity log did not receive the test line ({log_path})",
            "check logs/ is writable and logging.activity_log in config.yaml",
        )
    if "hunter2" in text:
        return _fail(
            "logging",
            "a password-shaped value was written to the activity log",
            "restore logging.redact_patterns from the shipped config.yaml",
        )
    return _ok("logging", f"writes and reads back {log_path}; redaction masked the test password")


def check_memory(ctx: Ctx) -> Result:
    from core.config import Config
    from core.memory import Memory

    real = Memory.from_config(ctx.cfg)              # read only
    profile = real.load_profile()
    project_log = real.load_project_log()
    stats = real.stats()

    # Prove writing works, in the temp copy - never in the user's own log.
    cfg = Config.load(ctx.cfg.config_path)
    cfg.set("memory.dir", str(ctx.workdir / "memory"))
    cfg.set("memory.profile_file", str(ctx.workdir / "memory" / "profile.md"))
    cfg.set("memory.project_log_file", str(ctx.workdir / "memory" / "project_log.md"))
    scratch = Memory.from_config(cfg)
    scratch.ensure_files()
    note = scratch.note(f"self-test at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not profile.strip():
        return _warn("memory", f"memory files exist ({stats}) but the profile is empty",
                     "put your name, timezone and preferences in memory/profile.md",
                     stats=stats)
    if not scratch.project_log_path.exists() or "self-test" not in note:
        return _fail("memory", "the project log could not be written",
                     "check memory.dir is writable and memory/ is not read-only")
    return _ok(
        "memory",
        f"profile ({len(profile)} chars) and project log ({len(project_log)} chars) read; "
        f"writing works ({note})",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# 3. the permission gate: everything it must refuse, refused live
# ---------------------------------------------------------------------------
def check_permissions(ctx: Ctx) -> Result:
    """The safety requirements, checked by trying to break them.

    A scripted user says "yes" to everything; the gate must still refuse the
    RED action without the repeat + confirmation word, refuse paths outside the
    allowlist, and refuse to type a password.
    """
    app = ctx.services()
    gate = app.gate
    if gate is None:
        return _fail("permissions", "no permission gate was built",
                     "check permissions.enabled in config.yaml (never run GARVIS without it)")

    from core.permissions import ALLOWED, BLOCKED, CONFIRMED, DENIED, ScriptedConfirmer

    original_confirmer = gate.confirmer
    problems: list[str] = []
    evidence: dict[str, Any] = {}

    # a) GREEN runs without asking
    green = _call(gate, "clock.now")
    evidence["green"] = green.decision
    if not green.ok or green.decision != ALLOWED:
        problems.append(f"a GREEN tool (clock.now) was not auto-allowed ({green.decision})")

    # b) RED with a "yes" to everything: must still be refused, because the
    #    user has to repeat the exact action and then say the confirm word.
    yes_only = ScriptedConfirmer(default_text="yes")
    gate.confirmer = yes_only
    red = _call(gate, "files.delete", path="does-not-exist-self-test.txt")
    evidence["red_with_endless_yes"] = red.decision
    if red.decision not in (DENIED, BLOCKED):
        problems.append(f"a RED action was not stopped by a script that only says yes ({red.decision})")

    # c) RED with the real cooperation still needs the exact confirmation
    gate.confirmer = ScriptedConfirmer(approve_all=True)
    red_ok = _call(gate, "files.delete", path="does-not-exist-self-test.txt")
    evidence["red_with_confirmation"] = red_ok.decision
    if red_ok.decision != CONFIRMED:
        problems.append(f"a properly confirmed RED action did not go through ({red_ok.decision})")

    # d) a path outside the allowlist is refused even when everything is approved
    outside = _call(gate, "files.read", path=str(Path.home() / ".ssh" / "id_rsa"))
    evidence["outside_allowlist"] = outside.decision
    if outside.ok or outside.decision not in (DENIED, BLOCKED):
        problems.append("reading ~/.ssh/id_rsa was not refused")

    # e) a password is never typed, whatever the user says
    gate.confirmer = ScriptedConfirmer(approve_all=True)
    pw = _call(gate, "browser.type", selector="#password", text="CorrectHorseBattery")
    evidence["password_typing"] = pw.decision
    if pw.ok or pw.decision not in (BLOCKED, DENIED):
        problems.append("typing into a password field was not refused")

    # f) the spoken kill phrase and the hotkey are wired to a real switch
    from core.killswitch import KillSwitch

    ks = KillSwitch(ctx.cfg)
    if not ks.matches_kill_phrase("garvis, stop everything") or ks.matches_kill_phrase("nice weather"):
        problems.append("the spoken stop phrase matcher is not working")
    evidence["hotkey"] = str(ctx.cfg.get("hotkeys.killswitch", "Ctrl+Alt+Esc"))

    gate.confirmer = original_confirmer        # leave the gate as we found it

    if problems:
        return _fail("permissions", "; ".join(problems),
                     "this is a safety bug - restore the shipped permissions.py/config.yaml",
                     problems=problems, **evidence)
    return _ok(
        "permissions",
        "gate refuses RED-without-confirmation, outside-allowlist paths and passwords; "
        f"GREEN auto-allowed; hotkey {evidence['hotkey']}",
        **evidence,
    )


def check_permissions_wired(ctx: Ctx) -> Result:
    """The gate must be the thing the brain actually calls, not decoration."""
    app = ctx.services()
    brain = app.services.get("brain")
    if app.gate is None:
        return _fail("gate wiring", "the gate is missing", "never start with --no-tools for real work")
    if brain is None:
        return _warn("gate wiring", "no brain service to check", "")
    if getattr(brain, "gate", None) is not app.gate:
        if getattr(brain, "gate", None) is None:
            # A brain without a gate refuses every tool call (fail closed), so
            # this is safe - but GARVIS cannot do anything. --no-tools asks for
            # exactly this, hence a warning rather than a failure.
            return _warn(
                "gate wiring",
                "the brain has no gate, so it refuses every tool call (fail closed)",
                "start without --no-tools if GARVIS should be able to act",
            )
        return _fail("gate wiring", "the brain is not using the permission gate",
                     "this is a safety bug: every tool call must go through the gate")
    return _ok("gate wiring", "every tool call the brain makes goes through the gate")


# ---------------------------------------------------------------------------
# 4. files, sandbox, tools
# ---------------------------------------------------------------------------
def check_sandbox(ctx: Ctx) -> Result:
    app = ctx.services()
    gate = app.gate
    # The app's own (temp) sandbox: the self-test must not write into the real one.
    sandbox = Path(str(app.cfg.get("files.sandbox_dir")))
    if not sandbox.is_absolute():
        sandbox = app.cfg.resolve_path(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)

    name = "self-test-note.txt"
    text = f"written by the self-test at {time.strftime('%H:%M:%S')}"
    steps: list[str] = []
    for tool, args in (
        ("files.write", {"path": name, "text": text}),
        ("files.read", {"path": name}),
        ("files.stat", {"path": name}),
        ("files.search", {"query": "self-test", "path": "."}),
        ("files.delete", {"path": name}),
    ):
        outcome = _call(gate, tool, **args)
        steps.append(f"{tool}:{'ok' if outcome.ok else outcome.decision}")
        if tool == "files.read" and text not in str(outcome.content):
            return _fail("sandbox", f"{tool} did not return what was written",
                         "check files.sandbox_dir and the files tools")
        if not outcome.ok:
            return _fail("sandbox", f"{tool} failed: {outcome.content}",
                         f"check that {sandbox} exists and is writable")
    if (sandbox / name).exists():
        return _fail("sandbox", "files.delete left the file behind", "check tools/files.py")
    return _ok("sandbox", f"write/read/stat/search/delete round trip in {sandbox} ({', '.join(steps)})")


def check_tools(ctx: Ctx) -> Result:
    app = ctx.services()
    registry = app.registry
    tools = list(registry.all())
    if not tools:
        return _fail("tools", "no tools registered", "check tools/__init__.py and the tool modules")
    problems: list[str] = []
    names = [tool.name for tool in tools]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        problems.append(f"duplicate tool names: {sorted(duplicates)}")
    for tool in tools:
        if not tool.description or len(tool.description) < 20:
            problems.append(f"{tool.name} has no usable description")
        if tool.tier not in ("green", "yellow", "red"):
            problems.append(f"{tool.name} has an unknown tier {tool.tier!r}")
    counts: dict[str, int] = {}
    for tool in tools:
        counts[tool.tier] = counts.get(tool.tier, 0) + 1
    if not any(tool.tier == "red" for tool in tools):
        problems.append("no RED tools are registered: the dangerous tiers are not gated")
    if problems:
        return _fail("tools", "; ".join(problems), "check the tool modules", problems=problems)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return _ok("tools", f"{len(tools)} tools registered ({summary}), all described and tiered")


# ---------------------------------------------------------------------------
# 5. task state: crash-resume, honesty about rollback
# ---------------------------------------------------------------------------
def check_state(ctx: Ctx) -> Result:
    from core.config import Config
    from core.state import StateStore

    cfg = Config.load(ctx.cfg.config_path)          # a copy, so the test cannot disturb a session
    cfg.set("state.dir", str(ctx.workdir / "state"))
    cfg.set("logging.activity_log", str(ctx.workdir / "activity_log.txt"))
    cfg.set("logging.activity_jsonl", str(ctx.workdir / "activity_log.jsonl"))

    store = StateStore(cfg, subscribe=False)
    store.begin_session()
    store.start_task("self-test task")
    store.record_step("files.move", "moved a file",
                      args={"path": "/tmp/a.txt", "destination": "/tmp/b.txt"})
    store.save()                                    # written, then "killed": no close()
    after_crash = StateStore(cfg, subscribe=False)
    pending = after_crash.begin_session()
    if pending is None:
        return _fail("state", "an unfinished task was not noticed after a simulated crash",
                     "check core/state.py begin_session()")
    if not pending.steps:
        return _fail("state", "the recorded steps were lost", "check core/state.py save()")
    resumed = after_crash.resume_task("self-test resume")
    if resumed is None or after_crash.current is None:
        return _fail("state", "the interrupted task could not be resumed",
                     "check core/state.py resume_task()")
    plan = after_crash.rollback_plan()
    if not plan or "move" not in plan[0]:
        return _fail("state", f"the rollback plan lost the undo hint ({plan})",
                     "check core/state.py _UNDO_HINTS")
    after_crash.finish_task("done", status="done")
    empty = StateStore(cfg, subscribe=False)
    empty.begin_session()
    honesty = empty.rollback_plan()
    if not honesty or "nothing" not in honesty[0].lower():
        return _fail("state", f"an empty rollback plan is not honest ({honesty})",
                     "check core/state.py rollback_plan()")
    return _ok("state", "crash detected, steps kept, resume works, rollback plan honest and read-only",
               plan=plan, honesty=honesty[0])


# ---------------------------------------------------------------------------
# 6. the model
# ---------------------------------------------------------------------------
def check_ollama(ctx: Ctx) -> Result:
    if ctx.quick:
        return _skip("ollama", "skipped in --quick mode")
    app = ctx.services()
    client = app.brain.client
    host = ctx.cfg.ollama_host
    if not client.is_up():
        return _fail("ollama", f"not answering at {host}",
                     "run `ollama serve` (and `ollama pull "
                     f"{ctx.cfg.model}`), then run this self-test again")
    version = client.version() or "?"
    compare_models = bool(ctx.cfg.get("self_test.check_models", True))
    models = client.list_models() if compare_models else []
    missing = [m for m in (ctx.cfg.model, ctx.cfg.vision_model) if not client.has_model(m)] if compare_models else []
    if compare_models and not client.has_model(ctx.cfg.model):
        return _fail("ollama", f"model '{ctx.cfg.model}' is not installed (have: {', '.join(models) or 'none'})",
                     f"ollama pull {ctx.cfg.model}")

    # Actually talk to it: "installed" is not the same as "answers".
    started = time.perf_counter()
    try:
        resp = client.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            tools=None,
        ) if hasattr(client, "chat") else None
    except Exception as exc:
        return _fail("ollama", f"the model did not answer: {exc}",
                     "check `ollama logs` and brain.request_timeout_s")
    took = time.perf_counter() - started
    if resp is None:
        return _warn("ollama", f"{ctx.cfg.model} is installed (ollama {version}) but this client cannot chat",
                     "check core/brain.py OllamaClient.chat")

    detail = f"ollama {version}, {ctx.cfg.model} answered in {took:.1f}s"
    if not compare_models:
        detail += " (self_test.check_models is false: the model list was not compared)"
    if missing:
        return _warn("ollama", detail + f"; not installed: {', '.join(missing)}",
                     " ".join(f"ollama pull {m}" for m in missing), models=models)
    return _ok("ollama", detail + f" (models: {', '.join(models[:6])})", models=models)


# ---------------------------------------------------------------------------
# 7. voice
# ---------------------------------------------------------------------------
def check_voice_out(ctx: Ctx) -> Result:
    app = ctx.services()
    tts = app.services.get("tts")
    if tts is None:
        return _fail("voice out", "the TTS service was not built",
                     "check voice_out in config.yaml and core/voice_out.py")
    if not tts.enabled:
        return _skip("voice out", "voice_out.enabled is false: replies are printed, not spoken")
    detail = tts.describe()
    if not getattr(tts.engine, "available", False):
        return _warn("voice out", f"no engine available ({detail})",
                     "pip install piper-tts, then put a .onnx voice in models/voices/ "
                     "(see the README's voice section); or set voice_out.engine: pyttsx3")
    if ctx.quick:
        return _ok("voice out", f"engine ready: {detail} (not spoken in --quick mode)")
    if not ctx.speak:
        return _ok("voice out", f"engine ready: {detail} (pass --speak to hear it)")
    if not bool(ctx.cfg.get("self_test.check_speaker", True)):
        return _ok("voice out", f"engine ready: {detail} "
                               f"(self_test.check_speaker is false: nothing was spoken)")
    tts.say("Self test: my voice works.", wait=True)
    if tts.last_error:
        return _fail("voice out", f"speaking failed: {tts.last_error}", "check voice_out.device")
    return _ok("voice out", f"spoke a test sentence through {detail}")


def check_voice_in(ctx: Ctx) -> Result:
    if ctx.quick:
        return _skip("voice in", "skipped in --quick mode (needs a microphone)")
    app = ctx.services()
    listener = app.services.get("voice_in")
    if listener is None:
        return _warn("voice in", "voice input is disabled or unavailable in config.yaml",
                     "set voice_in.enabled: true and install sounddevice")
    mic = listener.mic
    if not getattr(mic, "available", False):
        return _warn("voice in", mic.describe()[:300],
                     "pip install sounddevice  (Linux also: sudo apt install libportaudio2)")
    devices = type(mic).list_devices()
    if not devices:
        return _warn("voice in", "the microphone backend loaded but no device was found",
                     "plug in a microphone; `python main.py --devices` lists what the OS reports")

    if not mic.start():
        return _warn("voice in", f"could not open the microphone: {getattr(mic, 'reason', '?')}",
                     "close other apps using the microphone, or set voice_in.input_device")
    frames, peak = 0, 0.0
    try:
        deadline = time.time() + min(2.0, float(ctx.cfg.get("self_test.mic_seconds", 2)))
        while time.time() < deadline:
            frame = mic.read_frame(timeout=0.5)
            if frame is None:
                continue
            frames += 1
            try:
                peak = max(peak, float(abs(frame).max()))
            except Exception:
                pass
    finally:
        mic.stop()

    if frames == 0:
        return _fail("voice in", "the microphone opened but delivered no audio frames",
                     "try another voice_in.input_device (`python main.py --devices`)")
    level = "silent (that is fine if you were not talking)" if peak < 0.01 else f"peak {peak:.2f}"
    detail = f"microphone works: {len(devices)} device(s), {frames} frames in 2s, {level}"

    if not ctx.stress:
        return _ok("voice in", detail + " (pass --stress to load the speech model too)")
    started = time.perf_counter()
    ok = listener.stt.load()
    took = time.perf_counter() - started
    if not ok:
        return _warn("voice in", detail + f"; speech-to-text did not load: {listener.stt.reason}",
                     "pip install faster-whisper (it downloads the model on first use)")
    return _ok("voice in", detail + f"; speech-to-text loaded in {took:.1f}s")


def check_wake_word(ctx: Ctx) -> Result:
    app = ctx.services()
    wake = app.services.get("wake")
    if wake is None or not getattr(wake, "available", False):
        return _warn(
            "wake word",
            f"no wake-word model ({(wake.reason if wake is not None else 'not built')}); "
            "saying 'Garvis, ...' still wakes it via speech-to-text",
            "pip install openwakeword, or just say the name first",
        )
    return _ok("wake word", f"detector ready: {wake.describe()}")


# ---------------------------------------------------------------------------
# 8. browser, screen, ui, kill switch
# ---------------------------------------------------------------------------
def check_browser(ctx: Ctx) -> Result:
    if ctx.quick:
        return _skip("browser", "skipped in --quick mode")
    app = ctx.services()
    manager = app.services.get("browser")
    if manager is None:
        return _warn("browser", "the browser service is off in config.yaml (browser.enabled)",
                     "set browser.enabled: true to let GARVIS use a browser")
    if not _has_module("playwright"):
        return _warn("browser", "playwright is not installed",
                     "pip install playwright && playwright install chromium")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        if not path or not Path(path).exists():
            return _warn("browser", "playwright is installed but its Chromium is not",
                         "playwright install chromium")
    except Exception as exc:
        return _warn("browser", f"playwright could not be used: {exc}",
                     "playwright install chromium")
    return _ok("browser", f"playwright + chromium ready; {manager.describe()}")


def check_screen(ctx: Ctx) -> Result:
    if ctx.quick:
        return _skip("screen", "skipped in --quick mode")
    app = ctx.services()
    screen = app.services.get("screen")
    if screen is None:
        return _warn("screen", "screen capture is off in config.yaml (screen.enabled)",
                     "set screen.enabled: true to let GARVIS look at your screen")
    capturer = screen.capturer
    if not getattr(capturer, "available", False):
        return _warn("screen", f"cannot see the screen: {capturer.describe()[:250]}",
                     "pip install mss pillow  (macOS: allow Screen Recording for your terminal)")
    monitors = capturer.monitors() if hasattr(capturer, "monitors") else []
    result = screen.capture(name="self-test")
    if not result.ok:
        return _fail("screen", f"capture failed: {result.message}",
                     "check the Screen Recording permission (macOS) or the display session (Linux)")
    data = getattr(result, "data", {}) or {}
    size = ""
    path = data.get("path") or data.get("file")
    if path and Path(str(path)).exists():
        size = f", {Path(str(path)).stat().st_size // 1024} KiB saved"
    mode = "guidance only" if screen.guidance_only else "control ENABLED"
    return _ok("screen", f"captured a frame from {capturer.name} ({len(monitors)} monitor(s)){size}; {mode}")


def check_ui(ctx: Ctx) -> Result:
    from core.ui import build_ui

    ui = build_ui(ctx.cfg)          # built, never started: no windows pop up
    have_pystray, have_tk = _has_module("pystray"), _has_module("tkinter")
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ""
    lines = []
    if have_pystray:
        lines.append("tray: pystray installed (pillow needed too) "
                     + ("installed" if _has_module("PIL") else "MISSING: pip install pillow"))
    else:
        lines.append("tray: not available (pip install pystray pillow)")
    if have_tk:
        lines.append("overlay: tkinter present"
                     + ("" if (display or platform.system() in ("Windows", "Darwin"))
                        else " but there is no display session ($DISPLAY is empty)"))
    else:
        lines.append("overlay: no tkinter (Linux: sudo apt install python3-tk)")
    if display or platform.system() in ("Windows", "Darwin"):
        lines.append(f"display: {display or platform.system()}")
    detail = "; ".join(lines)
    if not getattr(ui, "available", False):
        return _warn("ui", detail + f" | {ui.describe()}",
                     "pip install pystray pillow (tray) and tkinter (overlay); "
                     "everything still works without them - the status goes to the terminal")
    return _ok("ui", detail + f" | {ui.describe()}")


def check_killswitch(ctx: Ctx) -> Result:
    app = ctx.services()
    ks = app.services.get("killswitch") or app.killswitch
    status = ks.status()
    hotkey = str(ctx.cfg.get("hotkeys.killswitch", "Ctrl+Alt+Esc"))
    phrases = ", ".join(str(p) for p in (ctx.cfg.get("safety.kill_phrases") or [])[:3])
    wired = ks.install_hotkey()
    detail = (f"hotkey {hotkey}: {'registered' if wired else 'NOT registered'}; "
              f"spoken: {phrases}")
    if not wired:
        return _warn("kill switch", detail,
                     "pip install keyboard (Windows/Linux) and run with the permissions it needs; "
                     "the spoken stop phrase and the tray STOP button still work",
                     status=status)
    return _ok("kill switch", detail + "; the STOP button and the spoken phrase use the same switch",
               status=status)


def check_stop_kills_work(ctx: Ctx) -> Result:
    """Start a real command, press stop, and check it is dead.

    "Stop everything" is only worth anything if it stops work that is already
    running, so this is not a unit test with a fake process: it runs a real
    command through the real gate and the real shell tool, stops it, and then
    checks the process is gone and did not finish what it was doing.
    """
    import threading
    import time as _time

    from core.killswitch import KillSwitch

    app = ctx.services()
    gate = app.gate
    if gate is None:
        return _skip("stop kills work", "no permission gate, so no tools can run")

    sandbox = Path(str(app.cfg.get("files.sandbox_dir")))
    if not sandbox.is_absolute():
        sandbox = app.cfg.resolve_path(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    marker = sandbox / "self-test-stopped-work.txt"
    marker.unlink(missing_ok=True)

    from core.permissions import ScriptedConfirmer

    switch = KillSwitch(app.cfg, activity=app.activity)
    original = app.services.get("killswitch")
    original_confirmer = gate.confirmer
    app.services["killswitch"] = switch          # tools register their children here
    gate.confirmer = ScriptedConfirmer(approve_all=True)   # never wait for a human
    command = (
        "python3 -c \"import time, pathlib; time.sleep(20); "
        "pathlib.Path('self-test-stopped-work.txt').write_text('finished')\""
    )
    box: dict[str, object] = {}

    def worker() -> None:
        started = _time.perf_counter()
        try:
            box["outcome"] = gate.execute("shell.run", {"command": command, "timeout_s": 45})
        except Exception as exc:
            box["error"] = exc
        box["seconds"] = _time.perf_counter() - started

    thread = threading.Thread(target=worker, name="selftest-stop", daemon=True)
    thread.start()
    registered = False
    deadline = _time.time() + 10
    while _time.time() < deadline:
        if switch.running_processes():
            registered = True
            break
        _time.sleep(0.05)
    if not registered:
        thread.join(5)
        app.services["killswitch"] = original
        gate.confirmer = original_confirmer
        detail = f"the command never registered with the kill switch ({box.get('error') or box.get('outcome')})"
        return _fail("stop kills work", detail,
                     "check tools/shell.py registers its process with the kill switch")

    switch.trigger("self-test", source="self-test")
    thread.join(20)
    _time.sleep(1.5)                     # give a survivor the chance to write its marker
    still_running = thread.is_alive()
    finished_anyway = marker.exists()
    stopped = switch.last_stop.extra.get("terminated_processes") or []
    switch.resume("self-test over")
    app.services["killswitch"] = original
    gate.confirmer = original_confirmer
    marker.unlink(missing_ok=True)

    if still_running or finished_anyway or not stopped:
        return _fail(
            "stop kills work",
            "a running command survived the stop"
            + (" (the tool never returned)" if still_running else "")
            + (" (it finished anyway)" if finished_anyway else ""),
            "this is a safety bug: check core/killswitch.py stop_processes() and "
            "tools/shell.py",
        )
    seconds = float(box.get("seconds") or 0)
    return _ok(
        "stop kills work",
        f"a running command was terminated {seconds:.1f}s in, before it could finish "
        f"({len(stopped)} process group(s))",
    )


# ---------------------------------------------------------------------------
# 9. the startup routine itself
# ---------------------------------------------------------------------------
def check_wake_up(ctx: Ctx) -> Result:
    app = ctx.services()
    try:
        text = app.wake_up(quiet=True)
    except Exception as exc:
        return _fail("wake up", f"the startup routine raised {exc!r}",
                     "check main.py wake_up() - it must never be able to break startup",
                     traceback=traceback.format_exc()[-800:])
    if not text.strip():
        return _fail("wake up", "the startup routine said nothing", "check safety.startup_greeting")
    return _ok("wake up", f"would say: {text}")


# ---------------------------------------------------------------------------
# running the checks
# ---------------------------------------------------------------------------
#: A check named here is skipped - and reported as skipped, never silently
#: dropped - when the matching switch in config.yaml is false.
CHECK_SWITCHES: dict[str, str] = {
    "ollama": "self_test.check_ollama",
    "voice out": "self_test.check_tts_voice",
    "voice in": "self_test.check_mic",
    "browser": "self_test.check_playwright",
    "ui": "self_test.check_tray",
    "sandbox": "self_test.check_folders",
}

CHECKS: tuple[tuple[str, Callable[[Ctx], Result], bool], ...] = (
    # (name, function, run in --quick mode too)
    ("environment", check_environment, True),
    ("packages", check_packages, True),
    ("config", check_config, True),
    ("logging", check_logging, True),
    ("memory", check_memory, True),
    ("permissions", check_permissions, True),
    ("gate wiring", check_permissions_wired, True),
    ("sandbox", check_sandbox, True),
    ("tools", check_tools, True),
    ("state", check_state, True),
    ("ollama", check_ollama, False),
    ("voice out", check_voice_out, True),
    ("voice in", check_voice_in, False),
    ("wake word", check_wake_word, True),
    ("browser", check_browser, False),
    ("screen", check_screen, False),
    ("ui", check_ui, True),
    ("kill switch", check_killswitch, True),
    ("stop kills work", check_stop_kills_work, True),
    ("wake up", check_wake_up, True),
)


def _run_one(name: str, fn: Callable[[Ctx], Result], ctx: Ctx, timeout_s: float) -> Result:
    """Run a check with a time limit, so one stuck device cannot hang the test."""
    box: list[Result] = []

    def worker() -> None:
        try:
            box.append(fn(ctx))
        except Exception as exc:                     # a check must never crash the report
            box.append(_fail(name, f"the check itself raised {exc!r}",
                             "this is a bug in tests/self_test.py - please report it",
                             traceback=traceback.format_exc()[-800:]))

    thread = threading.Thread(target=worker, name=f"selftest-{name}", daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout_s)
    took = time.perf_counter() - started
    if not box:
        return Result(name, FAIL, f"timed out after {timeout_s:.0f}s",
                      fix="check the device/driver it was opening, then run this test again",
                      seconds=took)
    result = box[0]
    result.seconds = took
    return result


def run_self_test(cfg: Any = None, argv: list[str] | None = None, stream: Any = None) -> int:
    """Run every applicable check. Returns the process exit code."""
    out = stream or sys.stdout
    parser = argparse.ArgumentParser(prog="self_test.py", description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="no model, no hardware, no network: the fast safety checks")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--speak", action="store_true", help="also speak a test sentence")
    parser.add_argument("--stress", action="store_true", help="include the slow paths (model loads)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        if cfg is None:
            from core.config import Config

            cfg = Config.load(args.config)
    except Exception as exc:
        print(f"could not load the configuration: {exc}", file=sys.stderr)
        return 2

    ctx = Ctx(cfg, quick=args.quick, speak=args.speak, stress=args.stress)
    timeout_s = float(ctx.cfg.get("self_test.timeout_s", 90) or 90)

    def emit(line: str = "") -> None:
        """One line of report, to the stream the caller asked for."""
        print(line, file=out)

    if not args.json:
        mode = "quick: config, sandbox, permissions, state, memory, tools" if args.quick else "full"
        emit(f"GARVIS self-test ({mode})")
        emit(f"  config : {getattr(cfg, 'config_path', '?')}")
        emit(f"  workdir: {ctx.workdir}")
        emit()

    results: list[Result] = []
    for name, fn, in_quick in CHECKS:
        # A check that does not run is still *reported*, with the reason. Silence
        # would leave the user unable to tell "tested and fine" from "never ran".
        skip_reason = ""
        if args.quick and not in_quick:
            skip_reason = "skipped in --quick mode"
        else:
            switch = CHECK_SWITCHES.get(name)
            if switch and not bool(ctx.cfg.get(switch, True)):
                skip_reason = f"disabled in config.yaml ({switch}: false)"
        result = _skip(name, skip_reason) if skip_reason else _run_one(name, fn, ctx, timeout_s)
        results.append(result)
        if not args.json:
            note = f"  {result.detail}"
            emit(f"  {name:<12} {LABEL[result.status]:<4}{note}")
            if result.fix and result.status in (FAIL, WARN):
                emit(f"  {'':<12}     fix: {result.fix}")

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    skipped = [r for r in results if r.status == SKIP]
    passed = [r for r in results if r.status == PASS]

    if not args.json:
        emit()
        emit(f"{len(results)} checks: {len(passed)} passed, {len(warnings)} warning(s), "
             f"{len(skipped)} skipped, {len(failures)} failed")
        if failures:
            emit("\nFAILED:")
            for r in failures:
                emit(f"  - {r.area}: {r.detail}")
                if r.fix:
                    emit(f"    fix: {r.fix}")
        emit("\nResult: " + ("OK" if not failures else "NOT OK - fix the failures above"))
        if args.quick:
            emit("(--quick mode: the model, microphone, speaker, browser and screen were not tested;"
                 " run without --quick once those are set up)")

    if args.json:
        emit(json.dumps({
            "quick": bool(args.quick),
            "results": [asdict(r) for r in results],
            "summary": {
                "passed": len(passed), "warnings": len(warnings),
                "skipped": len(skipped), "failed": len(failures),
            },
        }, indent=2, default=str))
    out.flush()

    # Leave nothing behind that could look like real work.
    try:
        app = ctx.app
        if app is not None:
            app.shutdown()
    except Exception:
        pass
    return 1 if failures else 0


def main() -> int:
    return run_self_test()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GARVIS - local voice-controlled assistant.

Main entry point. Stage 1 provides a text chat loop; later stages add voice in,
voice out, tools, browser, screen guidance and the tray UI. Everything is
controlled by config.yaml, and the CLI flags let you run individual pieces
without touching the config.

Usage
-----
    python main.py                     # normal startup (text loop until stage 5)
    python main.py --ask "what time is it"
    python main.py --check             # validate config + environment, then exit
    python main.py --today             # print the activity report for today
    python main.py --show-prompt       # print the assembled system prompt
    python main.py --model qwen2.5:7b  # override the model for this run
    python main.py --personality focus # start in a given tone
    python main.py --no-tools          # plain chat, tools disabled

What "healthy" looks like on startup:
    1. config loads (warnings are printed, not fatal)
    2. logs/ and memory/ exist
    3. Ollama answers, and the model in config.yaml is pulled
Then you get a `you>` prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make `python main.py` work from any directory, and let core/tools import each other.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain import Brain, BrainError, BrainEvent, SentenceChunker  # noqa: E402
from core.config import Config, ConfigError, describe  # noqa: E402
from core.logger import (  # noqa: E402
    ActivityLogger,
    configure_activity_logger,
    get_logger,
    setup_logging,
)
from core.memory import Memory  # noqa: E402

VERSION = "0.2.0-stage5"
BANNER = r"""
   ____    _    ____  __     __ ___ ____
  / ___|  / \  |  _ \ \ \   / /|_ _/ ___|
 | |  _  / _ \ | |_) | \ \ / /  | |\___ \
 | |_| |/ ___ \|  _ <   \ V /   | | ___) |
  \____/_/   \_\_| \_\   \_/   |___|____/
"""


class Garvis:
    """The application object. Later stages attach voice/UI to this."""

    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.log = get_logger("main")
        self.running = False
        self.started_at = time.time()

        self.activity = self._setup_logging()
        self.memory = Memory.from_config(cfg)
        self.memory.ensure_files()
        #: True when the voice loop is driving: swaps in the voice confirmation
        #: channel and lets barge-in stop speech mid-sentence.
        self.voice_mode = bool(args.voice)

        self.services: dict[str, Any] = {
            "cfg": cfg,
            "log": self.log,
            "activity": self.activity,
            "memory": self.memory,
            "brain": None,        # filled below
            "state": None,        # stage 8
            "ui": None,           # stage 8
            "browser": None,      # stage 6
            "tts": None,          # built in setup_voice()
            "voice_in": None,
            "wake": None,
            "killswitch": None,   # built in setup_safety()
        }

        # The kill switch owns the emergency stop for every code path, so it is
        # created before anything that can be stopped.
        from core.killswitch import KillSwitch

        self.killswitch = KillSwitch(cfg, activity=self.activity, log=self.log)
        self.services["killswitch"] = self.killswitch

        from tools import build_registry

        self.registry = build_registry(cfg, self.log, self.services)
        if args.no_tools:
            self.log.warning("tools disabled by --no-tools; the model cannot act this run")

        # The gate needs the registry, and the notifier lets it speak (stage 5)
        # or print (now) about confirmations and denials.
        self.services["registry"] = self.registry
        self.services["notifier"] = self.notify
        self.setup_voice()
        self.setup_browser()
        self.services["notifier"] = self.notify
        self.gate = self._build_gate()
        self.services["gate"] = self.gate

        self.brain = Brain(
            cfg=cfg,
            registry=None if args.no_tools else self.registry,
            gate=None if args.no_tools else self.gate,
            memory=self.memory,
            activity=self.activity,
        )
        self.services["brain"] = self.brain
        self._subscribe_killswitch()
        if args.personality:
            if not self.brain.set_personality(args.personality):
                self.log.warning("could not switch personality to %r", args.personality)

    # -- wiring ------------------------------------------------------------
    def _setup_logging(self) -> ActivityLogger:
        cfg = self.cfg
        log_path = cfg.resolve_path(cfg.get("logging.file", "logs/garvis.log"))
        setup_logging(
            log_path,
            level=cfg.get("logging.level", "INFO"),
            console=cfg.get("logging.console", True),
            redact_patterns=cfg.get("logging.redact_patterns", []),
            console_style=cfg.get("logging.console_style", "compact"),
        )
        return configure_activity_logger(cfg)

    def _build_gate(self) -> Any | None:
        """The permission gate is mandatory. If it is missing, we fail closed:
        the brain will refuse every tool call rather than run unguarded."""
        try:
            from core.permissions import PermissionGate  # noqa: PLC0415
        except ImportError as exc:
            self.log.error(
                "core/permissions.py is unavailable (%s). Running with the gate DISABLED: "
                "every tool call will be refused for safety.", exc,
            )
            return None
        gate = PermissionGate(
            cfg=self.cfg,
            activity=self.activity,
            services=self.services,
            log=self.log,
            registry=self.registry,
        )
        # In voice mode the microphone answers first, so "yes" and the RED
        # confirm word work hands-free; the console stays as the fallback.
        listener = self.services.get("voice_in")
        if self.voice_mode and listener is not None and getattr(listener.mic, "available", False):
            from core.voice_in import VoiceConfirmer

            gate.add_confirmer(
                VoiceConfirmer(
                    self.cfg, listener, voice_out=self.services.get("tts"), log=self.log,
                    accept_words=[str(self.cfg.get("permissions.yellow_confirm.phrase", "yes"))]
                    + [str(w) for w in (self.cfg.get("permissions.yellow_confirm.also_accept", []) or [])],
                ),
                first=True,
            )
        self.log.info(
            "permission gate ready: %d confirm channels, %d red keywords",
            len(getattr(gate.confirmer, "confirmers", []) or [1]),
            len(gate.red_keywords),
        )
        return gate

    # -- voice -------------------------------------------------------------
    def setup_voice(self) -> None:
        """Build TTS (always, so GARVIS can talk) and STT (only if enabled).

        TTS is constructed even when disabled, because the kill switch and the
        permission gate both want a handle they can call stop() on.
        """
        cfg = self.cfg
        try:
            from core.voice_out import VoiceOut

            tts = VoiceOut(
                cfg,
                activity=self.activity,
                log=self.log,
                on_speak=self._on_speaking,
            )
            self.services["tts"] = tts
            if tts.enabled:
                self.log.info("voice out: %s", tts.describe())
            if not tts.engine.available and tts.enabled:
                self.log.warning(
                    "no TTS engine available (%s) - GARVIS will print instead of speaking. "
                    "See the README's voice section.", tts.engine.describe(),
                )
        except Exception as exc:
            self.log.error("voice out failed to initialise: %s", exc)
            self.services["tts"] = None

        if not cfg.get("voice_in.enabled", True):
            self.log.info("voice input disabled in config; use the text prompt")
            return
        try:
            from core.voice_in import Listener, Microphone, Transcriber, WakeWordDetector

            mic = Microphone(cfg, self.log)
            transcriber = Transcriber(cfg, self.log)
            wake = WakeWordDetector(cfg, self.log)
            listener = Listener(cfg, mic, transcriber, wake, log=self.log, activity=self.activity)
            self.services["voice_in"] = listener
            self.services["wake"] = wake
            if mic.available:
                self.log.info("voice in: %s", listener.describe())
            else:
                self.log.warning("microphone unavailable: %s", mic.describe())
        except Exception as exc:
            self.log.error("voice input failed to initialise: %s", exc)

    def setup_browser(self) -> None:
        """Create the browser manager (the browser itself starts on first use)."""
        if not self.cfg.get("browser.enabled", True):
            self.log.info("browser control disabled in config")
            return
        try:
            from core.browser import BrowserManager

            manager = BrowserManager(
                self.cfg,
                activity=self.activity,
                log=self.log,
                notify=self.notify,
            )
            self.services["browser"] = manager
            self.log.info("browser manager ready (%s)", manager.describe())
        except Exception as exc:
            self.log.error("browser manager failed to initialise: %s", exc)
            self.services["browser"] = None

    def _on_speaking(self, text: str) -> None:
        """Called by TTS when an utterance actually starts."""
        if self.cfg.get("logging.level") == "DEBUG":
            self.log.debug("speaking: %s", text[:80])

    def _subscribe_killswitch(self) -> None:
        """Wire the emergency stop to everything it must halt.

        Services are looked up when the stop *happens*, not when this runs: TTS
        is rebuilt when the engine changes, the UI arrives in stage 8, and a
        stale reference would mean the stop button silently did nothing.
        """

        def on_stop(event) -> None:
            brain = self.services.get("brain")
            if brain is not None:
                try:
                    brain.interrupt()
                except Exception:
                    self.log.debug("brain.interrupt failed during stop", exc_info=True)
            tts = self.services.get("tts")
            if tts is not None:
                try:
                    tts.stop()      # drop everything queued
                    tts.pause()     # and refuse new speech until resume
                except Exception:
                    self.log.debug("tts stop failed", exc_info=True)
            browser = self.services.get("browser")
            if browser is not None and hasattr(browser, "halt_all"):
                try:
                    # Nothing may continue on a page after a stop: every open
                    # profile is handed back and needs an explicit "continue".
                    halted = browser.halt_all(f"stop everything: {event.reason}")
                    if halted:
                        self.log.warning("kill switch halted browser profiles: %s", ", ".join(halted))
                except Exception:
                    self.log.debug("browser halt failed", exc_info=True)
            ui = self.services.get("ui")
            if ui is not None and hasattr(ui, "set_status"):
                try:
                    ui.set_status("STOPPED", event.reason)
                except Exception:
                    self.log.debug("ui status failed", exc_info=True)

        self.killswitch.subscribe(on_stop)

    # -- startup -----------------------------------------------------------
    def preflight(self, quiet: bool = False) -> bool:
        """Check the basics before we pretend to be online. Returns True if usable."""
        problems: list[str] = []
        for warning in self.cfg.warnings:
            self.log.warning("config: %s", warning)

        if not quiet:
            print(BANNER)
            print(f"  GARVIS {VERSION} - local assistant. Ctrl+C or /quit to exit.\n")

        if self.cfg.cloud_fallback_enabled:
            print("  !! Cloud fallback is ENABLED in config.yaml: prompts may leave this machine.\n")

        if not self.brain.client.is_up():
            problems.append(
                f"Ollama is not answering at {self.cfg.ollama_host}.\n"
                f"     Fix: run `ollama serve` in another terminal."
            )
        else:
            version = self.brain.client.version()
            if not self.brain.client.has_model(self.cfg.model):
                problems.append(
                    f"Model '{self.cfg.model}' is not installed.\n"
                    f"     Fix: run `ollama pull {self.cfg.model}` "
                    f"(see `ollama list` for what you have)."
                )
            else:
                self.log.info("ollama %s ready, model %s", version or "?", self.cfg.model)

        if problems:
            print("Startup checks failed:\n")
            for problem in problems:
                print(f"  - {problem}")
            print()
            return False

        # Tool inventory is useful context, and proves registration worked.
        counts: dict[str, int] = {}
        for tool in self.registry.all():
            counts[tool.tier] = counts.get(tool.tier, 0) + 1
        self.log.info(
            "tools: %d (%s)", len(self.registry),
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none",
        )
        if not self.args.no_tools:
            self.log.info("permission gate: %s", "active" if self.gate else "DISABLED (fail-closed)")
        return True

    def say(self, text: str, wait: bool = True) -> None:
        """Print and, from stage 5, also speak."""
        print(f"\n{self.cfg.assistant_name}: {text}\n")
        tts = self.services.get("tts")
        if tts is not None and getattr(tts, "enabled", False):
            try:
                tts.say(text, wait=wait)
            except Exception:
                self.log.debug("tts say failed", exc_info=True)

    def notify(self, text: str) -> None:
        """Short out-of-band remark: confirmations, denials, hard stops.

        Wired into the permission gate as ``services['notifier']`` so the gate can
        tell the user what it is waiting for without going through the model.
        """
        tts = self.services.get("tts")
        if tts is not None and getattr(tts, "enabled", False):
            print(f"  [{self.cfg.assistant_name.lower()} speaks] {text}")
            try:
                tts.speak_async(text)
                return
            except Exception:
                self.log.debug("tts notify failed", exc_info=True)
        else:
            print(f"  [{self.cfg.assistant_name.lower()} speaks] {text}")

    # -- one turn ----------------------------------------------------------
    def handle_line(self, line: str, chunker: SentenceChunker | None = None) -> bool:
        """Process one user line. Returns False when the user wants to quit."""
        text = line.strip()
        if not text:
            return True
        if chunker is None:
            # Always have a chunker: it is what turns the token stream into
            # speakable sentences. Callers that keep their own reuse it.
            chunker = SentenceChunker(
                min_chars=int(self.cfg.get("voice_out.min_chunk_chars", 12)),
                max_chars=int(self.cfg.get("voice_out.max_chunk_chars", 220)),
            )

        if text.startswith("/"):
            return self._handle_command(text)

        # Spoken ways to end the session. In voice mode there is no keyboard,
        # so "goodbye" has to work; it is not sent to the model.
        if self.voice_mode and self._is_quit_phrase(text):
            self.say("Shutting down. Say my name when you need me.", wait=True)
            return False

        self.activity.event("user", text)

        tts = self.services.get("tts")
        speaking = tts is not None and getattr(tts, "enabled", False)

        def on_event(event: BrainEvent) -> None:
            if event.kind == "delta":
                print(event.text, end="", flush=True)
                # Speak whole sentences as they appear: that is what makes the
                # reply start quickly instead of waiting for the last token.
                if chunker is not None:
                    for sentence in chunker.feed(event.text):
                        if speaking:
                            tts.say_chunk(sentence)
                        else:
                            self.log.debug("speakable chunk: %s", sentence)
            elif event.kind == "tool_call":
                print(f"\n  [tool] {event.text}({_short(event.data.get('args'))})", flush=True)
            elif event.kind == "tool_result":
                ok = event.data.get("ok")
                tier = event.data.get("tier", "?")
                mark = "ok" if ok else "denied/failed"
                print(f"  [tool result: {event.text.splitlines()[0][:160]}...] ({mark}, {tier})", flush=True)
            elif event.kind == "status":
                print(f"  [{event.text}]", flush=True)
            elif event.kind == "error":
                print(f"\n  [error] {event.text}", flush=True)

        if self._handle_stop_phrases(text):
            return True

        started = time.perf_counter()
        try:
            result = self.brain.respond(text, on_event=on_event)
        except BrainError as exc:
            self.say(f"I could not reach the model. {exc}")
            self.activity.event("system", f"brain error: {exc}", ok=False)
            return True
        except KeyboardInterrupt:
            self.brain.interrupt()
            print("\n  [interrupted]")
            return True

        print()  # newline after streamed text
        if chunker is not None:
            tail = chunker.flush()
            if tail and speaking and not result.interrupted:
                tts.say_chunk(tail)
        if speaking and not self.voice_mode:
            # In text mode we still let the voice finish what it started, so a
            # long answer does not overlap the next prompt's audio.
            tts.wait(timeout=float(self.cfg.get("voice_out.max_wait_s", 120)))
        if result.interrupted:
            self.say("Stopped.", wait=False)
        elif result.error:
            self.say(f"That failed: {result.error}")
        elif not result.reply.strip():
            self.say("I have nothing to say to that.")

        self.activity.event(
            "assistant",
            result.reply,
            extra={
                "iterations": result.iterations,
                "duration_ms": round(result.duration_ms, 1),
                "tool_calls": [c["name"] for c in result.tool_calls],
                "provider": result.provider,
                "model": result.model,
            },
        )
        if self.cfg.get("logging.level") == "DEBUG":
            self.log.debug("turn took %.0f ms", (time.perf_counter() - started) * 1000)
        return True

    @staticmethod
    def _is_quit_phrase(text: str) -> bool:
        from core.killswitch import KillSwitch

        normalized = KillSwitch._normalize(text)
        return normalized in (
            "quit", "exit", "goodbye", "good bye", "bye", "shut down", "shutdown",
            "garvis quit", "garvis exit", "garvis goodbye", "garvis shut down",
            "thats all", "that is all", "stand down", "garvis stand down",
        )

    def _handle_stop_phrases(self, text: str) -> bool:
        """Spoken hard stop / resume. Handled locally, never sent to the model.

        This is deliberate: a stop command that has to survive a 2-second model
        round trip is not a stop command.
        """
        if self.killswitch.matches_kill_phrase(text):
            self.killswitch.trigger(f'spoken phrase: "{text[:60]}"', source="voice")
            self.say("Stopped everything.", wait=False)
            return True
        if self.killswitch.frozen:
            if self.killswitch.matches_resume_phrase(text):
                self.killswitch.resume("spoken resume")
                tts = self.services.get("tts")
                if tts is not None:
                    tts.resume()
                self.say("Back on. What do you need?", wait=False)
            else:
                self.notify("I am stopped. Say 'resume' when you want me to carry on.")
            return True
        return False

    def _handle_command(self, text: str) -> bool:
        """Local slash commands: they never reach the model."""
        cmd, _, rest = text[1:].partition(" ")
        cmd = cmd.strip().lower()
        rest = rest.strip()

        if cmd in ("quit", "exit", "q", "stop"):
            return False
        if cmd in ("help", "?"):
            print(
                "\n  /help              this list\n"
                "  /reset             forget the conversation (memory files stay)\n"
                "  /prompt            print the system prompt that is being used\n"
                "  /tools             list registered tools and their tiers\n"
                "  /today             what GARVIS logged today\n"
                "  /personality NAME  standard|sassy|formal|hyped|focus|chill\n"
                "  /memory            show what memory GARVIS loaded\n"
                "  /quit              exit\n"
            )
            return True
        if cmd == "reset":
            self.brain.reset_history()
            print("  [history cleared]")
            return True
        if cmd == "prompt":
            print("\n" + self.brain.build_system_prompt() + "\n")
            return True
        if cmd == "tools":
            for category, tools in sorted(self.registry.by_category().items()):
                print(f"\n  {category}:")
                for tool in tools:
                    print(f"    {tool.name:<28} [{tool.tier}] {'(read-only)' if tool.readonly else ''}")
            print()
            return True
        if cmd == "today":
            print("\n" + self.activity.today_report() + "\n")
            return True
        if cmd == "personality":
            if not rest:
                print(f"  current: {self.brain.personality}")
            elif self.brain.set_personality(rest):
                print(f"  [personality -> {rest.lower()}]")
            else:
                print(f"  unknown mode '{rest}' (standard|sassy|formal|hyped|focus|chill)")
            return True
        if cmd == "memory":
            loaded = self.memory.load_all()
            for name, body in loaded.items():
                print(f"\n  --- {name} ({len(body)} chars) ---")
                print("  " + (body[:600].replace("\n", "\n  ") if body else "(empty)"))
            print()
            return True
        print(f"  unknown command '/{cmd}' - try /help")
        return True

    # -- loops -------------------------------------------------------------
    def run_text_loop(self) -> int:
        """Stage 1 loop: typed input, streamed replies. Voice replaces this later."""
        chunker = SentenceChunker(
            min_chars=int(self.cfg.get("voice_out.min_chunk_chars", 12)),
            max_chars=int(self.cfg.get("voice_out.max_chunk_chars", 220)),
        )
        print("Type a message. /help lists local commands.\n")
        while self.running:
            try:
                line = input(f"{self.cfg.user_name.lower()}> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            try:
                if not self.handle_line(line, chunker):
                    break
            except KeyboardInterrupt:
                self.brain.interrupt()
                print("\n  [interrupted - type /quit to exit]")
            except Exception as exc:  # keep the loop alive; log the traceback
                self.log.exception("unhandled error in text loop")
                self.activity.event("system", f"unhandled error: {exc}", ok=False)
                print(f"\n  [internal error: {exc}] (see logs/garvis.log)\n")
        return 0

    def run_voice_loop(self) -> int:
        """Wake word -> listen -> think -> speak, until stopped.

        Falls back to the text loop when there is no microphone or no whisper
        model, rather than failing: a broken mic must not make GARVIS unusable.
        """
        listener = self.services.get("voice_in")
        if listener is None or not getattr(listener.mic, "available", False):
            reason = "no microphone" if listener is None else listener.mic.reason
            self.say(f"I could not start voice mode ({reason}). Switching to text.", wait=False)
            return self.run_text_loop()

        self.voice_mode = True
        if not listener.stt.load():
            self.say(
                f"Speech-to-text is not ready ({listener.stt.reason}). Switching to text mode.",
                wait=False,
            )
            self.voice_mode = False
            return self.run_text_loop()

        chunker = SentenceChunker(
            min_chars=int(self.cfg.get("voice_out.min_chunk_chars", 12)),
            max_chars=int(self.cfg.get("voice_out.max_chunk_chars", 220)),
        )
        tts = self.services.get("tts")
        wake = self.services.get("wake")
        wake_phrases = [str(p) for p in (self.cfg.get("voice_in.wake.phrases", []) or ["garvis"])]
        allow_barge_in = bool(self.cfg.get("safety.barge_in", True)) and bool(
            self.cfg.get("voice_out.allow_barge_in", True)
        )

        self.log.info("voice mode ready: %s", listener.describe())
        if getattr(self, "_shutdown_hotkey", None) is None:
            self.killswitch.install_hotkey()
            self._shutdown_hotkey = True

        greeting = str(self.cfg.get("safety.startup_greeting", "Systems online."))
        wake_name = str(self.cfg.get("app.wake_name", "Garvis"))
        self.say(f"{greeting} Say '{wake_name}' when you need me.", wait=False)

        while self.running:
            try:
                # Mute the wake detector while GARVIS is talking: with open
                # speakers it would hear itself. Barge-in still works (say the
                # wake word over the top of it) but needs a higher score.
                while tts is not None and getattr(tts, "is_speaking", False):
                    if not allow_barge_in:
                        time.sleep(0.1)
                        continue
                    if self._barge_in_check(listener, wake_phrases):
                        break
                    time.sleep(0.05)

                if self.killswitch.frozen:
                    # Still listen, so "resume" works, but do not accept commands.
                    result = listener.listen(timeout_s=6)
                    if result.ok:
                        self.handle_line(result.text, chunker)
                    continue

                wake_result = listener.wait_for_wake_word(timeout_s=None)
                if not self.running:
                    break
                if not wake_result.ok:
                    continue

                self.activity.event("wake", f"listening after {wake_result.reason}")
                self.notify("Yes?")

                heard = listener.listen(timeout_s=float(self.cfg.get("voice_in.listen.max_utterance_s", 20)))
                if not heard.ok:
                    if heard.reason not in ("heard nothing", "no words recognised"):
                        self.log.warning("listen failed: %s", heard.reason)
                    continue
                command = heard.text.strip()
                if not command:
                    continue
                self.log.info("heard: %r", command)

                if not self._wake_still_allowed(command, wake_phrases):
                    continue
                if self.killswitch.matches_kill_phrase(command) or self.killswitch.frozen:
                    self.handle_line(command, chunker)
                    continue

                self.handle_line(command, chunker)
            except KeyboardInterrupt:
                print()
                break
            except Exception as exc:
                self.log.exception("voice loop error")
                self.activity.event("system", f"voice loop error: {exc}", ok=False)
                time.sleep(0.5)
        return 0

    def _wake_still_allowed(self, text: str, wake_phrases: list[str]) -> bool:
        """Drop transcripts that are clearly GARVIS's own voice echoing back."""
        if not self.cfg.get("safety.wake_word_required", False):
            return True
        from core.voice_in import extract_wake_phrase

        woken, _ = extract_wake_phrase(text, wake_phrases)
        return woken

    def _barge_in_check(self, listener: Any, wake_phrases: list[str]) -> bool:
        """While speaking: stop immediately if the user says the wake word."""
        frame = listener.mic.read_frame(timeout=0.05)
        if frame is None:
            return False
        wake = self.services.get("wake")
        if wake is None or not wake.available:
            return False
        threshold = float(getattr(wake, "threshold", 0.5)) * 1.5
        if wake.score(frame) >= threshold:
            tts = self.services.get("tts")
            if tts is not None:
                tts.stop()
            self.activity.event("wake", "barge-in: wake word heard over speech")
            return True
        return False

    def shutdown(self) -> None:
        up = time.time() - self.started_at
        self.activity.event(
            "system",
            f"GARVIS shutting down after {up / 60:.1f} minutes",
            extra={"uptime_s": round(up, 1)},
        )
        self.log.info("shutdown after %.1f s", up)
        for service_name in ("browser", "tts", "state", "ui", "killswitch"):
            service = self.services.get(service_name)
            closer = getattr(service, "close", None) or getattr(service, "shutdown", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    self.log.debug("error closing %s", service_name, exc_info=True)


def _short(value: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garvis",
        description="GARVIS - local voice-controlled personal assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py\n"
            "  python main.py --ask 'summarise my project log'\n"
            "  python main.py --check\n"
            "  python main.py --today\n"
        ),
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--ask", metavar="TEXT", default=None, help="ask one question, print the answer, exit")
    parser.add_argument("--check", action="store_true", help="validate config and environment, then exit")
    parser.add_argument("--today", action="store_true", help="print today's activity report and exit")
    parser.add_argument("--show-prompt", action="store_true", help="print the assembled system prompt and exit")
    parser.add_argument("--model", default=None, help="override brain.model for this run")
    parser.add_argument("--ollama", default=None, help="override brain.host, e.g. http://127.0.0.1:11999")
    parser.add_argument("--personality", default=None, help="start in this tone mode")
    parser.add_argument("--no-tools", action="store_true", help="chat only; the model cannot act")
    parser.add_argument("--voice", action="store_true", help="start in voice mode (wake word + speech)")
    parser.add_argument("--text", action="store_true", help="force the text prompt (default)")
    parser.add_argument("--no-voice-out", action="store_true", help="print replies, do not speak them")
    parser.add_argument("--voice-check", action="store_true",
                        help="report TTS/STT/wake-word status and exit")
    parser.add_argument("--say", metavar="TEXT", default=None, help="speak one phrase through the TTS engine")
    parser.add_argument("--devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--browser-check", action="store_true",
                        help="start the browser, report profiles and rules, then exit")
    parser.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    parser.add_argument("--version", action="version", version=f"GARVIS {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except ConfigError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        return 2

    if args.model:
        cfg.set("brain.model", args.model)
    if args.ollama:
        cfg.set("brain.host", args.ollama)
    if args.log_level:
        cfg.set("logging.level", args.log_level.upper())
    if args.no_voice_out:
        cfg.set("voice_out.enabled", False)
    if args.text:
        cfg.set("voice_in.enabled", False)
    if args.voice:
        cfg.set("voice_in.enabled", True)

    # Audio diagnostics need no LLM either.
    if args.devices:
        from core.voice_in import Microphone

        devices = Microphone.list_devices()
        if not devices:
            print("No audio devices found (sounddevice/PortAudio missing?).")
            return 1
        print(f"{'idx':<4}{'in':<4}{'out':<4}{'rate':<8}name")
        for device in devices:
            print(f"{device['index']:<4}{device['inputs']:<4}{device['outputs']:<4}"
                  f"{int(device['default_samplerate']):<8}{device['name']}")
        print("\nPut a device name or index in config.yaml under voice_in.input_device / voice_out.device.")
        return 0

    # --check and --today need no LLM, so they work on a bare machine.
    if args.today:
        configure_activity_logger(cfg)
        from core.logger import activity

        print(activity().today_report())
        return 0

    try:
        app = Garvis(cfg, args)
    except BrainError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print(describe(cfg))
        print()
        manager = app.services.get("browser")
        if manager is not None:
            print(manager.describe())
            print()
        print("Tools registered:")
        for category, tools in sorted(app.registry.by_category().items()):
            names = ", ".join(f"{t.name}[{t.tier}]" for t in tools)
            print(f"  {category}: {names}")
        print()
        ok = app.preflight(quiet=True)
        print()
        print("RESULT:", "ready" if ok else "not ready (see above)")
        return 0 if ok else 1

    if args.show_prompt:
        print(app.brain.build_system_prompt())
        return 0

    if args.voice_check:
        return voice_check(app)

    if args.browser_check:
        return browser_check(app)

    if args.say:
        tts = app.services.get("tts")
        if tts is None or not tts.enabled:
            print("voice_out is disabled in config.yaml; nothing to speak.")
            return 1
        print(f"engine: {tts.describe()}")
        print(f"speaking: {args.say!r}")
        print(f"as spoken: {tts.preview(args.say)!r}")
        tts.say(args.say, wait=True)
        if tts.last_error:
            print(f"last error: {tts.last_error}")
            return 1
        print("done")
        return 0

    if not app.preflight():
        return 1

    if args.ask:
        app.handle_line(args.ask)
        app.shutdown()
        return 0

    app.running = True
    app.activity.event(
        "system",
        f"GARVIS started (model={cfg.model}, personality={app.brain.personality}, "
        f"gate={'on' if app.gate else 'OFF'}, tools={len(app.registry)}, "
        f"mode={'voice' if args.voice else 'text'})",
        extra={"version": VERSION, "python": sys.version.split()[0], "pid": os.getpid()},
    )
    try:
        return app.run_voice_loop() if args.voice else app.run_text_loop()
    finally:
        app.running = False
        app.shutdown()


def browser_check(app: "Garvis") -> int:
    """--browser-check: prove Chromium actually launches, with real profile paths."""
    manager = app.services.get("browser")
    print(f"{app.cfg.assistant_name} browser check")
    print("-" * 60)
    if manager is None:
        print("browser: disabled in config.yaml (browser.enabled = false)")
        return 0

    started = manager.start()
    print(f"driver : {started.message}")
    status = manager.status()
    if status.ok:
        data = status.data
        print(f"engine : {data.get('driver')}")
        print(f"allowed sites : {', '.join(data.get('allowed_sites') or []) or '(none)'}")
        print(f"never automated: {', '.join(data.get('blocked_sites') or []) or '(none)'}")
    profiles = manager.list_profiles()
    print(f"profiles ({manager.profiles_dir}): {', '.join(p['label'] for p in profiles) or 'none yet'}")
    manager.shutdown()
    print("-" * 60)
    if started.ok:
        print("RESULT: browser ready")
        return 0
    print("RESULT: browser NOT ready - install Chromium for Playwright with:")
    print("        pip install playwright && playwright install chromium")
    return 1


def voice_check(app: "Garvis") -> int:
    """--voice-check: say clearly what works and what does not."""
    print(f"{app.cfg.assistant_name} voice check")
    print("-" * 60)

    tts = app.services.get("tts")
    problems = 0
    if tts is None:
        print("output : FAILED to initialise")
        problems += 1
    else:
        print(f"output : enabled={tts.enabled} {tts.describe()}")
        if tts.enabled and not tts.engine.available:
            print(f"         !! {tts.engine.reason}")
            problems += 1
        if tts.enabled and not tts.player.available():
            print("         !! no audio playback backend found (sounddevice/simpleaudio/aplay)")
            problems += 1

    listener = app.services.get("voice_in")
    if listener is None:
        print("input  : voice_in disabled")
    else:
        print(f"input  : {listener.describe()}")
        if not listener.mic.available:
            problems += 1
        else:
            print(f"         devices: {len(type(listener.mic).list_devices())} found "
                  f"(use --devices to list them)")
        if not listener.stt.load():
            print(f"         !! speech-to-text unavailable: {listener.stt.reason}")
            problems += 1
        else:
            print(f"         speech-to-text ready: {listener.stt.describe()}")
        wake = app.services.get("wake")
        if wake is not None:
            print(f"wake   : {wake.describe()}")
            if not wake.available and wake.enabled:
                print("         (not fatal: GARVIS falls back to transcription-first wake)")

    print(f"killswitch: {app.killswitch.status()}")
    hotkey_ok = app.killswitch.install_hotkey()
    print(f"hotkey : {'armed' if hotkey_ok else 'NOT armed (see the log for the reason)'}")
    if not hotkey_ok:
        print("         the spoken phrase and the STOP button still work")
    print("-" * 60)
    print("RESULT:", "voice ready" if problems == 0 else f"{problems} problem(s) found")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

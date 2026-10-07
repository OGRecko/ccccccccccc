"""Watch the voice loop run with no microphone and no model.

    python tests/demo_voice.py

What is real here: `main.Garvis`, the wake-word decision, the utterance
segmentation, the kill switch, the permission gate, the tools, the activity log.
What is faked: the microphone (scripted frames), speech-to-text (scripted
transcripts), the model (scripted replies) and the speaker (prints what would be
spoken).

Use it to see the exact flow before spending time on drivers and models:

    wake word heard -> "Yes?" -> transcript -> tool + permission gate -> spoken reply
    "Garvis, stop everything" -> freeze -> "resume" -> unfreeze
    a YELLOW action -> spoken confirmation -> yes/no
    a RED action -> repeat the exact action -> say "confirm"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module  # noqa: E402
from core.config import Config  # noqa: E402
from core.logger import ActivityLogger, configure_activity_logger, get_logger, setup_logging  # noqa: E402
from core.voice_in import Listener  # noqa: E402
from tests.mock_ollama import MockOllama, text_chunks, tool_call_chunks  # noqa: E402

# ---------------------------------------------------------------------------
# The scripted session: (what you "say", what the model does about it)
# ---------------------------------------------------------------------------
SCRIPT: list[str] = [
    "garvis, what time is it",
    "garvis, write a note that says the demo works",
    "yes",                                          # YELLOW: spoken approval
    "garvis, what did you do today",
    "garvis, delete the note",                      # RED: needs two steps
    "delete the file voice demo dot txt",           # RED step 1: repeat it
    "confirm",                                      # RED step 2: the word
    "garvis, stop everything",                      # kill switch
    "resume",                                       # and back
    "garvis, quit",
]


class ScriptedMic:
    """Frames that look like steady speech, so the segmenter always finishes."""

    available = True
    reason = ""

    def read_frame(self, timeout: float = 0.5):
        time.sleep(0.005)
        return np.full(480, 0.3, dtype="float32")

    def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return "scripted microphone (demo)"


class ScriptedSTT:
    available = True
    reason = ""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.seen: list[str] = []

    def load(self) -> bool:
        return True

    def transcribe(self, audio, sample_rate: int = 16000) -> str:
        text = self.script.pop(0) if self.script else ""
        self.seen.append(text)
        return text

    def describe(self) -> str:
        return "scripted speech-to-text (demo)"


class ScriptedWake:
    """Scores 0.9 on every other utterance, so wake -> command -> wake -> command."""

    available = True
    threshold = 0.5
    reason = ""

    def __init__(self) -> None:
        self._tick = 0

    def reset(self) -> None:
        pass

    def score(self, frame) -> float:
        self._tick += 1
        return 0.9

    def detect(self, frame) -> bool:
        return True

    def describe(self) -> str:
        return "scripted wake word (demo)"


class PrintingEngine:
    """No audio: prints exactly what the voice would say, cleaned up."""

    name = "demo-print"
    available = True
    reason = ""

    def synthesize(self, text: str) -> bytes:
        return b""

    def describe(self) -> str:
        return "printing engine (demo)"


class PrintingVoiceOut:
    """Minimal VoiceOut stand-in: prints spoken sentences as they arrive."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = True
        self.spoken: list[str] = []
        self._paused = False

    def say_chunk(self, text: str) -> None:
        if self._paused or not str(text).strip():
            return
        from core.voice_out import clean_for_speech

        spoken = clean_for_speech(str(text), max_chars=400)
        if spoken:
            self.spoken.append(spoken)
            print(f"\n    >> SPOKEN: {spoken}")

    def say(self, text: str, wait: bool = False) -> None:
        self.say_chunk(text)

    speak = say
    speak_async = say_chunk

    def stop(self, hard: bool = True) -> int:
        return 0

    def pause(self) -> None:
        self._paused = True
        print("    >> SPEECH PAUSED")

    def resume(self) -> None:
        self._paused = False
        print("    >> SPEECH RESUMED")

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def is_speaking(self) -> bool:
        return False

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def close(self) -> None:
        pass


def main() -> int:
    cfg = Config.load(PROJECT_ROOT / "config.yaml")
    sandbox = PROJECT_ROOT / "sandbox"
    memory_dir = PROJECT_ROOT / "memory"
    logs = PROJECT_ROOT / "logs"
    sandbox.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    # Start clean so the demo is repeatable (paths stay inside the sandbox).
    for stale in sandbox.glob("voice-demo.txt"):
        stale.unlink()
    cfg.set("files.allowed_read", [str(sandbox), str(memory_dir)])
    cfg.set("files.allowed_write", [str(sandbox), str(memory_dir)])
    cfg.set("shell.default_cwd", str(sandbox))
    cfg.set("voice_in.listen.silence_end_s", 0.2)
    cfg.set("voice_in.listen.max_utterance_s", 0.6)
    cfg.set("voice_in.listen.start_timeout_s", 0.6)
    cfg.set("safety.startup_greeting", "Voice demo online.")

    setup_logging(logs / "garvis.log", level="INFO", console=False)
    configure_activity_logger(cfg)
    activity = ActivityLogger(log_dir=logs)
    import core.logger as logger_module

    logger_module._ACTIVITY = activity
    log = get_logger("demo.voice")

    def responder(payload: dict) -> list[dict]:
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            result = str(last.get("content", "")).strip()
            # Only a *leading* verdict counts: tool output may well contain the
            # word "denied" without the call having been denied.
            if result.upper().startswith(("DENIED", "BLOCKED", "ABORTED")):
                return text_chunks("That was refused by the permission gate, so I did not do it.") + [{"done": True}]
            return text_chunks("Done, and I checked the result.") + [{"done": True}]

        said = str(last.get("content", "")).lower()
        if "time" in said:
            return tool_call_chunks("clock.now", {}, preamble="Checking the clock. ") + [{"done": True}]
        if "delete" in said:
            return tool_call_chunks(
                "files.delete", {"path": "voice-demo.txt"},
                preamble="That is a delete, so I need your say-so. ",
            ) + [{"done": True}]
        if "write a note" in said or "note" in said:
            return tool_call_chunks(
                "files.write",
                {"path": "voice-demo.txt", "text": "The voice demo works."},
                preamble="Writing that down. ",
            ) + [{"done": True}]
        if "what did you do" in said:
            return tool_call_chunks("activity.today", {"limit": 5}, preamble="One moment. ") + [{"done": True}]
        return text_chunks("Understood.") + [{"done": True}]

    with MockOllama(responder=responder) as mock:
        cfg.set("brain.host", mock.url)
        args = main_module.build_parser().parse_args(["--voice"])
        app = main_module.Garvis(cfg, args)

        mic, stt = ScriptedMic(), ScriptedSTT(SCRIPT)
        app.services["voice_in"] = Listener(cfg, mic, stt, ScriptedWake(), log=log, activity=activity)
        app.services["wake"] = app.services["voice_in"].wake
        voice = PrintingVoiceOut(cfg)
        app.services["tts"] = voice

        # Answer confirmations by "voice": the scripted transcripts are what the
        # microphone would have produced, and the real gate rules still apply.
        # (In a real session the console stays in the chain as a fallback for
        # when you are at the keyboard; here it would block on stdin.)
        from core.voice_in import VoiceConfirmer

        app.gate.confirmer = VoiceConfirmer(
            cfg, app.services["voice_in"], voice_out=voice, log=log
        )

        print("=" * 74)
        print("  GARVIS voice demo - scripted ears, scripted model, real everything else")
        print("=" * 74)
        for step in SCRIPT:
            print(f"\n  << YOU SAY: {step!r}")

        print("\n" + "-" * 74)

        original_handle = app.handle_line

        def handle_and_track(text, chunker=None):
            result = original_handle(text, chunker)
            if not stt.script:
                app.running = False
            return result

        app.handle_line = handle_and_track
        app.running = True
        app.run_voice_loop()

        print("\n" + "-" * 74)
        print("  What the activity log recorded:")
        for record in activity.read_records()[-16:]:
            kind = record.get("kind")
            if kind in ("wake", "user", "permission", "tool", "assistant_spoken"):
                extra = record.get("decision") or ""
                print(f"    {kind:<17}{extra:<11}{str(record.get('message'))[:88]}")
        print("\n  Sandbox contents:")
        for entry in sorted(sandbox.iterdir()):
            print(f"    {entry.name} ({entry.stat().st_size} bytes)")
        print("\n  Spoken this session:")
        for line in voice.spoken:
            print(f"    - {line[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

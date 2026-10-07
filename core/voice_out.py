"""Voice output: text to speech, streamed by sentence, interruptible.

Engines, in order of preference (all configurable in ``config.yaml``):

* **piper** - fast, small, offline, deep male voices like ``en_US-ryan-high``.
  Runs the ``piper`` binary as a subprocess (robust, no Python deps beyond the
  binary) or falls back to the ``piper-tts`` Python package.
* **kokoro** - higher quality neural TTS, needs torch. Voice names are lower
  case, e.g. ``am_onyx`` (American male), ``am_michael``, ``bm_george`` (British).
* **pyttsx3** - the OS's built-in voice (SAPI5 / NSSpeechSynthesizer / espeak).
  Poor quality, zero setup: the safety net so GARVIS is never mute.
* **null** - prints instead of speaking (headless servers, tests).

Playback goes through sounddevice, then simpleaudio, then an OS player command
(``aplay`` / ``afplay`` / PowerShell), then finally a print.

Everything is interruptible. ``stop()`` flushes the pending queue and kills the
current utterance immediately - that is what the kill switch, the STOP button and
barge-in all call. A generation counter makes sure a late audio callback cannot
resume speaking after a stop.
"""

from __future__ import annotations

import io
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Markdown / formatting cleanup: what sounds right spoken is not what looks
# right written. Without this, TTS reads out asterisks and code fences.
# ---------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL_RE = re.compile(r"https?://\S+")
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF\u2B00-\u2BFF]"
)
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def clean_for_speech(text: str, max_chars: int = 1200) -> str:
    """Turn model output into something a human voice can read out loud."""
    if not text:
        return ""
    out = str(text)
    out = _CODE_FENCE_RE.sub(" (code block omitted) ", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _LINK_RE.sub(r"\1", out)
    out = _URL_RE.sub(lambda m: _speakable_url(m.group(0)), out)
    out = _HEADING_RE.sub("", out)
    out = _BULLET_RE.sub("", out)
    out = _TABLE_RE.sub("", out)
    out = _EMPHASIS_RE.sub(r"\2", out)
    out = _EMOJI_RE.sub("", out)
    out = out.replace("&amp;", "and").replace("&lt;", "<").replace("&gt;", ">")
    out = _MULTISPACE_RE.sub(" ", out)
    out = re.sub(r"\n\s*\n+", ". ", out)
    out = re.sub(r"\s*\n\s*", " ", out)
    out = out.strip()
    if len(out) > max_chars:
        cut = out[:max_chars]
        # Cut at a sentence boundary when there is one nearby.
        boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        out = (cut[: boundary + 1] if boundary > max_chars * 0.6 else cut).strip() + " ..."
    return out


def _speakable_url(url: str) -> str:
    """'https://example.com/a/b' -> 'example dot com slash a slash b'."""
    stripped = re.sub(r"^https?://", "", url).rstrip("/")
    stripped = stripped.replace("www.", "")
    spoken = stripped.replace(".", " dot ").replace("/", " slash ").replace("_", " ")
    return " " + re.sub(r"\s+", " ", spoken).strip() + " "


# ---------------------------------------------------------------------------
# Audio playback backends
# ---------------------------------------------------------------------------
@dataclass
class PlaybackResult:
    ok: bool
    backend: str
    error: str | None = None


class AudioPlayer:
    """Plays 16-bit PCM WAV bytes, and can stop mid-playback."""

    def __init__(self, device: str | int | None = None, volume: float = 1.0) -> None:
        self.device = None if device in (None, "", "default") else device
        self.volume = max(0.0, min(1.0, float(volume)))
        self.backend: str = "none"
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._sd = None
        self._np = None
        self._pick_backend()

    # -- backend selection -------------------------------------------------
    def _pick_backend(self) -> None:
        if self._try_sounddevice():
            self.backend = "sounddevice"
            return
        try:
            import simpleaudio  # noqa: F401

            self.backend = "simpleaudio"
            return
        except Exception:
            pass
        for name, probe in (
            ("aplay", ("aplay", "--version")),
            ("afplay", ("afplay", "-h")),
            ("powershell", ("powershell", "-Command", "$PSVersionTable.PSVersion.Major")),
        ):
            if shutil.which(name):
                try:
                    subprocess.run(probe, capture_output=True, timeout=5)
                    self.backend = name
                    return
                except Exception:
                    continue
        self.backend = "none"

    def _try_sounddevice(self) -> bool:
        try:
            import numpy  # noqa: PLC0415

            import sounddevice  # noqa: PLC0415

            self._sd = sounddevice
            self._np = numpy
            return True
        except Exception:
            return False

    def available(self) -> bool:
        return self.backend != "none"

    def describe(self) -> str:
        return f"{self.backend}" + (f" (device={self.device})" if self.device else "")

    # -- playback ----------------------------------------------------------
    def play_wav_bytes(self, data: bytes, timeout_s: float = 120.0) -> PlaybackResult:
        if not data:
            return PlaybackResult(False, self.backend, "empty audio")
        if self.backend == "sounddevice":
            return self._play_sounddevice(data, timeout_s)
        if self.backend == "simpleaudio":
            return self._play_simpleaudio(data)
        if self.backend in ("aplay", "afplay", "powershell"):
            return self._play_external(data, timeout_s)
        return PlaybackResult(False, "none", "no audio backend available")

    def _play_sounddevice(self, data: bytes, timeout_s: float) -> PlaybackResult:
        try:
            channels, rate, samples = _decode_wav(data)
            if self.volume != 1.0:
                samples = samples * self.volume
            self._sd.play(samples, rate, device=self.device, blocking=False)
            deadline = time.time() + timeout_s
            while self._sd.get_stream() is not None and self._sd.get_stream().active:
                if time.time() > deadline:
                    self._sd.stop()
                    return PlaybackResult(False, "sounddevice", "playback timed out")
                time.sleep(0.02)
            return PlaybackResult(True, "sounddevice")
        except Exception as exc:
            return PlaybackResult(False, "sounddevice", str(exc))

    def _play_simpleaudio(self, data: bytes) -> PlaybackResult:
        try:
            import simpleaudio as sa  # noqa: PLC0415

            wave_read = wave.open(io.BytesIO(data), "rb")
            play = sa.play_buffer(
                wave_read.readframes(wave_read.getnframes()),
                wave_read.getnchannels(),
                wave_read.getsampwidth(),
                wave_read.getframerate(),
            )
            with self._lock:
                self._proc = None
            play.wait_done()
            return PlaybackResult(True, "simpleaudio")
        except Exception as exc:
            return PlaybackResult(False, "simpleaudio", str(exc))

    def _play_external(self, data: bytes, timeout_s: float) -> PlaybackResult:
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(data)
                path = handle.name
            if self.backend == "aplay":
                cmd = ["aplay", "-q", path]
            elif self.backend == "afplay":
                cmd = ["afplay", path]
            else:
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    f"(New-Object Media.SoundPlayer '{path}').PlaySync()",
                ]
            with self._lock:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                proc = self._proc
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                return PlaybackResult(False, self.backend, "player timed out")
            if proc.returncode != 0:
                err = ""
                try:
                    err = (proc.stderr.read() or b"").decode(errors="replace")[:200]
                except Exception:
                    pass
                return PlaybackResult(False, self.backend, f"exit {proc.returncode} {err}")
            return PlaybackResult(True, self.backend)
        except Exception as exc:
            return PlaybackResult(False, self.backend, str(exc))
        finally:
            with self._lock:
                self._proc = None
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def stop(self) -> None:
        """Stop playback right now, whatever backend is running."""
        with self._lock:
            if self._sd is not None:
                try:
                    self._sd.stop()
                except Exception:
                    pass
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass


def _decode_wav(data: bytes) -> tuple[int, int, Any]:
    """Decode WAV bytes -> (channels, framerate, float32 numpy array [-1..1])."""
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype="uint8").astype("float32") - 128.0) / 128.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype("float32") / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return channels, rate, samples


# ---------------------------------------------------------------------------
# TTS engines
# ---------------------------------------------------------------------------
class TTSEngine:
    """Base class. ``synthesize`` returns WAV bytes (16-bit PCM)."""

    name = "base"
    available = False
    reason = ""

    def synthesize(self, text: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}{'' if self.available else ' (unavailable: ' + self.reason + ')'}"


class NullEngine(TTSEngine):
    """No audio: print what would be said. Keeps headless runs usable."""

    name = "null"
    available = True
    reason = "print-only"

    def __init__(self, logger: Any = None) -> None:
        self.logger = logger

    def synthesize(self, text: str) -> bytes:
        return b""

    def describe(self) -> str:
        return "null (text only)"


class PiperEngine(TTSEngine):
    """Piper TTS via the ``piper`` binary, or the ``piper-tts`` package."""

    name = "piper"

    def __init__(
        self,
        voice: str = "en_US-ryan-high",
        voices_dir: Path | None = None,
        exe: str = "piper",
        speed: float = 1.0,
        speaker: str | int | None = None,
    ) -> None:
        self.voice = voice
        self.voices_dir = Path(voices_dir) if voices_dir else Path("models/voices")
        self.exe = exe
        self.speed = float(speed)
        self.speaker = speaker
        self.model_path = self._find_model(voice)
        self._python_voice = None
        self.available = False
        self.reason = ""
        self._detect()

    def _find_model(self, voice: str) -> Path | None:
        for candidate in (
            self.voices_dir / f"{voice}.onnx",
            self.voices_dir / voice / f"{voice}.onnx",
            Path(voice) if voice.endswith(".onnx") else None,
        ):
            if candidate and Path(candidate).exists():
                return Path(candidate)
        return None

    def _detect(self) -> None:
        binary = shutil.which(self.exe) if not os.path.isabs(self.exe) else self.exe
        self.binary = binary
        if self.model_path is None:
            self.reason = (
                f"voice file not found: {self.voices_dir}/{self.voice}.onnx "
                f"(download a voice from https://huggingface.co/rhasspy/piper-voices)"
            )
            self.available = False
            return
        if binary:
            self.available = True
            return
        try:
            import piper  # noqa: F401,PLC0415

            self.available = True
            self.reason = "using the piper-tts python package"
            return
        except Exception:
            pass
        self.available = False
        self.reason = f"neither the '{self.exe}' binary nor the piper-tts package is available"

    def synthesize(self, text: str) -> bytes:
        if not self.available:
            raise RuntimeError(self.reason)
        length_scale = 1.0 / max(0.25, self.speed)
        if self.binary:
            cmd = [self.binary, "--model", str(self.model_path), "--length_scale", f"{length_scale:.3f}"]
            if self.speaker not in (None, ""):
                cmd += ["--speaker", str(self.speaker)]
            cmd += ["--output_file", "-"]
            completed = subprocess.run(
                cmd, input=text.encode("utf-8"), capture_output=True, timeout=120
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"piper exited {completed.returncode}: "
                    f"{completed.stderr.decode(errors='replace')[:300]}"
                )
            return completed.stdout
        # Python package path.
        if self._python_voice is None:
            from piper import PiperVoice  # noqa: PLC0415

            self._python_voice = PiperVoice.load(str(self.model_path))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._python_voice.synthesize(text, wav_file)
        return buffer.getvalue()

    def describe(self) -> str:
        return f"piper voice={self.voice} model={self.model_path}" + (
            "" if self.available else f" UNAVAILABLE: {self.reason}"
        )


class KokoroEngine(TTSEngine):
    """Kokoro TTS through the ``kokoro`` package (needs torch)."""

    name = "kokoro"

    def __init__(
        self,
        voice: str = "am_onyx",
        model_path: str | None = None,
        voices_path: str | None = None,
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> None:
        self.voice = voice
        self.model_path = model_path
        self.voices_path = voices_path
        self.speed = float(speed)
        self.lang_code = lang_code
        self._pipeline = None
        self.available = False
        self.reason = ""
        try:
            import kokoro  # noqa: F401,PLC0415

            self.available = True
        except Exception as exc:
            self.reason = f"kokoro is not installed ({exc})"

    def _get_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline  # noqa: PLC0415

            kwargs: dict[str, Any] = {"lang_code": self.lang_code}
            if self.model_path and Path(self.model_path).exists():
                kwargs["model"] = self.model_path
            if self.voices_path and Path(self.voices_path).exists():
                kwargs["voices"] = self.voices_path
            self._pipeline = KPipeline(**kwargs)
        return self._pipeline

    def synthesize(self, text: str) -> bytes:
        if not self.available:
            raise RuntimeError(self.reason)
        import numpy as np  # noqa: PLC0415

        pipeline = self._get_pipeline()
        chunks: list[Any] = []
        for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
            if audio is None:
                continue
            array = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(array)
        if not chunks:
            raise RuntimeError("kokoro produced no audio")
        samples = np.concatenate(chunks)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def describe(self) -> str:
        return f"kokoro voice={self.voice}" + ("" if self.available else f" UNAVAILABLE: {self.reason}")


class Pyttsx3Engine(TTSEngine):
    """Last-resort local voice: whatever the OS ships with."""

    name = "pyttsx3"

    def __init__(self, rate: int = 175, volume: float = 0.9, voice_hint: str = "") -> None:
        self.rate = int(rate)
        self.volume = float(volume)
        self.voice_hint = voice_hint
        self.available = False
        self.reason = ""
        try:
            import pyttsx3  # noqa: F401,PLC0415

            self.available = True
        except Exception as exc:
            self.reason = f"pyttsx3 is not installed ({exc})"

    def synthesize(self, text: str) -> bytes:
        raise RuntimeError(
            "pyttsx3 speaks directly and cannot return WAV bytes; "
            "VoiceOut uses engine.speak_direct() for it"
        )

    def speak_direct(self, text: str) -> None:
        import pyttsx3  # noqa: PLC0415

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        if self.voice_hint:
            for voice in engine.getProperty("voices"):
                if self.voice_hint.lower() in (voice.name or "").lower():
                    engine.setProperty("voice", voice.id)
                    break
        engine.say(text)
        engine.runAndWait()

    def describe(self) -> str:
        return "pyttsx3 (OS voice)" + ("" if self.available else f" UNAVAILABLE: {self.reason}")


# ---------------------------------------------------------------------------
# VoiceOut: the queue, the worker and the interrupts
# ---------------------------------------------------------------------------
@dataclass
class SpokenChunk:
    text: str
    generation: int = 0
    enqueued_at: float = field(default_factory=time.time)


class VoiceOut:
    """Speaks sentences as they arrive from the model.

    Usage from the brain's streaming callback::

        voice_out.say_chunk("All systems nominal.")   # returns immediately
        voice_out.wait(timeout=30)                    # optional
        voice_out.stop()                              # barge-in / kill switch
    """

    def __init__(
        self,
        cfg: Any,
        activity: Any = None,
        log: Any = None,
        engine: TTSEngine | None = None,
        player: AudioPlayer | None = None,
        on_speak: Any = None,
    ) -> None:
        self.cfg = cfg
        self.activity = activity
        self.log = log
        self.enabled = bool(cfg.get("voice_out.enabled", True))
        self.clean_text = True
        self.max_chunk_chars = int(cfg.get("voice_out.max_chunk_chars", 220))
        self.on_speak = on_speak  # callback(text) fired when an utterance really starts

        volume = float(cfg.get("voice_out.volume", 0.9))
        device = cfg.get("voice_out.device", "default")
        self.player = player or AudioPlayer(device=device, volume=volume)
        self.engine = engine or self._build_engine(cfg, log)

        self._queue: queue.Queue[SpokenChunk | None] = queue.Queue()
        self._generation = 0
        self._speaking = threading.Event()
        self._stopped = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._current: SpokenChunk | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.spoken_history: list[str] = []
        self.last_error: str | None = None

    # -- construction ------------------------------------------------------
    @staticmethod
    def _build_engine(cfg: Any, log: Any = None) -> TTSEngine:
        engine_name = str(cfg.get("voice_out.engine", "piper")).lower()
        voice = str(cfg.get("voice_out.voice", "en_US-ryan-high"))
        speed = float(cfg.get("voice_out.speed", 1.0))
        voices_dir = cfg.resolve_path(cfg.get("voice_out.voices_dir", "models/voices"))
        if engine_name == "kokoro":
            return KokoroEngine(
                voice=voice,
                model_path=str(cfg.resolve_path(cfg.get("voice_out.kokoro_model", "models/kokoro/kokoro-v1.0.onnx"))),
                voices_path=str(cfg.resolve_path(cfg.get("voice_out.kokoro_voices", "models/kokoro/voices-v1.0.bin"))),
                speed=speed,
            )
        if engine_name == "pyttsx3":
            return Pyttsx3Engine(volume=float(cfg.get("voice_out.volume", 0.9)), voice_hint=voice)
        return PiperEngine(
            voice=voice,
            voices_dir=voices_dir,
            exe=str(cfg.get("voice_out.piper_exe", "piper")),
            speed=speed,
        )

    def describe(self) -> str:
        return f"engine={self.engine.describe()} output={self.player.describe()}"

    # -- lifecycle ---------------------------------------------------------
    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, name="garvis-tts", daemon=True)
                self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._idle.set()
                return
            generation = item.generation
            if generation != self._generation:
                continue  # a stop happened while this was queued
            self._current = item
            self._speaking.set()
            self._idle.clear()
            try:
                self._speak_one(item.text, generation)
            except Exception as exc:  # never let TTS kill the loop
                self.last_error = str(exc)
                if self.log:
                    self.log.warning("TTS failed: %s", exc)
                if self.activity:
                    self.activity.event("system", f"tts failed: {exc}", ok=False)
            finally:
                self._speaking.clear()
                self._current = None
                if self._queue.empty():
                    self._idle.set()

    def _speak_one(self, text: str, generation: int) -> None:
        spoken = clean_for_speech(text, max_chars=self.max_chunk_chars * 2) if self.clean_text else text
        if not spoken:
            return
        if self.on_speak:
            try:
                self.on_speak(spoken)
            except Exception:
                pass
        self.spoken_history.append(spoken)
        if self.activity:
            self.activity.event("assistant_spoken", spoken[:500])

        if isinstance(self.engine, Pyttsx3Engine):
            # Blocks until the OS finishes speaking; stop() cannot interrupt this
            # backend, which is exactly why it is the last resort.
            self.engine.speak_direct(spoken)
            return

        started = time.perf_counter()
        audio = self.engine.synthesize(spoken)
        if generation != self._generation:
            return
        if not audio:
            if self.log:
                self.log.debug("no audio produced for %r", spoken[:60])
            return
        result = self.player.play_wav_bytes(audio)
        if not result.ok:
            self.last_error = result.error
            if self.log:
                self.log.warning("playback failed (%s): %s", result.backend, result.error)
        elif self.log:
            self.log.debug("spoke %d chars in %.0f ms", len(spoken), (time.perf_counter() - started) * 1000)

    # -- public API --------------------------------------------------------
    def say(self, text: str, wait: bool = False) -> None:
        """Queue a whole utterance."""
        self.say_chunk(text)
        if wait:
            self.wait()

    def say_chunk(self, text: str) -> None:
        """Queue one sentence/chunk. Returns immediately (non-blocking)."""
        if not self.enabled or not text or not str(text).strip():
            return
        if self._stopped.is_set():
            return
        self._ensure_thread()
        chunk = SpokenChunk(text=str(text).strip(), generation=self._generation)
        self._idle.clear()
        self._queue.put(chunk)

    # Convenience aliases used by main.py and the gate's notifier.
    speak = say
    speak_async = say_chunk

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def wait(self, timeout: float | None = 60.0) -> bool:
        """Block until the queue is empty. True if it drained."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if self._idle.is_set() and self._queue.empty() and not self._speaking.is_set():
                return True
            if deadline is not None and time.time() > deadline:
                return False
            time.sleep(0.02)

    def stop(self, hard: bool = True) -> int:
        """Stop speaking now and drop anything queued. Returns dropped count."""
        self._generation += 1
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if hard:
            self.player.stop()
        self._idle.set()
        if self.log:
            self.log.debug("tts stop: dropped %d queued chunk(s)", dropped)
        return dropped

    def pause(self) -> None:
        """Emergency stop that also refuses new speech until resume()."""
        self.stop()
        self._stopped.set()

    def resume(self) -> None:
        self._stopped.clear()

    @property
    def paused(self) -> bool:
        return self._stopped.is_set()

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def preview(self, text: str) -> str:
        """What would actually be spoken for this text (used by tests and --voice-check)."""
        return clean_for_speech(text, max_chars=self.max_chunk_chars * 2)


__all__ = [
    "AudioPlayer",
    "KokoroEngine",
    "NullEngine",
    "PiperEngine",
    "Pyttsx3Engine",
    "TTSEngine",
    "VoiceOut",
    "clean_for_speech",
]

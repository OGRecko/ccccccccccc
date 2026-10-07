"""Voice input: microphone, wake word, speech-to-text, spoken confirmations.

Pipeline
--------
1. :class:`Microphone` streams 16 kHz mono frames from sounddevice into a
   callback that hands them to whoever is listening.
2. :class:`UtteranceSegmenter` is pure logic (no audio dependency, fully
   testable): it watches frame energy and decides when an utterance starts and
   ends. That is what makes wake word -> listen -> stop-on-silence reliable.
3. :class:`WakeWordDetector` runs openWakeWord on those frames ("Garvis").
   When no model is available we fall back to transcription-first mode, where
   any "garvis ..." phrase in the transcript wakes GARVIS - slower but honest.
4. :class:`Transcriber` wraps faster-whisper: it owns the model load, device and
   compute-type selection, and the VAD filter.
5. :class:`VoiceConfirmer` implements the permission gate's Confirmer protocol,
   so "yes" (YELLOW) and "repeat the action, then say confirm" (RED) actually
   work by voice.

Everything degrades: with no microphone, no sounddevice, or no whisper model,
the module still imports and reports *why* it cannot listen, instead of
crashing the assistant.
"""

from __future__ import annotations

import array
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

SAMPLE_RATE = 16000
FRAME_MS = 30  # openWakeWord and webrtcvad both want 10/20/30 ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


# ---------------------------------------------------------------------------
# Small audio helpers (numpy optional: these work with plain sequences too)
# ---------------------------------------------------------------------------
def frame_rms(frame: Any) -> float:
    """Root-mean-square amplitude of a frame, normalised to 0.0-1.0.

    Works with a numpy array (float32 in [-1, 1] or int16) or any Python
    sequence of numbers, so the segmentation logic is testable without numpy.
    """
    if frame is None:
        return 0.0
    try:
        values = frame.tolist() if hasattr(frame, "tolist") else list(frame)
    except TypeError:
        return 0.0
    if not values:
        return 0.0
    peak = max(abs(float(v)) for v in values[:64]) if values else 0.0
    scale = 32768.0 if peak > 1.5 else 1.0  # looks like int16 data
    total = 0.0
    for value in values:
        normalised = float(value) / scale
        total += normalised * normalised
    return math.sqrt(total / len(values))


def pcm16_to_float_list(frame: bytes) -> list[float]:
    """Decode raw 16-bit PCM bytes into floats in [-1, 1]."""
    samples = array.array("h")
    samples.frombytes(frame)
    return [value / 32768.0 for value in samples]


# ---------------------------------------------------------------------------
# Utterance segmentation (pure logic - the part worth unit testing)
# ---------------------------------------------------------------------------
@dataclass
class UtteranceSegmenter:
    """Decides where an utterance starts and ends from frame energies.

    States: ``idle`` -> ``speech`` -> ``idle``. A frame counts as speech when its
    RMS is above ``speech_threshold`` (adaptively raised if the room is noisy).
    The utterance ends after ``silence_end_s`` of continuous quiet, or at
    ``max_utterance_s``. Utterances shorter than ``min_utterance_s`` are
    discarded as coughs/clicks.
    """

    sample_rate: int = SAMPLE_RATE
    frame_ms: int = FRAME_MS
    speech_threshold: float = 0.015
    noise_floor: float = 0.006
    silence_end_s: float = 0.9
    max_utterance_s: float = 20.0
    min_utterance_s: float = 0.35
    hangover_s: float = 0.15

    _speaking: bool = False
    _frames: list = field(default_factory=list)
    _silence_frames: int = 0
    _speech_frames: int = 0
    _started_at: float = 0.0
    _adaptive: float = 0.0

    def __post_init__(self) -> None:
        self.frame_s = self.frame_ms / 1000.0
        self._reset_state()

    def _reset_state(self) -> None:
        self._frames = []
        self._silence_frames = 0
        self._speech_frames = 0
        self._speaking = False
        self._started_at = 0.0

    # -- configuration from config.yaml ------------------------------------
    @classmethod
    def from_config(cls, cfg: Any) -> "UtteranceSegmenter":
        return cls(
            sample_rate=int(cfg.get("voice_in.sample_rate", SAMPLE_RATE)),
            silence_end_s=float(cfg.get("voice_in.listen.silence_end_s", 0.9)),
            max_utterance_s=float(cfg.get("voice_in.listen.max_utterance_s", 20.0)),
            min_utterance_s=float(cfg.get("voice_in.listen.min_utterance_s", 0.35)),
        )

    @property
    def speaking(self) -> bool:
        return self._speaking

    def threshold_now(self) -> float:
        """Effective threshold: adapts upwards in a noisy room."""
        return max(self.speech_threshold, self._adaptive * 3.0)

    def feed(self, frame: Any) -> Any:
        """Feed one frame.

        Returns the finished utterance (list of frames) when the utterance just
        ended, otherwise ``None``. Frames of a too-short utterance are discarded.
        """
        energy = frame_rms(frame)
        threshold = self.threshold_now()
        is_speech = energy >= threshold

        # Track the noise floor on EVERY frame, not just while idle. Otherwise a
        # room whose hiss is just loud enough to look like speech locks the
        # estimate at zero and GARVIS never notices it is in a noisy room.
        self._adaptive = 0.95 * self._adaptive + 0.05 * min(energy, self._adaptive * 4 + 0.5)

        if not self._speaking:
            if is_speech:
                self._speaking = True
                self._frames = [frame]
                self._speech_frames = 1
                self._silence_frames = 0
                self._started_at = time.time()
            return None

        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        duration = len(self._frames) * self.frame_s
        silence_s = self._silence_frames * self.frame_s
        first_utterance = len(self._frames) == 1 and duration <= self.frame_s
        if first_utterance:
            return None
        if silence_s >= self.silence_end_s or duration >= self.max_utterance_s:
            return self.end()
        return None

    def end(self) -> Any:
        """Force the end of the current utterance and return it."""
        frames, self._frames = self._frames, []
        speech_frames = self._speech_frames
        self._reset_state()
        if not frames:
            return None
        duration = speech_frames * self.frame_s
        if duration < self.min_utterance_s:
            return None  # too short: a click, a cough, a door
        return list(frames)


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
def normalize_wake_text(text: str) -> str:
    """Lowercase, strip punctuation - so 'Garvis, are you there?' matches."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def extract_wake_phrase(text: str, phrases: Iterable[str]) -> tuple[bool, str]:
    """Check whether a transcript starts with a wake phrase.

    Returns ``(woken, remainder)``. "(hey) garvis, what time is it" ->
    ``(True, "what time is it")``.
    """
    cleaned = normalize_wake_text(text)
    if not cleaned:
        return False, ""
    for phrase in sorted((normalize_wake_text(p) for p in phrases), key=len, reverse=True):
        if not phrase:
            continue
        if cleaned == phrase:
            return True, ""
        if cleaned.startswith(phrase + " "):
            return True, cleaned[len(phrase) :].strip()
    return False, cleaned


class WakeWordDetector:
    """openWakeWord wrapper with an honest fallback."""

    def __init__(self, cfg: Any, log: Any = None) -> None:
        self.cfg = cfg
        self.log = log
        self.enabled = bool(cfg.get("voice_in.wake.enabled", True))
        self.engine = str(cfg.get("voice_in.wake.engine", "openwakeword")).lower()
        self.model_name = str(cfg.get("voice_in.wake.model", "garvis"))
        self.builtin_fallback = str(cfg.get("voice_in.wake.builtin_fallback", "hey_jarvis"))
        self.threshold = float(cfg.get("voice_in.wake.threshold", 0.5))
        self.available = False
        self.reason = ""
        self.active_model = ""
        self._model = None
        self._last_score = 0.0
        self._load()

    def _load(self) -> None:
        if not self.enabled or self.engine in ("", "none"):
            self.reason = "wake word disabled in config"
            return
        try:
            import openwakeword  # noqa: PLC0415
            from openwakeword.model import Model  # noqa: PLC0415
        except Exception as exc:
            self.reason = (
                f"openwakeword is not installed ({exc}). Until it is, GARVIS transcribes "
                f"everything and wakes on the phrase '{self.model_name}' in the transcript."
            )
            return

        candidates: list[tuple[str, str]] = []
        model_path = Path(str(self.model_name))
        if model_path.exists():
            candidates.append((str(model_path), model_path.stem))
        else:
            candidates.append((self.model_name, self.model_name))
            candidates.append((self.builtin_fallback, self.builtin_fallback))

        for name, label in candidates:
            try:
                self._model = Model(wakeword_models=[name], inference_framework="onnx")
                self.active_model = label
                self.available = True
                if name == self.builtin_fallback and self.model_name != self.builtin_fallback:
                    self.reason = (
                        f"custom model '{self.model_name}' was not found; using the built-in "
                        f"'{self.builtin_fallback}' model. See the README for training your own."
                    )
                return
            except Exception as exc:  # try the next candidate
                self.reason = str(exc)
                continue
        self.reason = self.reason or "could not load any wake word model"

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass
        self._last_score = 0.0

    @property
    def last_score(self) -> float:
        return self._last_score

    def score(self, frame: Any) -> float:
        """Feed one 30 ms frame; returns the best model score."""
        if not self.available or self._model is None:
            return 0.0
        try:
            import numpy as np  # noqa: PLC0415

            audio = frame if isinstance(frame, np.ndarray) else np.asarray(frame, dtype=np.int16)
            if audio.dtype != np.int16:
                audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            predictions = self._model.predict(audio)
            if isinstance(predictions, dict) and predictions:
                self._last_score = float(max(predictions.values()))
            return self._last_score
        except Exception as exc:
            if self.log:
                self.log.debug("wake word scoring failed: %s", exc)
            return 0.0

    def detect(self, frame: Any) -> bool:
        return self.score(frame) >= self.threshold

    def describe(self) -> str:
        if not self.enabled:
            return "wake word disabled"
        if self.available:
            model = self.active_model or self.model_name
            extra = f" ({self.reason})" if self.reason else ""
            return f"openWakeWord model={model} threshold={self.threshold}{extra}"
        return f"UNAVAILABLE: {self.reason}"


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------
class Transcriber:
    """faster-whisper wrapper: load once, transcribe many."""

    def __init__(self, cfg: Any, log: Any = None) -> None:
        self.cfg = cfg
        self.log = log
        self.engine = str(cfg.get("voice_in.stt.engine", "faster_whisper")).lower()
        self.model_name = str(cfg.get("voice_in.stt.model", "distil-large-v3"))
        self.device_pref = str(cfg.get("voice_in.stt.device", "auto")).lower()
        self.compute_pref = str(cfg.get("voice_in.stt.compute_type", "auto")).lower()
        self.language = str(cfg.get("voice_in.stt.language", "en")) or None
        self.beam_size = int(cfg.get("voice_in.stt.beam_size", 1))
        self.vad_filter = bool(cfg.get("voice_in.stt.vad_filter", True))
        self.available = False
        self.reason = ""
        self.device = "cpu"
        self.compute_type = "int8"
        self._model = None
        self._lock = threading.RLock()

    def resolve_device(self) -> tuple[str, str]:
        """Pick device + compute type. CUDA if torch/cuda is usable, else CPU."""
        device = self.device_pref
        compute = self.compute_pref
        if device == "auto":
            device = "cpu"
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    def load(self) -> bool:
        """Load the whisper model. Expensive the first time: call it early."""
        if self._model is not None:
            return True
        self.device, self.compute_type = self.resolve_device()
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except Exception as exc:
            self.available = False
            self.reason = f"faster-whisper is not installed ({exc})"
            return False
        try:
            started = time.perf_counter()
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
            self.available = True
            self.reason = ""
            if self.log:
                self.log.info(
                    "whisper model %s loaded on %s/%s in %.1fs",
                    self.model_name, self.device, self.compute_type, time.perf_counter() - started,
                )
            return True
        except Exception as exc:
            self.available = False
            self.reason = (
                f"could not load whisper model '{self.model_name}' on {self.device}/{self.compute_type}: {exc}"
            )
            return False

    def transcribe(self, audio: Any, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe a numpy float32 array (or a list of floats) to text."""
        if not self.load():
            raise RuntimeError(self.reason)
        import numpy as np  # noqa: PLC0415

        samples = audio if isinstance(audio, np.ndarray) else np.asarray(audio, dtype=np.float32)
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        with self._lock:
            segments, info = self._model.transcribe(
                samples,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        if self.log:
            self.log.debug("stt (%.2fs audio): %r", len(samples) / sample_rate, text[:200])
        return text

    def describe(self) -> str:
        state = f"{self.model_name} on {self.device}/{self.compute_type}"
        return state if self.available else f"{state} (not loaded: {self.reason or 'lazy'})"


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------
class Microphone:
    """Streams 16 kHz mono frames from sounddevice."""

    def __init__(self, cfg: Any, log: Any = None, device: Any = None) -> None:
        self.cfg = cfg
        self.log = log
        self.sample_rate = int(cfg.get("voice_in.sample_rate", SAMPLE_RATE))
        self.frame_ms = FRAME_MS
        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        configured = cfg.get("voice_in.input_device", "default")
        self.device = device if device is not None else (None if configured in ("default", "") else configured)
        self.available = False
        self.reason = ""
        self._sd = None
        self._np = None
        self._stream = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=200)
        self._probe()

    def _probe(self) -> None:
        try:
            import numpy  # noqa: PLC0415

            import sounddevice  # noqa: PLC0415

            self._sd, self._np = sounddevice, numpy
            self.available = True
        except Exception as exc:
            self.available = False
            self.reason = (
                f"microphone backend unavailable ({exc}). "
                f"Install sounddevice (and PortAudio: `apt install libportaudio2` on Linux)."
            )

    # -- device listing ----------------------------------------------------
    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except Exception:
            return []
        found: list[dict[str, Any]] = []
        try:
            for index, device in enumerate(sd.query_devices()):
                found.append(
                    {
                        "index": index,
                        "name": device.get("name", f"device {index}"),
                        "inputs": device.get("max_input_channels", 0),
                        "outputs": device.get("max_output_channels", 0),
                        "default_samplerate": device.get("default_samplerate", 0),
                    }
                )
        except Exception:
            return []
        return found

    # -- streaming ---------------------------------------------------------
    def start(self) -> bool:
        if not self.available:
            return False
        if self._stream is not None:
            return True
        try:
            self._stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_samples,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception as exc:
            self.reason = f"could not open the microphone: {exc}"
            self._stream = None
            return False

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:  # noqa: ARG002
        if status and self.log:
            self.log.debug("audio status: %s", status)
        try:
            # Copy: sounddevice reuses the buffer.
            self._queue.put_nowait(self._np.copy(indata[:, 0]))
        except queue.Full:
            # The consumer is behind: drop the oldest frame rather than stall audio.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(self._np.copy(indata[:, 0]))
            except Exception:
                pass
        except Exception:
            pass

    def read_frame(self, timeout: float = 0.5) -> Any | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def close(self) -> None:
        self.stop()

    def describe(self) -> str:
        if not self.available:
            return f"UNAVAILABLE: {self.reason}"
        device = self.device if self.device is not None else "system default"
        return f"{self.sample_rate} Hz mono, device={device}, frame={self.frame_ms} ms"


# ---------------------------------------------------------------------------
# Listening: wake word -> utterance -> text
# ---------------------------------------------------------------------------
@dataclass
class ListenResult:
    ok: bool
    text: str = ""
    reason: str = ""
    duration_s: float = 0.0
    wake_score: float = 0.0


class Listener:
    """High-level "give me a spoken utterance" service."""

    def __init__(
        self,
        cfg: Any,
        microphone: Microphone,
        transcriber: Transcriber,
        wakeword: WakeWordDetector | None = None,
        log: Any = None,
        activity: Any = None,
    ) -> None:
        self.cfg = cfg
        self.mic = microphone
        self.stt = transcriber
        self.wake = wakeword
        self.log = log
        self.activity = activity
        self.segmenter = UtteranceSegmenter.from_config(cfg)
        self.wake_phrases = [str(p) for p in (cfg.get("voice_in.wake.phrases", []) or ["garvis"])]

    # -- helpers -----------------------------------------------------------
    def _collect_frames(self, frames: list[Any]) -> Any:
        """Stack frames into one array (or a plain list if numpy is missing)."""
        try:
            import numpy as np  # noqa: PLC0415

            return np.concatenate([np.asarray(f) for f in frames])
        except Exception:
            flat: list[float] = []
            for frame in frames:
                flat.extend(frame.tolist() if hasattr(frame, "tolist") else list(frame))
            return flat

    def _record_utterance(self, timeout_s: float, first_frame: Any | None = None) -> tuple[Any, float] | None:
        """Record until the segmenter says the utterance ended.

        ``first_frame`` is used by barge-in: the frame that interrupted playback
        is the first frame of the user's new utterance and must not be dropped.
        """
        self.segmenter = UtteranceSegmenter.from_config(self.cfg)
        started = time.time()
        frames: list[Any] = []
        if first_frame is not None:
            frames.append(first_frame)
        finished: Any = None
        while time.time() - started < timeout_s:
            frame = self.mic.read_frame(timeout=0.5)
            if frame is None:
                if frames:
                    break  # stream stalled mid-utterance: use what we have
                continue
            finished = self.segmenter.feed(frame)
            frames.append(frame)
            if finished:
                frames = finished
                break
        else:
            # Timeout while someone is still talking: take what we have.
            finished = self.segmenter.end()
            if finished:
                frames = finished
        if not frames:
            return None
        audio = self._collect_frames(frames)
        return audio, time.time() - started

    # -- public API --------------------------------------------------------
    def wait_for_wake_word(self, timeout_s: float | None = None, on_frame: Callable[[Any], None] | None = None) -> ListenResult:
        """Block until the wake word is heard (or the timeout expires)."""
        if not self.mic.available:
            return ListenResult(False, reason=self.mic.reason)
        if self.wake is None or not self.wake.available:
            if self.log:
                self.log.info(
                    "wake word model unavailable (%s); using transcription-first mode: "
                    "say 'garvis, ...' and the phrase itself wakes me.",
                    getattr(self.wake, "reason", "no detector"),
                )
            return self._wait_without_wakeword(timeout_s, on_frame)

        self.mic.drain()
        self.wake.reset()
        started = time.time()
        best = 0.0
        while timeout_s is None or time.time() - started < timeout_s:
            frame = self.mic.read_frame(timeout=0.5)
            if frame is None:
                continue
            if on_frame:
                on_frame(frame)
            score = self.wake.score(frame)
            best = max(best, score)
            if score >= self.wake.threshold:
                if self.activity:
                    self.activity.event("wake", f"wake word detected (score {score:.2f})")
                return ListenResult(True, reason="wake", wake_score=score, duration_s=time.time() - started)

        return ListenResult(False, reason="no wake word heard", wake_score=best, duration_s=time.time() - started)

    def _wait_without_wakeword(
        self, timeout_s: float | None, on_frame: Callable[[Any], None] | None
    ) -> ListenResult:
        """Fallback: record utterances and check their transcript for 'garvis'."""
        started = time.time()
        while timeout_s is None or time.time() - started < timeout_s:
            recorded = self._record_utterance(min(12.0, timeout_s or 12.0))
            if not recorded:
                continue
            audio, _ = recorded
            try:
                text = self.stt.transcribe(audio)
            except Exception as exc:
                if self.log:
                    self.log.debug("stt failed while waiting for wake word: %s", exc)
                continue
            woken, remainder = extract_wake_phrase(text, self.wake_phrases)
            if woken:
                if self.activity:
                    self.activity.event("wake", f"wake phrase in transcript: {text!r}")
                return ListenResult(True, text=remainder, reason="wake-in-transcript")
        return ListenResult(False, reason="no wake word heard")

    def listen(self, timeout_s: float | None = None, prompt: str | None = None) -> ListenResult:
        """Record one utterance and transcribe it (no wake word needed)."""
        if not self.mic.available:
            return ListenResult(False, reason=self.mic.reason)
        if prompt and self.log:
            self.log.debug("listening: %s", prompt)
        start_timeout = float(self.cfg.get("voice_in.listen.start_timeout_s", 8))
        recorded = self._record_utterance(start_timeout if timeout_s is None else timeout_s)
        if not recorded:
            return ListenResult(False, reason="heard nothing")
        audio, duration = recorded
        try:
            text = self.stt.transcribe(audio)
        except Exception as exc:
            return ListenResult(False, reason=f"speech-to-text failed: {exc}", duration_s=duration)
        if not text.strip():
            return ListenResult(False, reason="no words recognised", duration_s=duration)
        return ListenResult(True, text=text, duration_s=duration)

    def transcribe_array(self, audio: Any) -> str:
        """Transcribe audio you already have (used by the self-test)."""
        return self.stt.transcribe(audio)

    def close(self) -> None:
        self.mic.close()

    def describe(self) -> str:
        bits = [f"mic: {self.mic.describe()}", f"stt: {self.stt.describe()}"]
        bits.append(f"wake: {self.wake.describe() if self.wake else 'not configured'}")
        return " | ".join(bits)


# ---------------------------------------------------------------------------
# Spoken confirmations for the permission gate
# ---------------------------------------------------------------------------
class VoiceConfirmer:
    """Confirmer that listens for the answer instead of reading it.

    Implements the same protocol as the console confirmer, and is bound by the
    same rule: for a RED action the *gate* checks that the user repeated the
    action and said the confirm word. This class only reports what it heard.
    """

    name = "voice"

    def __init__(
        self,
        cfg: Any,
        listener: Listener,
        voice_out: Any = None,
        log: Any = None,
        accept_words: Iterable[str] | None = None,
        deny_words: Iterable[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.listener = listener
        self.voice_out = voice_out
        self.log = log
        self.accept = [w.lower() for w in (accept_words or ["yes", "yeah", "yep", "ok", "okay", "go ahead", "do it", "affirmative", "confirm"])]
        self.deny = [w.lower() for w in (deny_words or ["no", "nope", "stop", "cancel", "abort", "deny"])]
        self.available = listener.mic.available and not getattr(listener.wake, "wake_word_required", False)
        self.last_heard = ""
        self.requests: list[Any] = []

    def _ask_aloud(self, prompt: str) -> None:
        if self.voice_out is not None and getattr(self.voice_out, "enabled", False):
            try:
                self.voice_out.say(prompt)
                self.voice_out.wait(timeout=15)
            except Exception:
                pass

    def confirm(self, request: Any) -> Any:
        """Listen for one utterance and report it to the gate."""
        from core.permissions import ConfirmAnswer  # local import: avoids a cycle

        timeout_s = float(getattr(request, "timeout_s", 20) or 20)
        needed = "repeat" if getattr(request, "stage", "single") == "repeat" else "answer"
        spoken_form = getattr(request, "challenge_spoken", "") or ""
        challenge = spoken_form or getattr(request, "challenge", "") or getattr(request, "exact", "")
        prompt = {
            "repeat": f"Repeat after me: {challenge}",
            "confirm": f"Say {getattr(request, 'require_phrase', 'confirm')} to go ahead.",
        }.get(getattr(request, "stage", "single"), "Yes or no?")

        if not self.available:
            return ConfirmAnswer(False, method=self.name, note="voice confirmation is not available")
        if getattr(request, "sensitive", False):
            # A secret-bearing action must be confirmed on screen: speaking the
            # challenge aloud would leak it, and masking it makes it useless.
            # `unavailable` hands this one to the next channel (console/UI)
            # instead of counting as a refusal.
            return ConfirmAnswer(
                False, method=self.name, unavailable=True,
                note="this action carries a secret; it must be confirmed on screen",
            )

        # Speak only when the channel adds information the user has not heard:
        # the gate already asks the yes/no question and announces the confirm
        # word, but only this channel knows the *phrase* to repeat.
        if getattr(request, "stage", "single") == "repeat":
            self._ask_aloud(prompt)
        # The mic may still be streaming while TTS plays: drop the backlog first.
        self.listener.mic.drain()
        result = self.listener.listen(timeout_s=timeout_s, prompt=prompt)
        if not result.ok:
            return ConfirmAnswer(False, method=self.name, timed_out=True, note=result.reason)

        heard = result.text.strip()
        self.last_heard = heard
        self.requests.append(request)
        lowered = heard.lower()

        if self.log:
            self.log.info("voice confirmation heard: %r", heard[:120])

        if getattr(request, "stage", "single") in ("repeat", "confirm"):
            # Hand the words to the gate: it owns the RED comparison rules.
            return ConfirmAnswer(True, method=self.name, text=heard)

        # YELLOW: a plain yes/no answer, decided here.
        if any(word in lowered for word in self.deny):
            return ConfirmAnswer(False, method=self.name, text=heard, note="declined by voice")
        if any(word in lowered for word in self.accept):
            return ConfirmAnswer(True, method=self.name, text=heard)
        return ConfirmAnswer(False, method=self.name, text=heard, note=f"'{heard}' is not a yes")

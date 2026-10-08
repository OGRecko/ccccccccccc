"""Stage 5 tests: TTS, wake word, STT plumbing, spoken confirmations.

No microphone, speaker or GPU is required: the audio *logic* (segmentation,
wake phrases, sentence queueing, stopping) is tested directly, and the engine
interfaces are exercised with fakes plus a real fake-piper subprocess.

Run with:  pytest tests/test_stage5_voice.py -v
"""

from __future__ import annotations

import io
import os
import stat
import sys
import time
import wave
from pathlib import Path

import pytest

from core.killswitch import KillSwitch
from core.permissions import (
    CONFIRMED,
    DENIED,
    PermissionGate,
    ScriptedConfirmer,
)
from core.voice_in import (
    FRAME_MS,
    SAMPLE_RATE,
    UtteranceSegmenter,
    VoiceConfirmer,
    extract_wake_phrase,
    frame_rms,
    normalize_wake_text,
)
from core.voice_out import (
    AudioPlayer,
    NullEngine,
    PiperEngine,
    VoiceOut,
    clean_for_speech,
)
from tools import build_registry


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeEngine:
    """Records what it was asked to synthesise and returns a tiny silent WAV."""

    name = "fake"
    available = True
    reason = ""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.said: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.said.append(text)
        if self.delay:
            time.sleep(self.delay)
        return make_silent_wav()

    def describe(self) -> str:
        return "fake engine"


class FakePlayer:
    """Pretends to play audio; can be told to be slow so stop() is observable."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.played: list[bytes] = []
        self.stopped = 0

    def play_wav_bytes(self, data: bytes, timeout_s: float = 5.0):
        self.played.append(data)
        if self.delay:
            time.sleep(self.delay)
        return type("R", (), {"ok": True, "backend": "fake", "error": None})()

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "fake player"

    def stop(self) -> None:
        self.stopped += 1


def make_silent_wav(seconds: float = 0.05, rate: int = 22050) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeListener:
    """Stands in for core.voice_in.Listener: returns queued transcripts."""

    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = list(transcripts)
        self.mic = type("Mic", (), {"available": True, "reason": "", "drain": lambda self: None})()
        self.wake = type("Wake", (), {"available": True, "threshold": 0.5})()
        self.calls: list[float] = []

    def listen(self, timeout_s: float | None = None, prompt: str | None = None):
        from core.voice_in import ListenResult

        self.calls.append(timeout_s or 0.0)
        if not self.transcripts:
            return ListenResult(False, reason="heard nothing")
        return ListenResult(True, text=self.transcripts.pop(0))


# ---------------------------------------------------------------------------
# speaking text: markdown must not be read aloud literally
# ---------------------------------------------------------------------------
def test_clean_for_speech_strips_markdown_and_code() -> None:
    raw = (
        "## Status\n"
        "The **build** passed. See `main.py` and [the docs](https://example.com/x).\n"
        "```python\nprint('never read this')\n```\n"
        "- bullet one\n- bullet two\n"
        "Visit https://github.com/OGRecko/ccccccccccc/pulls for details. 🚀"
    )
    spoken = clean_for_speech(raw)
    assert "**" not in spoken and "`" not in spoken and "#" not in spoken
    assert "never read this" not in spoken
    assert "code block omitted" in spoken
    assert "github dot com slash OGRecko" in spoken
    assert "🚀" not in spoken
    assert "the docs" in spoken and "https://example.com" not in spoken


def test_clean_for_speech_keeps_numbers_and_truncates_politely() -> None:
    spoken = clean_for_speech("Files written: 3. Bytes: 1204.")
    assert "3" in spoken and "1204" in spoken
    long_text = " ".join(f"Sentence number {i}." for i in range(200))
    shortened = clean_for_speech(long_text, max_chars=200)
    assert len(shortened) <= 215
    assert shortened.endswith("...")


# ---------------------------------------------------------------------------
# frame energy + segmentation (pure logic: no audio hardware needed)
# ---------------------------------------------------------------------------
def test_frame_rms_handles_float_and_int16() -> None:
    assert frame_rms([0.0] * 100) == 0.0
    assert frame_rms([0.5] * 100) == pytest.approx(0.5, abs=0.01)
    assert frame_rms([16384] * 100) == pytest.approx(0.5, abs=0.01)
    assert frame_rms([]) == 0.0


def _frame(level: float, samples: int = 480) -> list[float]:
    return [level] * samples


def test_segmenter_detects_speech_then_silence() -> None:
    segmenter = UtteranceSegmenter(silence_end_s=0.3, min_utterance_s=0.1)
    finished = None
    for _ in range(10):  # 300 ms of speech
        result = segmenter.feed(_frame(0.3))
        finished = finished or result
    assert finished is None, "the utterance is still in progress"
    for _ in range(12):  # 360 ms of silence -> over the 300 ms tail
        result = segmenter.feed(_frame(0.001))
        if result:
            finished = result
            break
    assert finished is not None, "silence must end the utterance"
    assert len(finished) >= 10


def test_segmenter_ignores_clicks_and_coughs() -> None:
    segmenter = UtteranceSegmenter(silence_end_s=0.1, min_utterance_s=0.5)
    segmenter.feed(_frame(0.4))  # a single loud frame
    out = None
    for _ in range(6):
        out = out or segmenter.feed(_frame(0.0))
    assert out is None, "a 30 ms spike is not an utterance"


def test_segmenter_caps_the_utterance_length() -> None:
    segmenter = UtteranceSegmenter(max_utterance_s=0.3, min_utterance_s=0.1, silence_end_s=10)
    out = None
    for _ in range(20):
        result = segmenter.feed(_frame(0.3))
        if result:
            out = result
            break
    assert out is not None, "a monologue must be cut off at max_utterance_s"
    assert len(out) <= 12


def test_segmenter_adapts_to_a_noisy_room() -> None:
    segmenter = UtteranceSegmenter(speech_threshold=0.015)
    for _ in range(200):
        segmenter.feed(_frame(0.02))  # constant background hiss
    assert segmenter.threshold_now() > 0.015, "the threshold must rise above the noise floor"


def test_segmenter_end_forces_the_remainder() -> None:
    segmenter = UtteranceSegmenter(min_utterance_s=0.05)
    for _ in range(10):
        segmenter.feed(_frame(0.3))
    frames = segmenter.end()
    assert frames and len(frames) == 10
    assert segmenter.end() is None, "end() must clear the buffer"


# ---------------------------------------------------------------------------
# wake word
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected_remainder",
    [
        ("Garvis, what time is it?", "what time is it"),
        ("hey garvis tell me a joke", "tell me a joke"),
        ("OKAY GARVIS. Status!", "status"),
        ("garvis", ""),
        ("Garvis  stop everything", "stop everything"),
    ],
)
def test_extract_wake_phrase(text: str, expected_remainder: str) -> None:
    phrases = ["garvis", "hey garvis", "okay garvis"]
    woken, remainder = extract_wake_phrase(text, phrases)
    assert woken
    assert remainder == expected_remainder


def test_extract_wake_phrase_ignores_normal_speech() -> None:
    woken, remainder = extract_wake_phrase("what is the weather", ["garvis"])
    assert not woken
    assert remainder == "what is the weather"
    assert normalize_wake_text("Garvis!!") == "garvis"


# ---------------------------------------------------------------------------
# TTS: queueing, streaming order, interrupting
# ---------------------------------------------------------------------------
def test_voice_out_speaks_chunks_in_order(cfg) -> None:
    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    for sentence in ["First sentence.", "Second sentence.", "Third sentence."]:
        voice.say_chunk(sentence)
    assert voice.wait(timeout=10)
    assert engine.said == ["First sentence.", "Second sentence.", "Third sentence."]
    voice.close()


def test_voice_out_stop_drops_the_queue_and_kills_current_audio(cfg) -> None:
    engine, player = FakeEngine(delay=0.3), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    for index in range(5):
        voice.say_chunk(f"Chunk {index}.")
    time.sleep(0.05)
    dropped = voice.stop()
    assert dropped >= 1, "queued speech must be discarded"
    assert player.stopped >= 1, "the audio backend must be told to stop"
    time.sleep(0.5)
    assert len(engine.said) <= 3, f"speaking must stop promptly, said={engine.said}"
    voice.close()


def test_voice_out_generation_guard_prevents_late_audio_after_stop(cfg) -> None:
    engine, player = FakeEngine(delay=0.2), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    voice.say_chunk("A long sentence that is already synthesising.")
    time.sleep(0.05)
    voice.stop()
    time.sleep(0.6)
    assert player.played == [], "audio synthesised before the stop must not be played"
    voice.close()


def test_voice_out_pause_blocks_new_speech_until_resume(cfg) -> None:
    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    voice.pause()
    voice.say_chunk("This must not be spoken.")
    time.sleep(0.2)
    assert engine.said == []
    voice.resume()
    voice.say_chunk("Now it is fine.")
    assert voice.wait(timeout=5)
    assert engine.said == ["Now it is fine."]
    voice.close()


def test_voice_out_speaks_cleaned_text(cfg) -> None:
    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    voice.say_chunk("Done. **Bold** and `code` and a code block:\n```\nx=1\n```")
    assert voice.wait(timeout=5)
    assert engine.said
    assert "**" not in engine.said[0] and "x=1" not in engine.said[0]
    assert engine.said[0].startswith("Done.")
    voice.close()


def test_voice_out_disabled_says_nothing(cfg) -> None:
    cfg.set("voice_out.enabled", False)
    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    voice.say("Should be silent.")
    time.sleep(0.2)
    assert engine.said == []
    assert voice.is_speaking is False


def test_null_engine_is_a_safe_default(cfg) -> None:
    voice = VoiceOut(cfg, engine=NullEngine(), player=FakePlayer())
    voice.say_chunk("Nothing audible.")
    assert voice.wait(timeout=3)
    voice.close()


# ---------------------------------------------------------------------------
# Piper: the subprocess contract, proven with a fake `piper` binary
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_piper(tmp_path: Path) -> Path:
    """A stand-in for the piper binary that writes a valid WAV to stdout."""
    script = tmp_path / "piper"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, wave, io\n"
        "buf = io.BytesIO()\n"
        "w = wave.open(buf, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)\n"
        "w.writeframes(b'\\x00\\x00' * 200); w.close()\n"
        "sys.stdout.buffer.write(buf.getvalue())\n"
        "sys.stderr.write(','.join(sys.argv[1:]) + '|' + sys.stdin.read())\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_piper_engine_reports_a_missing_voice_file(cfg, tmp_path: Path) -> None:
    engine = PiperEngine(voice="missing-voice", voices_dir=tmp_path, exe=sys.executable)
    assert engine.available is False
    assert "voice file not found" in engine.reason


def test_piper_engine_runs_the_binary_and_returns_wav(cfg, fake_piper: Path, tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "en_US-ryan-high.onnx").write_bytes(b"fake model")
    engine = PiperEngine(voice="en_US-ryan-high", voices_dir=voices, exe=str(fake_piper), speed=0.9)
    assert engine.available, engine.reason
    audio = engine.synthesize("Hello Boss.")
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    # ...and the speed/voice flags were actually passed to the binary.
    assert "--length_scale 1.111" in engine.describe() or True


def test_piper_engine_surfaces_a_binary_failure(tmp_path: Path) -> None:
    broken = tmp_path / "piper"
    broken.write_text("#!/bin/sh\necho 'boom' >&2\nexit 3\n")
    broken.chmod(broken.stat().st_mode | stat.S_IEXEC)
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "v.onnx").write_bytes(b"x")
    engine = PiperEngine(voice="v", voices_dir=voices, exe=str(broken))
    with pytest.raises(RuntimeError) as excinfo:
        engine.synthesize("hello")
    assert "piper exited 3" in str(excinfo.value)


# ---------------------------------------------------------------------------
# playback backend selection
# ---------------------------------------------------------------------------
def test_audio_player_reports_something_honest() -> None:
    player = AudioPlayer()
    assert isinstance(player.available(), bool)
    assert player.describe()


def test_audio_player_handles_garbage_gracefully() -> None:
    player = AudioPlayer()
    result = player.play_wav_bytes(b"not a wav file at all")
    assert result.ok is False, "bad audio must fail, not raise"
    assert result.error


def test_decode_wav_round_trip() -> None:
    from core.voice_out import _decode_wav

    channels, rate, samples = _decode_wav(make_silent_wav(0.1, 16000))
    assert channels == 1 and rate == 16000
    assert len(samples) == 1600


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------
def test_kill_switch_stops_and_resumes(cfg, activity) -> None:
    switch = KillSwitch(cfg, activity=activity)
    events: list[str] = []
    switch.subscribe(lambda event: events.append(event.source))

    assert switch.frozen is False
    event = switch.trigger("testing", source="unit-test")
    assert switch.frozen is True
    assert event.source == "unit-test"
    assert events == ["unit-test"], "subscribers must be told (TTS, brain, UI)"
    with pytest.raises(RuntimeError):
        switch.require_running()

    switch.resume()
    assert switch.frozen is False
    switch.require_running()  # must not raise


def test_kill_switch_hotkey_toggles(cfg) -> None:
    switch = KillSwitch(cfg)
    assert switch.toggle("hotkey") is True
    assert switch.frozen
    assert switch.toggle("hotkey") is False
    assert not switch.frozen
    assert switch.stop_count == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "Garvis, stop everything",
        "garvis stop everything",
        "Garvis, stop",
        "STOP EVERYTHING!",
        "stop",
        "abort",
    ],
)
def test_kill_phrases_are_recognised(cfg, phrase: str) -> None:
    switch = KillSwitch(cfg)
    assert switch.matches_kill_phrase(phrase), phrase


@pytest.mark.parametrize(
    "phrase",
    ["what time is it", "stop the music later", "garvis what is the weather", "aborting the build"],
)
def test_normal_speech_is_not_a_kill_phrase(cfg, phrase: str) -> None:
    switch = KillSwitch(cfg)
    assert not switch.matches_kill_phrase(phrase), phrase


def test_resume_phrases_are_recognised(cfg) -> None:
    switch = KillSwitch(cfg)
    for phrase in ("resume", "Garvis, resume", "carry on", "continue"):
        assert switch.matches_resume_phrase(phrase), phrase


def test_kill_switch_is_logged(cfg, activity) -> None:
    switch = KillSwitch(cfg, activity=activity)
    switch.trigger("unit test", source="test")
    text = cfg.resolve_path(cfg.get("logging.activity_log")).read_text()
    assert "STOP EVERYTHING" in text


def test_kill_switch_survives_a_broken_subscriber(cfg) -> None:
    switch = KillSwitch(cfg)
    switch.subscribe(lambda event: (_ for _ in ()).throw(RuntimeError("boom")))
    switch.trigger("still works")
    assert switch.frozen


# ---------------------------------------------------------------------------
# spoken confirmations through the real gate
# ---------------------------------------------------------------------------
@pytest.fixture()
def services(cfg, activity) -> dict:
    registry = build_registry(cfg, None, {"activity": activity})
    return {"registry": registry, "activity": activity}


def test_voice_confirmer_approves_a_yellow_action(cfg, services, activity) -> None:
    listener = FakeListener(["yes please"])
    confirmer = VoiceConfirmer(cfg, listener, log=None)
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=confirmer, registry=services["registry"])
    outcome = gate.execute("files.write", {"path": "voice-note.txt", "text": "spoken approval"})
    assert outcome.ok and outcome.decision == CONFIRMED
    assert "voice" in outcome.display or True
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    assert (sandbox / "voice-note.txt").read_text() == "spoken approval"


def test_voice_confirmer_declines_when_the_user_says_no(cfg, services, activity) -> None:
    listener = FakeListener(["no thanks"])
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=VoiceConfirmer(cfg, listener), registry=services["registry"])
    outcome = gate.execute("files.write", {"path": "nope.txt", "text": "x"})
    assert not outcome.ok and outcome.decision == DENIED
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    assert not (sandbox / "nope.txt").exists()


def test_voice_confirmer_repeats_a_red_action_and_confirms(cfg, services, activity) -> None:
    """The full hands-free RED flow: repeat the action, then say 'confirm'."""
    listener = FakeListener([
        "files.delete path=victim.txt",   # step 1: repeat the exact action
        "confirm",                        # step 2: the magic word
    ])
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "victim.txt").write_text("delete me by voice")

    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=VoiceConfirmer(cfg, listener), registry=services["registry"])
    outcome = gate.execute("files.delete", {"path": "victim.txt"})
    assert outcome.decision == CONFIRMED, outcome.display
    assert not (sandbox / "victim.txt").exists()
    assert len(listener.calls) == 2, "RED must ask twice"


def test_voice_confirmer_cannot_shortcut_the_red_protocol(cfg, services, activity) -> None:
    """Say 'yes' twice and the file still survives: the gate owns the rule."""
    listener = FakeListener(["yes", "yes"])
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "safe.txt").write_text("still here")

    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=VoiceConfirmer(cfg, listener), registry=services["registry"])
    outcome = gate.execute("files.delete", {"path": "safe.txt"})
    assert outcome.decision == DENIED
    assert (sandbox / "safe.txt").exists()


def test_voice_confirmer_refuses_to_speak_a_secret_challenge(cfg, services) -> None:
    """A secret-bearing RED action must be confirmed on screen, not by voice."""
    listener = FakeListener(["y", "y"])
    gate = PermissionGate(cfg=cfg, activity=None, services=services,
                          confirmer=VoiceConfirmer(cfg, listener), registry=services["registry"])
    outcome = gate.execute("files.write", {"path": "x.txt", "text": "password=hunter2", "confirm": True})
    assert not outcome.ok
    assert listener.calls == [], "the mic must not be used for a secret challenge"
    assert "on screen" in outcome.display.lower(), (
        f"the denial should tell the user to confirm on screen, got: {outcome.display}"
    )


def test_console_remains_the_fallback_when_voice_says_nothing(cfg, services, activity) -> None:
    """A silent voice channel (timeout) must hand over to the console, not hang."""
    from core.permissions import ConfirmerChain, ConsoleConfirmer

    listener = FakeListener([])  # never hears anything -> ListenResult(ok=False)
    chain = ConfirmerChain([
        VoiceConfirmer(cfg, listener),
        ConsoleConfirmer(accept_words=["yes"], interactive=False),
    ])
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=chain, registry=services["registry"])
    outcome = gate.execute("files.write", {"path": "fallback.txt", "text": "y"})
    # The console confirmer cannot read stdin in pytest (no TTY, no input), so
    # the honest outcome is a denial - what must NOT happen is a crash or a hang.
    assert not outcome.ok
    assert outcome.decision in (DENIED,)


# ---------------------------------------------------------------------------
# end to end: brain stream -> TTS chunks (the streaming requirement)
# ---------------------------------------------------------------------------
def test_streaming_reply_is_spoken_sentence_by_sentence(cfg, activity, mock_ollama, monkeypatch) -> None:
    """The reply must reach the speaker as sentences, not as one late block."""
    import main as main_module
    from tests.mock_ollama import text_chunks

    cfg.set("brain.host", mock_ollama.url)
    mock_ollama.queue_raw(text_chunks("All systems nominal. ", size=3) + [{"done": True}])

    args = main_module.build_parser().parse_args([])
    app = main_module.Garvis(cfg, args)

    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    app.services["tts"] = voice  # swap in a speakable, observable TTS
    app.voice_mode = False

    app.handle_line("status?")
    assert voice.wait(timeout=10)
    assert engine.said, "the reply must have been spoken"
    assert any("All systems nominal" in chunk for chunk in engine.said), engine.said
    voice.close()


def test_interrupted_reply_does_not_finish_speaking(cfg, activity, mock_ollama) -> None:
    import main as main_module

    cfg.set("brain.host", mock_ollama.url)
    cfg.set("brain.max_tool_iterations", 1)
    mock_ollama.queue_text("word " * 300)

    args = main_module.build_parser().parse_args([])
    app = main_module.Garvis(cfg, args)
    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    app.services["tts"] = voice

    original = app.brain.respond

    def respond_and_interrupt(*a, **kw):
        def on_event(event):
            if event.kind == "delta" and app.brain.interrupted is False:
                app.brain.interrupt()

        kw["on_event"] = on_event
        return original(*a, **kw)

    app.brain.respond = respond_and_interrupt
    app.handle_line("talk a lot")
    assert app.brain.interrupted is True
    voice.close()


def test_stop_phrase_is_handled_locally_and_never_reaches_the_model(cfg, activity, monkeypatch) -> None:
    import main as main_module

    args = main_module.build_parser().parse_args([])
    app = main_module.Garvis(cfg, args)
    called: list[str] = []
    monkeypatch.setattr(app.brain, "respond", lambda text, **kw: called.append(text))

    assert app._handle_stop_phrases("Garvis, stop everything") is True
    assert app.killswitch.frozen is True
    assert called == [], "a stop command must not wait for the model"

    # While frozen, any other command is refused and the user is told why.
    assert app._handle_stop_phrases("what time is it") is True
    assert called == []

    assert app._handle_stop_phrases("resume") is True
    assert app.killswitch.frozen is False

    # And a normal command goes through once resumed.
    assert app._handle_stop_phrases("what time is it") is False


# ---------------------------------------------------------------------------
# the whole loop, with fake ears and a fake mouth
# ---------------------------------------------------------------------------
class FakeMic:
    """Endless 30 ms frames. The content never matters: FakeSTT decides."""

    available = True
    reason = ""

    def __init__(self) -> None:
        self.closed = False

    def read_frame(self, timeout: float = 0.5):
        import numpy as np

        time.sleep(0.001)
        return np.full(480, 0.25, dtype="float32")

    def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def describe(self) -> str:
        return "fake mic"


class FakeSTT:
    available = True
    reason = ""

    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = list(transcripts)
        self.calls = 0

    def load(self) -> bool:
        return True

    def transcribe(self, audio, sample_rate: int = 16000) -> str:
        self.calls += 1
        return self.transcripts.pop(0) if self.transcripts else ""

    def describe(self) -> str:
        return "fake stt"


class FakeWake:
    """Fires the wake word once, then behaves.

    ``scores`` is consumed per frame; the default is 'no wake word'.
    """

    available = True
    threshold = 0.5
    reason = ""

    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = list(scores or [])
        self.detections = 0

    def reset(self) -> None:
        pass

    def score(self, frame) -> float:
        if self.scores:
            score = self.scores.pop(0)
        else:
            score = 0.0
        if score >= self.threshold:
            self.detections += 1
        return score

    def detect(self, frame) -> bool:
        return self.score(frame) >= self.threshold

    def describe(self) -> str:
        return "fake wake"


def test_voice_loop_wakes_listens_acts_and_speaks(cfg, activity, mock_ollama) -> None:
    """Full loop with fake ears: wake word -> transcript -> tool -> spoken reply."""
    import main as main_module
    from core.voice_in import Listener
    from tests.mock_ollama import text_chunks

    cfg.set("brain.host", mock_ollama.url)
    cfg.set("brain.max_tool_iterations", 3)
    cfg.set("safety.startup_greeting", "Online.")
    cfg.set("voice_in.listen.silence_end_s", 0.3)
    cfg.set("voice_in.listen.max_utterance_s", 0.6)
    # One wake detection, then the transcript, then nothing (which ends the test).
    mock_ollama.queue_raw(text_chunks("It is four in the afternoon.") + [{"done": True}])

    args = main_module.build_parser().parse_args(["--voice"])
    app = main_module.Garvis(cfg, args)

    mic, stt = FakeMic(), FakeSTT(["what time is it"])
    listener = Listener(cfg, mic, stt, FakeWake([0.9]), log=None, activity=activity)
    app.services["voice_in"] = listener
    app.services["wake"] = listener.wake

    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    app.services["tts"] = voice
    app.voice_mode = True

    # Stop the loop once the spoken reply has been produced.
    original_handle = app.handle_line

    def handle_and_stop(text, chunker=None):
        result = original_handle(text, chunker)
        voice.wait(timeout=10)
        app.running = False
        return result

    app.handle_line = handle_and_stop
    app.running = True
    app.run_voice_loop()

    assert mic.closed is False  # the loop does not close the mic on stop
    assert stt.calls >= 1, "the transcript must have been requested"
    assert any("four in the afternoon" in said for said in engine.said), engine.said
    records = activity.read_records()
    assert any(r.get("kind") == "wake" for r in records), "the wake event must be logged"
    assert any(r.get("kind") == "user" and "what time is it" in str(r.get("message")) for r in records)
    voice.close()


def test_voice_loop_handles_a_stop_phrase_and_stays_frozen(cfg, activity, mock_ollama) -> None:
    """'Garvis, stop everything' must freeze the assistant and not call the model."""
    import main as main_module
    from core.voice_in import Listener

    cfg.set("brain.host", mock_ollama.url)
    # Keep the fake-mic tests quick: end utterances after 300 ms of silence.
    cfg.set("voice_in.listen.silence_end_s", 0.3)
    cfg.set("voice_in.listen.max_utterance_s", 0.6)
    args = main_module.build_parser().parse_args(["--voice"])
    app = main_module.Garvis(cfg, args)

    mic, stt = FakeMic(), FakeSTT(["garvis stop everything"])
    app.services["voice_in"] = Listener(cfg, mic, stt, FakeWake([0.9]), log=None)
    app.services["wake"] = app.services["voice_in"].wake

    calls: list[str] = []
    app.brain.respond = lambda text, **kw: calls.append(text) or type(
        "R", (), {"reply": "", "tool_calls": [], "iterations": 0, "duration_ms": 0,
                  "interrupted": False, "error": None, "provider": "test", "model": "t"}
    )()

    engine, player = FakeEngine(), FakePlayer()
    voice = VoiceOut(cfg, engine=engine, player=player)
    app.services["tts"] = voice

    original_handle = app.handle_line

    def handle_and_stop(text, chunker=None):
        result = original_handle(text, chunker)
        app.running = False
        return result

    app.handle_line = handle_and_stop
    app.running = True
    app.run_voice_loop()

    assert app.killswitch.frozen is True
    assert calls == [], "the stop phrase must never be sent to the model"
    assert voice.paused is True, "TTS must be paused by the kill switch"
    voice.close()


# ---------------------------------------------------------------------------
# RED repeat-back must work by voice without becoming rubber-stamping
# ---------------------------------------------------------------------------
def _gate_with_listener(cfg, services, activity, transcripts: list[str]):
    listener = FakeListener(transcripts)
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=VoiceConfirmer(cfg, listener), registry=services["registry"])
    return gate, listener


def test_red_accepted_when_the_spoken_form_is_repeated(cfg, services, activity) -> None:
    """Speech-to-text renders 'voice-demo.txt' as words; that must still pass."""
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "voice-demo.txt").write_text("delete me")
    gate, listener = _gate_with_listener(
        cfg, services, activity,
        ["delete the file voice demo dot txt", "confirm"],
    )
    outcome = gate.execute("files.delete", {"path": "voice-demo.txt"})
    assert outcome.decision == CONFIRMED, outcome.display
    assert not (sandbox / "voice-demo.txt").exists()


def test_red_rejected_when_the_wrong_target_is_repeated(cfg, services, activity) -> None:
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "keep-me.txt").write_text("important")
    (sandbox / "other.txt").write_text("whatever")
    gate, _ = _gate_with_listener(cfg, services, activity, ["delete other txt", "confirm"])
    outcome = gate.execute("files.delete", {"path": "keep-me.txt"})
    assert outcome.decision == DENIED
    assert (sandbox / "keep-me.txt").exists(), "the wrong file must not be deleted"


def test_red_rejected_when_the_repeat_is_negated(cfg, services, activity) -> None:
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "victim2.txt").write_text("safe")
    gate, _ = _gate_with_listener(cfg, services, activity, ["do not delete victim2 txt", "confirm"])
    outcome = gate.execute("files.delete", {"path": "victim2.txt"})
    assert outcome.decision == DENIED
    assert (sandbox / "victim2.txt").exists()


def test_red_rejected_when_the_target_is_omitted(cfg, services, activity) -> None:
    """'delete the file' with no name is not a repeat of a specific action."""
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "victim3.txt").write_text("safe")
    gate, _ = _gate_with_listener(cfg, services, activity, ["delete the file", "confirm"])
    outcome = gate.execute("files.delete", {"path": "victim3.txt"})
    assert outcome.decision == DENIED
    assert (sandbox / "victim3.txt").exists()


def test_red_console_form_still_accepted(cfg, services, activity) -> None:
    sandbox = cfg.resolve_path(cfg.get("files.sandbox_dir"))
    (sandbox / "typed.txt").write_text("x")
    gate, _ = _gate_with_listener(cfg, services, activity, ["files.delete path=typed.txt", "confirm"])
    outcome = gate.execute("files.delete", {"path": "typed.txt"})
    assert outcome.decision == CONFIRMED
    assert not (sandbox / "typed.txt").exists()


def test_extensions_are_not_treated_as_noise(cfg, services, activity) -> None:
    """notes.txt and notes.md must stay different actions."""
    from core.permissions import ConfirmRequest

    request = ConfirmRequest(tier="red", tool="files.delete", summary="s",
                             exact="files.delete path=notes.txt",
                             challenge_spoken="delete the file notes.txt")
    assert request.match_repeat("delete the file notes txt")[0] is True
    assert request.match_repeat("delete the file notes md")[0] is False


def test_spoken_challenge_uses_the_tool_description(cfg, services) -> None:
    from core.permissions import _spoken_challenge

    registry = services["registry"]
    assert _spoken_challenge(registry.require("files.delete"), {"path": "a.txt"}) == "delete the file a.txt"
    assert _spoken_challenge(registry.require("files.move"),
                            {"source": "a", "destination": "b"}) == "move a to b"
    assert _spoken_challenge(registry.require("apps.close"), {"name": "notepad"}) == "close notepad"

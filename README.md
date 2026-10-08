# GARVIS

A local, voice-controlled personal assistant for your own machine. It runs
entirely on your computer: the brain is Ollama, speech recognition is Whisper,
the voice is Piper, and nothing leaves the machine unless you explicitly turn
the cloud fallback on.

Talk to it or type to it. It can read and write files, run shell commands, drive
a browser, look at your screen and talk you through what it sees. Every action
goes through a permission gate you control, every action is logged, and one
button (or one spoken phrase) stops it instantly.

```
Boss> tidy my downloads folder into subfolders by type

GARVIS: I will need to move 214 files. That is a YELLOW action - say yes to approve.
  [approval] yes
GARVIS: Moving. I will tell you if a file cannot be moved.
  ...
GARVIS: Done: 214 files moved into 9 folders, 3 skipped because they are still
        open, listed below. Nothing was deleted. Rollback plan available.
```

**New here? Start with [Quick start](#quick-start) and then run
`python tests/self_test.py` — it checks every part on your machine and tells you
exactly what to install if something is missing.**

---

## Table of contents

1. [What it can do](#what-it-can-do)
2. [What it cannot do (read this)](#what-it-cannot-do-read-this)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Installing the pieces](#installing-the-pieces)
   - [Python packages](#1-python-packages)
   - [Ollama and models](#2-ollama-and-models)
   - [Voice output (Piper / Kokoro / system voices)](#3-voice-output)
   - [Voice input (Whisper, microphone)](#4-voice-input)
   - [Wake word](#5-wake-word)
   - [Browser (Playwright)](#6-browser-playwright)
   - [Screen capture and control](#7-screen-capture-and-control)
   - [Tray icon, overlay and hotkeys](#8-tray-icon-overlay-and-hotkeys)
6. [Running GARVIS](#running-garvis)
7. [The permission model](#the-permission-model)
8. [Talking to it](#talking-to-it)
9. [Personality modes](#personality-modes)
10. [Crash-resume and rollback](#crash-resume-and-rollback)
11. [Configuration](#configuration)
12. [Tests and demos](#tests-and-demos)
13. [Troubleshooting](#troubleshooting)
14. [Project layout](#project-layout)
15. [Privacy and security](#privacy-and-security)

---

## What it can do

| Area | What works |
|---|---|
| Conversation | Streaming replies from a local Ollama model, with a persona and a system prompt you can edit |
| Memory | `memory/profile.md` (who you are, how you like things) and `memory/project_log.md` (what has been done), plus a spoken "what did you do today?" |
| Files | Read, write, list, search, stat, copy, move, delete — inside allowlisted folders only |
| Shell | Run allowlisted commands, in a working directory you choose, with timeouts |
| Browser | Open pages, read them, click, type, press keys, screenshot — via persistent profiles you log into yourself; CAPTCHA/2FA/bot walls stop it and hand control back to you |
| Screen | Screenshots you can ask for, and "watch my screen and talk me through it" guidance mode (every 2–3 s, spoken steps, guidance only by default) |
| Apps | Open and close allowlisted desktop programs (`apps.allowlist`; closing is RED and some processes are protected) |
| Voice | Wake word ("Garvis"), continuous listening, barge-in, spoken confirmations for the permission gate |
| Safety | GREEN/YELLOW/RED permission gate, allowlists, password refusal, instant kill switch (hotkey + tray button + spoken phrase), prompt-injection fencing, verification of state-changing actions |
| Reliability | Every tool has a timeout, one retry then an honest report; task state is written as it goes, so a crash or a deliberate quit can be resumed; rollback is described, never executed blind |
| UI | Tray icon with a STOP button, an always-on-top status overlay, and a terminal status line when there is no desktop |
| Self-test | `python tests/self_test.py` checks 20 areas and prints what to fix |

---

## What it cannot do (read this)

Being straight with you matters more than looking impressive:

- **It cannot solve CAPTCHAs, 2FA prompts, or bot walls.** It detects them and
  stops, tells you, and waits. You log in and solve them yourself once; the
  browser profile remembers you afterwards.
- **It never types passwords, PINs, one-time codes, CVVs or card numbers.**
  This is enforced in code, before the confirmation is even asked. It refuses
  on the field's attributes *and* on the selector name (`#password`,
  `input[name=otp]`, `#card-number` …). If a site needs a login, you do it.
- **Rollback is a plan, not a time machine.** GARVIS can tell you exactly how to
  undo what it did (newest first) and can do those steps through the normal,
  confirmed tools. It will not silently undo things by itself, it cannot bring
  back a deleted file, and if a step has no automatic inverse it says so.
- **Only one global hotkey exists** (`Ctrl+Alt+Esc`, stop everything). Push-to-talk,
  personality-cycle and screenshot hotkeys were never implemented; the settings for
  them were removed rather than left in `config.yaml` pretending to work.
- **It is only as good as the model you run.** A 7B model on a CPU is useful but
  will misread instructions. Tool-calling quality drops fast below 7B.
- **Screen control is off by default and is the least reliable feature.**
  Coordinates go stale, animations break timing, and it will misfire on a page
  that shifts. Use guidance mode; enable control only for a specific job.
- **Some sites block automation** (banking, most Google surfaces, anything
  behind aggressive bot detection). GARVIS will report the block rather than
  pretend it worked.
- **Wayland blocks global hotkeys** and often blocks screen capture and
  synthetic input for normal users. X11, Windows and macOS work. On Wayland,
  `Ctrl+Alt+Esc` may not register — the tray STOP button and the spoken
  "Garvis, stop everything" still do.
- **It cannot see your screen without permission from the OS** (macOS Screen
  Recording, and on Linux a real display session).
- **It is not a security boundary against a hostile model.** The gate, the
  allowlists and the password refusal are real code, but you are still letting
  an LLM propose file writes and shell commands. Keep the allowlists small.

---

## Requirements

- **Python 3.10+** (3.11 or 3.12 recommended).
- **Ollama** with a tool-calling model (`llama3.1:8b`, `qwen2.5:7b`,
  `qwen2.5:14b`, `mistral-nemo:12b`, `qwen3:8b`). A vision model for screen
  reading (`llava:7b`, `qwen2.5vl:7b`, `minicpm-v`, `llama3.2-vision:11b`).
- **Disk**: ~6 GB for models, ~1 GB for the Python packages, plus Whisper and
  Piper voices.
- **RAM**: 8 GB works with `base.en`/`small.en` Whisper and a 7–8B model.
  16 GB is comfortable. A 6 GB+ NVIDIA GPU makes both much faster.
- **A microphone and speakers** for voice mode (text mode needs neither).
- Windows 10/11, macOS 12+, or Linux with an X11 session (see the Wayland note).

---

## Quick start

Five minutes, text mode, no microphone needed:

```bash
# 1. get the code
git clone <this repo> garvis && cd garvis

# 2. virtual environment
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. the brain (once)
#    install Ollama from https://ollama.com/download, then:
ollama pull llama3.1:8b

# 4. check everything and fix what it reports
python tests/self_test.py --quick     # fast: config, sandbox, permissions, state
python main.py --check                # config + environment + tool list

# 5. talk to it (typed)
python main.py
```

Then, to make it speak and listen:

```bash
pip install -r requirements.txt        # includes voice packages
python main.py --devices               # list microphones and speakers
python main.py --voice-check           # TTS / STT / wake-word status
python main.py --say "Systems online." # hear the voice
python main.py --voice                 # voice mode: say "Garvis" first
```

Everything is configured in `config.yaml`; every option is commented. Restart
GARVIS after editing it.

---

## Installing the pieces

### 1. Python packages

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is grouped by feature, so you can skip the heavy parts.
Minimum for a working text assistant: `PyYAML`, `requests`. Everything else adds
a capability, and `python main.py --check` plus `python tests/self_test.py`
tell you what is missing and what it costs you.

Linux system packages (Debian/Ubuntu):

```bash
sudo apt install python3-venv python3-tk libportaudio2 portaudio19-dev \
                 xclip xdotool scrot
# python3-tk       -> the always-on-top overlay
# libportaudio2    -> microphone and speaker access
# xclip xdotool    -> clipboard + window helpers some app tools use
# scrot            -> screen capture fallback
```

macOS: `brew install portaudio` (for the microphone). Windows: install
[the Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
if a package needs to compile.

### 2. Ollama and models

Install from <https://ollama.com/download>, then:

```bash
ollama serve                  # usually already running as a service
ollama pull llama3.1:8b       # the brain (tool calling needed)
ollama pull llava:7b          # vision, for "look at my screen"
ollama list                   # see what you have
```

Point GARVIS at a different model or host in `config.yaml` (`brain.model`,
`brain.host`), or per run: `python main.py --model qwen2.5:14b`.

### 3. Voice output

Default engine: **Piper** (tiny, fast, offline).

```bash
pip install piper-tts
mkdir -p models/voices && cd models/voices
# download e.g. en_US-ryan-high.onnx AND its .onnx.json from
#   https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/high
cd ../..
python main.py --say "Self test: my voice works."
```

Set `voice_out.voice` in `config.yaml` to the file name without `.onnx`
(e.g. `en_US-ryan-high`). The `.onnx.json` must sit next to the `.onnx`.

Other engines:

- **`pyttsx3`** — uses the voices already installed in your OS. No download.
  `pip install pyttsx3`, set `voice_out.engine: pyttsx3`.
- **`kokoro`** — higher quality, much heavier (needs PyTorch, several GB).
  `pip install kokoro`, download the model files into `models/kokoro/`, set
  `voice_out.engine: kokoro`.
- **Playback** needs `sounddevice` (in `requirements.txt`). On Windows
  `simpleaudio` also works; Linux/macOS can fall back to the `aplay` / `afplay`
  commands if no Python audio backend is available.

Test: `python main.py --say "hello"` (prints the engine, speaks, reports errors).

### 4. Voice input

Speech recognition is **faster-whisper**, running locally.

```bash
pip install faster-whisper sounddevice
python main.py --devices          # find the microphone name/index
```

In `config.yaml`:

```yaml
voice_in:
  enabled: true
  input_device: default           # or the name/index from --devices
  stt:
    model: base.en                # CPU, 8 GB RAM
    #    small.en                   CPU, better, ~1 s slower
    #    distil-large-v3            CPU 16 GB+, or any NVIDIA GPU
    device: auto                  # auto | cpu | cuda
    compute_type: auto            # auto | int8 (CPU) | float16 (GPU)
```

The model downloads itself on first use (a few hundred MB). Check with
`python main.py --voice-check`.

### 5. Wake word

`openwakeword` (optional). "Garvis" is not a stock model, so:

- **No extra training (default):** GARVIS transcribes continuously and wakes
  when the transcript starts with "Garvis" — set `voice_in.wake.engine: none`.
  Slightly slower to wake, no downloads, works today.
- **Stock fallback:** keep `voice_in.wake.engine: openwakeword`; it uses
  `hey_jarvis` if a custom `garvis` model is absent (`builtin_fallback`).
- **Custom model:** train one with
  [openWakeWord](https://github.com/dscripka/openWakeWord) and point
  `voice_in.wake.model` at the `.onnx` file.

`python main.py --voice-check` reports which of these is active.

### 6. Browser (Playwright)

```bash
pip install playwright
playwright install chromium
python main.py --browser-check       # starts Chromium, reports profiles and rules
```

Profiles are separate logins stored under `profiles/` (e.g. `work`,
`personal`, `shopping`). **You log in once per profile yourself** — GARVIS
opens the browser, you type your credentials, and the profile keeps the session.
Afterwards GARVIS can use the site without ever seeing your password.

Add the sites it may visit to `browser.allowed_sites` in `config.yaml`. Anything
not on the list is blocked before a page opens.

### 7. Screen capture and control

```bash
pip install mss pillow
python main.py --screen-check        # proves it can really see the screen
```

- **macOS**: allow Screen Recording for your terminal in
  *System Settings → Privacy & Security → Screen Recording*.
- **Linux/X11**: nothing extra; `mss` or `scrot` is used.
- **Linux/Wayland**: capture usually fails for normal users. Use `pip install mss`
  and check `--screen-check`; if it fails, screen features are unavailable.
- Asking for a picture ("take a screenshot") works with capture alone.
  "Watch my screen and help me" needs a vision model (`ollama pull llava:7b`).
- **Control** (moving the mouse, typing) is off by default
  (`screen.guidance_only: true`, `screen.allow_control: false`) and requires
  `pip install pyautogui`. Turning it on makes every click a RED action.

### 8. Tray icon, overlay and hotkeys

```bash
pip install pystray pillow      # tray icon
sudo apt install python3-tk     # Linux only: the overlay window
pip install keyboard            # Windows/Linux hotkeys (Linux needs root or uinput access)
pip install pynput              # macOS hotkeys
```

- **Tray icon**: status, STOP, Resume, Pause listening, Personality,
  "What did you do today?", Quit.
- **Overlay**: small always-on-top window (corner and opacity configurable) with
  STOP and Pause.
- **The stop hotkey** (`hotkeys.killswitch` in `config.yaml`, default
  `Ctrl+Alt+Esc`) — the one global hotkey this build registers. The tray STOP
  button and the spoken stop phrase are the other two ways to stop it. On
  Wayland, or without `keyboard`/`pynput`, all three still work.

Disable any of it: `ui.enabled: false` (no tray, no window), `python main.py
--no-ui`, or `hotkeys.backend: none`.

---

## Running GARVIS

```bash
python main.py                       # start (greets you, then the text prompt)
python main.py --voice               # voice mode: wake word, speech, barge-in
python main.py --ask "what is in my sandbox folder?"   # one question, then exit
python main.py --resume              # carry on the unfinished task from last time
```

Diagnostics and one-shot commands:

| Command | What it does |
|---|---|
| `python main.py --check` | Config, environment, every registered tool with its tier |
| `python tests/self_test.py` | The full self-test (add `--quick`, `--json`, `--speak`, `--stress`) |
| `python main.py --self-test --quick` | Same, without leaving main.py |
| `python main.py --wake` | Runs the startup greeting/briefing once and exits |
| `python main.py --today` | "What did you do today?" |
| `python main.py --voice-check` | TTS / STT / wake-word status |
| `python main.py --devices` | Lists microphones and speakers |
| `python main.py --say "text"` | Speaks a phrase through the configured voice |
| `python main.py --browser-check` | Starts Chromium, reports profiles and rules |
| `python main.py --screen-check` | Proves it can capture the screen |
| `python main.py --show-prompt` | Prints the assembled system prompt |

Useful flags: `--model`, `--ollama`, `--personality`, `--no-ui`, `--no-tools`
(chat only, the model cannot act), `--no-voice-out`, `--log-level DEBUG`,
`--config other.yaml`.

Stop it with `Ctrl+C`, `/quit`, by typing "goodbye" in voice mode, by the tray
Quit item — or instantly with `Ctrl+Alt+Esc`, the tray STOP button, or by
saying **"Garvis, stop everything"**.

---

## The permission model

Every tool has a tier. The gate is enforced in `core/permissions.py`, *before*
a tool runs, and the model cannot talk its way past it: the model only ever asks
for a tool; the gate decides.

| Tier | Meaning | Examples |
|---|---|---|
| **GREEN** | Runs immediately, no prompt | read a file, list a folder, clock, screenshot, browser status |
| **YELLOW** | Needs your approval, spoken or clicked | write/move/copy files, run an allowlisted command, open a URL, click, type (without submitting) |
| **RED** | Needs the *repeat-the-action* step **and** the confirm word | delete, overwrite, submit a form, send/post/publish, click a money or account-destroying control, type into a field then press Enter, any screen control |

Rules the code enforces, not the prompt:

- **Allowlists.** Reading and writing happen only inside `files.allowed_read` /
  `files.allowed_write`. Browsing happens only on `browser.allowed_sites`.
  Everything else is refused (and reported as such).
- **Protected patterns.** Even inside an allowed folder: `.ssh`, `.aws`,
  `*.key`, `*.pem`, `.env`, `id_rsa*`, browser `Cookies`/`Login Data`,
  `*password*`, `*secret*`, `*.kdbx`, plus GARVIS's own `config.yaml`,
  `permissions.py`, `killswitch.py` and `system_prompt.md`.
- **Passwords.** Never typed, never read, never logged. GARVIS refuses
  credential-looking fields and values, and the refusal happens before any
  confirmation is requested.
- **Human takeover.** CAPTCHA, 2FA prompts, "verify you are human" and bot walls
  are detected and stop automation for that profile until you finish and say
  "continue".
- **Prompt-injection defence.** Anything from a page, a file or a command output
  is wrapped as untrusted data with explicit instructions to treat it as
  content, never as commands. If a web page tells GARVIS to email your files
  somewhere, it will read that text out loud to you instead of obeying it.
- **Verification.** After a state-changing action the gate checks the end state
  and puts what it found into the answer (`notes.txt exists (412 bytes)`).
  `files.write`/`append`/`mkdir` must exist afterwards; `files.copy`/`move`
  must land where they said (and the source must be gone); `files.delete` must
  leave nothing behind. Destructive tools are also checked against what was
  there *before* the call, so "deleted" a file that never existed is caught
  too. A failed check is not a footnote: it turns the call into `FLAGGED` and
  the failure is what you and the model see. Browser actions are followed by a
  screenshot of the page; screen controls take their own in stage 7.
  What it cannot do: see inside a tool that reports nothing re-checkable. Those
  are *not* quietly counted as verified — the result says
  `[not independently verified: ...]` instead (turn it off with
  `permissions.note_unverified: false`). "I sent that email" has no filesystem
  proof, and pretending otherwise would be worse than saying so.
- **Timeouts and retries.** Every tool call has a timeout (`safety.tool_timeout_s`)
  and is retried once (`safety.tool_retries`) before GARVIS reports the failure.
  Three consecutive failures (`max_consecutive_errors`) and it stops and asks.
- **The kill switch.** `Ctrl+Alt+Esc`, the tray STOP button and the spoken
  stop phrase all call the same switch: it interrupts the model mid-answer,
  stops and mutes speech, halts every open browser profile, stops screen
  guidance, shows a red STOPPED status until you resume — and **terminates any
  command that is still running**. Commands run in their own process group, so
  the whole group is signalled: `SIGTERM` first, then `SIGKILL` after
  `safety.stop_grace_s` (0.5 s) if anything is still alive (`taskkill /T /F` on
  Windows). The tool then reports "Stopped by the kill switch" rather than
  pretending the command finished. What it cannot do: un-run something already
  finished, or cancel a call already inside a third-party function (a
  screenshot backend, a Playwright action) — those end when that call returns.

Logs: everything is appended to `logs/activity_log.txt` (human-readable) and
`logs/activity_log-YYYY-MM-DD.jsonl` (machine-readable), with credentials
redacted before they are written, and tool results cut to one redacted line
(`logging.result_max_chars`). `python main.py --today` (or the tray item)
summarises the day.

---

## Talking to it

Text mode: just type. Voice mode: say the wake word ("Garvis") first, then the
request.

Examples:

```
"what's in my downloads folder?"
"read the file notes.md and summarise it"
"write a shopping list to shopping.txt"            (YELLOW: confirm)
"search the sandbox for 'invoice' and tell me what you find"
"open example.com and tell me the headline"
"go to github.com and check whether I have any new notifications"
"make a screenshot" / "look at my screen and help me with this dialog"
"remember that I prefer dark mode"
"what did you do today?"
"switch to sassy mode"
"stop everything"                                   (kills it instantly)
"where were we?" / "resume"                         (unfinished work)
"rollback plan"                                     (how to undo it)
```

Local commands (they never reach the model):

| Command | What it does |
|---|---|
| `/help` | The list |
| `/reset` | Forget the conversation (memory files stay) |
| `/prompt` | Print the system prompt in use |
| `/tools` | Every registered tool and its tier |
| `/today` | What was logged today |
| `/personality NAME` | Switch tone mode |
| `/memory` | Show what memory was loaded |
| `/quit` | Exit |

---

## Personality modes

`brain.personality` in `config.yaml`, or `/personality <name>`, or just say
"switch to focus mode":

| Mode | Tone |
|---|---|
| `standard` | Clear and neutral. The default. |
| `sassy` | Dry, a little cheeky, never rude, never at the cost of accuracy. |
| `formal` | Precise and professional; "sir/madam" register. |
| `hyped` | Enthusiastic, high energy. |
| `focus` | Terse. Answers and results only, no chatter. |
| `chill` | Relaxed, unhurried, casual. |

The mode changes *how* it talks, never what it is allowed to do.

---

## Crash-resume and rollback

While GARVIS works, it writes what it is doing to `state/task_state.json`:
the task, each step, and how you would undo it. That file is written atomically,
so a power cut cannot corrupt it.

- **After a crash or a kill**, the next start says so and offers the work:
  `python main.py --resume` carries it on, or say **"resume"** / **"where were
  we"** in a session.
- **Quitting on purpose mid-task** keeps the task too — the next start still
  offers it.
- **`--check` never resumes anything**: it reports and leaves the choice to you.
- **Rollback is a plan**: ask for it ("rollback plan") and GARVIS lists the undo
  steps newest-first. Executing them goes through the normal gate, one
  confirmation at a time. Nothing is undone behind your back, and a `delete`
  is reported as *not* recoverable by GARVIS.

---

## Configuration

Everything lives in `config.yaml`, with comments. The sections:

| Section | What it controls |
|---|---|
| `app` | Name, wake name, how it addresses you, data directory |
| `brain` | Ollama host/model, vision model, context, timeouts, personality, system prompt, cloud fallback (off by default) |
| `permissions` | Tier rules, RED keywords, confirm words, timeouts, protected patterns |
| `files` | Sandbox folder, read/write allowlists, denied globs, size limits, symlinks |
| `shell` | Allowed executables, working directory, timeouts, blocked patterns |
| `browser` | Allowed sites, profiles, headless, takeover detection, screenshots |
| `screen` | Capture backend, monitor, guidance cadence, vision model, control on/off |
| `screen.guidance_only` / `allow_control` | Guidance (default) vs. actually driving the mouse |
| `apps` | Launchable programs and close rules |
| `voice_in` | Microphone, Whisper model/device, listen timing, wake word |
| `voice_out` | Engine, voice, speed, volume, device, chunking, barge-in |
| `hotkeys` | The stop hotkey and the backend it is registered with |
| `memory` | Memory files, how much of the log is loaded into context |
| `logging` | Level, file, activity log, redaction patterns |
| `ui` | Tray, overlay, position, opacity, status line, STOP confirmation |
| `state` | Task recording, history, auto-task, resume phrases |
| `safety` | Kill phrases, timeouts, retries, verification, barge-in, wake-up briefing |
| `self_test` | Which checks the self-test runs (a disabled one is reported as skipped, never silently dropped) and its timeout |

The optional cloud fallback lives inside `brain.cloud_fallback` and is
**disabled by default**; turning it on also needs an API key in an environment
variable, and `--check` prints a warning whenever it is on.

Config is validated at startup; anything wrong is printed by `--check` with the
reason. Paths support `~`, `${ENV_VAR}` and are resolved against the project
root.

---

## Tests and demos

```bash
.venv/bin/pytest tests/ -q                 # the full suite: 499 passed, 1 skipped (~95 s)
.venv/bin/pytest tests/test_stage3_permissions.py -v   # one stage
```

Every build stage has its own tests (stage 1 → 9), plus three suites that test
the *guarantees* rather than the features:

- `tests/test_killswitch_work.py` runs real processes, presses stop, and checks
  they really died (including a grandchild that ignores `SIGTERM`).
- `tests/test_log_privacy.py` writes canaries into files, reads them with the
  real tools, and then searches both logs for them.
- `tests/test_verification.py` puts tools that *lie* into the registry — they
  return success without doing anything — and checks that the lie is caught,
  reported as a failure, and written to the log as one.
- `tests/test_cloud_fallback.py` proves the fallback is off by default with a
  bomb client (nothing constructs one), that a missing key stays local, that a
  fallback turn cannot act without `allow_tools`, and that the API key never
  reaches a log.
- `tests/test_config_coverage.py` enforces that `config.yaml` has no dead knobs
  (every key is read), no hidden knobs (every key the code reads is in the file)
  and no unexplained knobs (every key carries a comment, or its name says
  everything). `tests/test_config_wiring.py` then proves the settings it covers
  actually change behaviour. There are also offline demos that need no model, no
microphone and no display:

| Demo | Shows |
|---|---|
| `python tests/demo_session.py` | A whole session: streaming reply, tool call, YELLOW confirm, RED repeat+confirm, denial |
| `python tests/demo_red_gate.py` | Every RED rule, one by one, with the exact wording |
| `python tests/demo_voice.py` | The voice loop with fake ears and mouth: wake, barge-in, kill phrase |
| `python tests/demo_browser.py` | Profiles, blocked sites, CAPTCHA takeover, credential refusal |
| `python tests/demo_screen.py` | Guidance mode, change detection, an injection hidden in a screenshot |
| `python tests/demo_state.py` | The state file, a simulated crash, resume, honest rollback |
| `python tests/self_test.py` | Your actual machine: what works, what is missing, what to install |

---

## Troubleshooting

Start with `python tests/self_test.py` — it names the fix for whatever is wrong.
The most common ones:

| Symptom | Cause | Fix |
|---|---|---|
| `Ollama is not answering` | Ollama is not running | `ollama serve` (or start the Ollama app) |
| `Model 'x' is not installed` | Model never pulled | `ollama pull x` |
| No speech | No TTS engine or no voice file | `pip install piper-tts`, put `.onnx` + `.onnx.json` in `models/voices/`, set `voice_out.voice` |
| Audio plays to the wrong device | Wrong output | `python main.py --devices`, set `voice_out.device` |
| Microphone: no module | `sounddevice`/PortAudio missing | `pip install sounddevice`, Linux also `sudo apt install libportaudio2` |
| It never wakes | Wake model missing or the mic is muted | Set `voice_in.wake.engine: none` (was "Garvis" as text), check `--voice-check` |
| Whisper is very slow | Model too big for the CPU | Use `base.en`/`small.en`, or a GPU (`voice_in.stt.device: cuda`) |
| First reply takes forever | Model loading into RAM | Normal for the first call; `brain.keep_alive: 30m` keeps it resident |
| `playwright` errors | Chromium not installed | `playwright install chromium` |
| CAPTCHA blocks it | By design | Solve it yourself in the window, then say "continue" |
| Screen is black / capture fails | OS permission or Wayland | macOS: allow Screen Recording; Linux: use an X11 session |
| `Ctrl+Alt+Esc` does nothing | No `keyboard`/`pynput`, or Wayland | `pip install keyboard`; on Wayland use the tray STOP or say "stop everything" |
| No tray icon / no overlay | Missing packages | `pip install pystray pillow`; Linux overlay needs `python3-tk` |
| `Permission denied` writing files | Path is outside the allowlists | Add the folder to `files.allowed_write` |
| Tool calls are refused with `BLOCKED` | Protected pattern or outside allowlist | Check the reason in the log; that is the gate working |
| It says it cannot undo something | No automatic inverse (deletes, external effects) | It is telling you the truth; restore from your own backup |

Logs are in `logs/` (`garvis.log` for detail, `activity_log.txt` for the audit
trail). `--log-level DEBUG` is much noisier but shows every decision the gate
makes.

---

## Project layout

```
garvis/
├── main.py                 entry point, the loops, UI wiring, CLI flags
├── config.yaml             every setting, commented
├── requirements.txt        dependencies, grouped by feature
├── core/
│   ├── brain.py            Ollama client, streaming, tool-call loop, prompting
│   ├── voice_in.py         microphone, Whisper STT, wake word, segmentation
│   ├── voice_out.py        Piper/Kokoro/pyttsx3 engines, players, chunking
│   ├── memory.py           profile + project log, prompt memory block
│   ├── permissions.py      the GREEN/YELLOW/RED gate, confirmers, RED keywords
│   ├── logger.py           activity log (txt + jsonl), redaction, "today"
│   ├── state.py            task state, crash-resume, rollback plans
│   ├── ui.py               tray icon, overlay, status, STOP button
│   ├── killswitch.py       Ctrl+Alt+Esc, spoken stop phrase, halt everything
│   ├── browser.py          Playwright driver, profiles, takeover detection
│   ├── screen.py           capture, vision guidance, optional control
│   └── safety.py           shared safety helpers (redaction, classification)
├── tools/                  what the model can actually call
│   ├── base.py             ToolRegistry, ToolResult, tiers
│   ├── builtin.py          clock, assistant.about, capabilities, personality
│   ├── files.py            read/write/list/search/stat/copy/move/delete
│   ├── shell.py            allowlisted command execution
│   ├── browser.py          open/read/click/type/press/screenshot/profiles
│   ├── screen.py           capture/look/guide_start/guide_stop/click/type
│   ├── apps.py             open/close allowlisted desktop apps
│   └── tasks.py            task.start/finish/status/steps/resume/rollback_plan
├── prompts/system_prompt.md  the persona and the rules it is told about
├── memory/                 profile.md (you) + project_log.md (what was done)
├── profiles/               browser profiles (separate logins, gitignored)
├── sandbox/                scratch folder for files (gitignored)
├── logs/                   activity log + detail log (gitignored)
├── state/                  task_state.json for crash-resume (gitignored)
└── tests/                  per-stage tests, offline demos, self_test.py
```

---

## Privacy and security

- **Local by default.** Ollama, Whisper, Piper and the vision model all run on
  this machine. The cloud fallback is `enabled: false` and, when off, no code
  path can send a prompt anywhere.
- **Screenshots never leave the machine** unless `screen.allow_cloud_vision` is
  explicitly true.
- **Passwords are never read, typed, displayed or logged** — and the refusals
  are covered by tests, including a live check in the self-test.
- **Redaction** runs before anything is written to a log
  (`logging.redact_patterns`): password-like keys, card numbers, tokens, keys.
- **A tool result cannot smuggle a secret into the log.** Tool output is
  redacted *and* cut to one line before it is written
  (`logging.result_max_chars`, default 400): the model and you still get the
  whole thing, the log does not. `tests/test_log_privacy.py` drives real files
  containing canaries through `files.read`, `files.search` and `shell.run`, then
  reads the log back looking for them.
- **The log is an audit trail.** `logs/activity_log.txt` answers "what did you
  do today?" honestly, including the things that failed and the things that
  were refused.
- **State-changing actions are checked afterwards** (see below), and a check
  that comes back negative is reported as a *failure* — a tool cannot claim it
  wrote or deleted something that is not there.
- **Give it the smallest allowlists you can live with**, and prefer running it
  under your own user account with the browser profiles you are willing to have
  logged in.
- **Optional password managers**: the design assumes you log in yourself once
  and the profile remembers. If you use Bitwarden or KeePassXC, keep them in
  manual mode — GARVIS will not read from them (the vault files are in the
  protected patterns).

---

## Where this came from

Built in ten stages, each one tested before the next: text loop → memory and
logging → the permission gate → file and shell tools → voice in/out → browser →
screen and vision → tray UI, hotkey and crash-resume → self-test and the wake-up
routine → this README. The stage tests live in `tests/test_stage*.py`, and the
demos above are the guided tour of each stage.

"""Configuration loading for GARVIS.

One file the user edits: config.yaml. Everything else reads from here.

Features
--------
* YAML with comments preserved (we never write back to the file).
* ``~`` and ``${ENV_VAR}`` expansion.
* Relative paths resolved against the project root, so GARVIS works no matter
  which directory you launch it from.
* Dotted-path access: ``cfg.get("brain.model")``.
* Human-readable validation warnings instead of stack traces on a typo.

Run ``python main.py --check`` to print a resolved summary of this file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

# Project root = the folder containing core/ , tools/ , config.yaml ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

VALID_PERSONALITIES = ("standard", "sassy", "formal", "hyped", "focus", "chill")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or structurally unusable."""


def _expand_env(value: str) -> str:
    """Replace ${VAR} with the environment value; missing vars become ''."""

    def _sub(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_RE.sub(_sub, value)


def _walk_expand(node: Any) -> Any:
    """Recursively expand env vars in every string of a config tree."""
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, list):
        return [_walk_expand(item) for item in node]
    if isinstance(node, dict):
        return {key: _walk_expand(val) for key, val in node.items()}
    return node


def _parse_command_list(raw: Iterable[Any]) -> list[str]:
    """Shell allowlists are written either as YAML lists or as comma strings.

    Both of these mean the same thing::

        allowlist: [echo, ls, git]
        allowlist:
          - echo, ls, git
    """
    out: list[str] = []
    for item in raw or []:
        if item is None:
            continue
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


class Config:
    """Wraps the parsed config.yaml tree with dotted access and helpers."""

    def __init__(self, data: dict[str, Any], path: Path, root: Path | None = None) -> None:
        self.data: dict[str, Any] = data or {}
        self.config_path = Path(path)
        self.root = Path(root) if root else self.config_path.resolve().parent
        self.warnings: list[str] = []
        self._runtime_overrides: dict[str, Any] = {}
        self.validate()

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, root: Path | None = None) -> "Config":
        cfg_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_absolute():
            cfg_path = (PROJECT_ROOT / cfg_path).resolve()
        if not cfg_path.exists():
            raise ConfigError(
                f"Config file not found: {cfg_path}\n"
                f"Copy config.yaml into the project root, or pass --config <path>."
            )
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level.")
        return cls(_walk_expand(raw), cfg_path, root or cfg_path.resolve().parent)

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """cfg.get('brain.cloud_fallback.enabled', False)"""
        if dotted in self._runtime_overrides:
            return self._runtime_overrides[dotted]
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return default if node is None else node

    def require(self, dotted: str) -> Any:
        missing = object()
        value = self.get(dotted, missing)
        if value is missing:
            raise ConfigError(f"Missing required config key: {dotted}")
        return value

    def set(self, dotted: str, value: Any) -> None:
        """Session-scoped override (used for personality swaps, --model, etc.).

        Deliberately in-memory only: rewriting config.yaml would destroy the
        comments, which are the documentation.
        """
        self._runtime_overrides[dotted] = value

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        return value if isinstance(value, dict) else {}

    # -- paths -------------------------------------------------------------
    def resolve_path(self, value: str | os.PathLike[str], base: Path | None = None) -> Path:
        """Resolve a config path: expand ~ and ${VARS}, anchor relative paths."""
        text = _expand_env(str(value)).strip()
        base = Path(base) if base else self.root
        p = Path(text).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        return p

    def path(self, dotted: str, default: str | None = None) -> Path:
        raw = self.get(dotted, default)
        if raw is None:
            raise ConfigError(f"No path configured at '{dotted}'")
        return self.resolve_path(raw)

    # -- convenience properties -------------------------------------------
    @property
    def project_root(self) -> Path:
        return self.root

    @property
    def assistant_name(self) -> str:
        return str(self.get("app.name", "GARVIS"))

    @property
    def user_name(self) -> str:
        return str(self.get("app.user_name", "Boss"))

    @property
    def personality(self) -> str:
        value = str(self.get("brain.personality", "standard")).lower().strip()
        return value if value in VALID_PERSONALITIES else "standard"

    @property
    def model(self) -> str:
        return str(self.get("brain.model", "llama3.1:8b"))

    @property
    def vision_model(self) -> str:
        return str(self.get("brain.vision_model", "llava:7b"))

    @property
    def ollama_host(self) -> str:
        return str(self.get("brain.host", "http://127.0.0.1:11434")).rstrip("/")

    @property
    def cloud_fallback_enabled(self) -> bool:
        return bool(self.get("brain.cloud_fallback.enabled", False))

    @property
    def personality_default(self) -> str:
        return self.personality

    def allowed_folders(self, kind: str = "read") -> list[Path]:
        raw = self.get(f"files.allowed_{kind}", []) or []
        return [self.resolve_path(item) for item in raw]

    def allowed_sites(self) -> list[str]:
        return [str(s).strip().lower() for s in (self.get("browser.allowed_sites", []) or []) if str(s).strip()]

    def blocked_sites(self) -> list[str]:
        return [str(s).strip().lower() for s in (self.get("browser.bank_sites", []) or []) if str(s).strip()]

    def shell_allowlist(self) -> list[str]:
        return _parse_command_list(self.get("shell.allowlist", []))

    def kill_phrases(self) -> list[str]:
        return [str(p).strip().lower() for p in (self.get("safety.kill_phrases", []) or []) if str(p).strip()]

    def tool_timeout(self) -> float:
        return float(self.get("safety.tool_timeout_s", 60))

    def tool_retries(self) -> int:
        return int(self.get("safety.tool_retries", 1))

    # -- validation --------------------------------------------------------
    def validate(self) -> list[str]:
        """Collect human-readable warnings. Never raises for soft problems."""
        warnings: list[str] = []

        personality = str(self.get("brain.personality", "standard")).lower()
        if personality not in VALID_PERSONALITIES:
            warnings.append(
                f"brain.personality='{personality}' is not one of {', '.join(VALID_PERSONALITIES)}; "
                f"falling back to 'standard'."
            )

        provider = str(self.get("brain.provider", "ollama")).lower()
        if provider not in ("ollama", "openai_compatible", "anthropic"):
            warnings.append(f"brain.provider='{provider}' is unknown; expected 'ollama' (default).")

        if self.cloud_fallback_enabled:
            key_env = str(self.get("brain.cloud_fallback.api_key_env", ""))
            if not key_env:
                warnings.append("Cloud fallback is enabled but brain.cloud_fallback.api_key_env is empty.")
            elif not os.environ.get(key_env):
                warnings.append(
                    f"Cloud fallback is enabled but environment variable {key_env} is not set; "
                    f"GARVIS will stay local."
                )

        prompt_rel = str(self.get("brain.system_prompt", ""))
        if prompt_rel and not self.resolve_path(prompt_rel).exists():
            warnings.append(f"System prompt file not found: {self.resolve_path(prompt_rel)}")

        if not self.get("files.allowed_read"):
            warnings.append("files.allowed_read is empty: file reads will all be denied.")
        if not self.get("files.allowed_write"):
            warnings.append("files.allowed_write is empty: file writes will all be denied.")

        tts_engine = str(self.get("voice_out.engine", "piper")).lower()
        if tts_engine not in ("piper", "kokoro", "pyttsx3"):
            warnings.append(f"voice_out.engine='{tts_engine}' is unknown (piper|kokoro|pyttsx3).")

        stt_engine = str(self.get("voice_in.stt.engine", "faster_whisper")).lower()
        if stt_engine not in ("faster_whisper", "whisper"):
            warnings.append(f"voice_in.stt.engine='{stt_engine}' is unknown; using faster_whisper.")

        return warnings


def load_config(path: str | os.PathLike[str] | None = None, root: Path | None = None) -> Config:
    """Convenience wrapper: ``from core.config import load_config``."""
    return Config.load(path, root)


def describe(cfg: Config) -> str:
    """Short human-readable summary used by ``main.py --check``."""
    lines = [
        f"config file      : {cfg.config_path}",
        f"assistant        : {cfg.assistant_name} (user: {cfg.user_name})",
        f"brain provider   : {cfg.get('brain.provider')}",
        f"ollama host      : {cfg.ollama_host}",
        f"model            : {cfg.model}",
        f"vision model     : {cfg.vision_model}",
        f"personality      : {cfg.personality}",
        f"cloud fallback   : {'ENABLED' if cfg.cloud_fallback_enabled else 'disabled (local only)'}",
        f"voice out        : {cfg.get('voice_out.engine')} / {cfg.get('voice_out.voice')} "
        f"({'on' if cfg.get('voice_out.enabled') else 'off'})",
        f"voice in         : whisper={cfg.get('voice_in.stt.model')} "
        f"wake={cfg.get('voice_in.wake.model')} {'on' if cfg.get('voice_in.enabled') else 'off'}",
        f"allowed reads    : {', '.join(str(p) for p in cfg.allowed_folders('read')) or '(none)'}",
        f"allowed writes   : {', '.join(str(p) for p in cfg.allowed_folders('write')) or '(none)'}",
        f"allowed sites    : {', '.join(cfg.allowed_sites()) or '(none)'}",
        f"kill switch      : {cfg.get('hotkeys.killswitch')} + {cfg.kill_phrases()[:2]}",
    ]
    return "\n".join(lines)

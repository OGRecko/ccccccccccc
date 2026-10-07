"""Shared pytest fixtures.

Tests never touch the real memory files, logs, or sandbox: every fixture points
the config at a tmp_path and switches Ollama to the mock server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# tests/self_test.py is a runnable script (`python tests/self_test.py`), not a
# pytest module: pytest would collect it just because the name ends in _test.py.
collect_ignore = ["self_test.py"]

from core.config import Config  # noqa: E402
from core.logger import ActivityLogger, configure_activity_logger, setup_logging  # noqa: E402
from core.memory import Memory  # noqa: E402
from tests.mock_ollama import MockOllama  # noqa: E402


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """The real config.yaml, with every path relocated into a temp dir."""
    config = Config.load(PROJECT_ROOT / "config.yaml")
    config.validate()

    sandbox = tmp_path / "sandbox"
    memory_dir = tmp_path / "memory"
    logs = tmp_path / "logs"
    state = tmp_path / "state"
    for folder in (sandbox, memory_dir, logs, state):
        folder.mkdir(parents=True, exist_ok=True)

    config.set("files.sandbox_dir", str(sandbox))
    config.set("files.allowed_read", [str(sandbox), str(memory_dir)])
    config.set("files.allowed_write", [str(sandbox), str(memory_dir)])
    config.set("shell.default_cwd", str(sandbox))
    config.set("memory.dir", str(memory_dir))
    config.set("memory.profile_file", str(memory_dir / "profile.md"))
    config.set("memory.project_log_file", str(memory_dir / "project_log.md"))
    config.set("logging.file", str(logs / "garvis.log"))
    config.set("logging.activity_log", str(logs / "activity_log.txt"))
    config.set("logging.activity_jsonl", str(logs / "activity_log.jsonl"))
    config.set("state.dir", str(state))
    config.set("browser.profiles_dir", str(tmp_path / "profiles"))
    config.set("browser.screenshots_dir", str(logs / "browser_shots"))
    config.set("screen.frames_dir", str(logs / "screen_frames"))
    config.set("screen.shots_dir", str(logs / "screen_shots"))
    config.set("files.follow_symlinks", False)
    return config


@pytest.fixture()
def activity(cfg: Config) -> ActivityLogger:
    log_dir = cfg.resolve_path(cfg.get("logging.file")).parent
    logger = ActivityLogger(
        log_dir=log_dir,
        activity_log_name="activity_log.txt",
        redact_patterns=cfg.get("logging.redact_patterns", []),
    )
    configure_activity_logger(cfg)
    setup_logging(cfg.resolve_path(cfg.get("logging.file")), level="DEBUG", console=False)
    # Point the module-level singleton at our temp logger.
    import core.logger as logger_module

    logger_module._ACTIVITY = logger
    return logger


@pytest.fixture()
def memory(cfg: Config) -> Memory:
    mem = Memory.from_config(cfg)
    mem.ensure_files()
    return mem


@pytest.fixture()
def mock_ollama() -> MockOllama:
    with MockOllama() as server:
        yield server

"""GARVIS tools package.

Importing this package must stay side-effect free (no network, no browser, no
microphone). ``build_registry`` wires up the tools that are enabled by config;
missing optional dependencies downgrade a tool to "unavailable" with a clear
reason instead of crashing the app.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

from .base import GREEN, RED, YELLOW, Tool, ToolError, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from core.config import Config

__all__ = [
    "GREEN",
    "YELLOW",
    "RED",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_registry",
]


def build_registry(
    cfg: "Config",
    log: object | None = None,
    services: dict[str, object] | None = None,
) -> ToolRegistry:
    """Create the registry and register every tool whose deps are present.

    Each tool module exposes ``register(registry, cfg, log, services)``. A module
    whose optional dependency is missing is skipped with a warning, so stage 1
    works with nothing but PyYAML and requests installed.
    """
    registry = ToolRegistry()
    services = services or {}

    # Always available: stdlib only.
    from . import builtin  # deliberate lazy import

    builtin.register(registry, cfg, log, services)

    for module_name in ("files", "shell", "browser", "screen", "apps", "tasks"):
        try:
            if importlib.util.find_spec(f"tools.{module_name}") is None:
                if log:
                    getattr(log, "debug", lambda *_: None)(
                        "tools.%s not implemented yet; skipping", module_name
                    )
                continue
            module = importlib.import_module(f"tools.{module_name}")
        except ImportError as exc:  # missing optional dependency
            if log:
                getattr(log, "warning", lambda *_: None)(
                    "tools.%s not loaded (missing dependency: %s)", module_name, exc
                )
            continue
        module.register(registry, cfg, log, services)

    if log:
        tiers = {}
        for tool in registry.all():
            tiers[tool.tier] = tiers.get(tool.tier, 0) + 1
        getattr(log, "info", lambda *_: None)(
            "tools ready: %d total (%s)", len(registry),
            ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())),
        )
    return registry

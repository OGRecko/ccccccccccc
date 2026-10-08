"""Tool framework.

A tool is a plain Python function with a JSON schema, a declared *default*
permission tier, and metadata. The brain never calls a tool directly: it asks
:class:`~core.permissions.PermissionGate`, which classifies the call and either
auto-runs it, asks for confirmation, or refuses.

Tiers declared here are only a starting point. Rules and keyword checks in
``config.yaml`` can promote an action (never demote one the gate considers RED
by keyword). A tool that declares no tier fails closed as RED.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core import safety

GREEN = "green"
YELLOW = "yellow"
RED = "red"
TIER_ORDER = {GREEN: 0, YELLOW: 1, RED: 2}


@dataclass
class ToolResult:
    """What a tool returns.

    ``content`` goes back to the model, fenced as untrusted data. Keep it short
    and factual: it lands in the context window.

    ``display`` is for the human (UI/log) and may be richer.
    """

    ok: bool
    content: str
    display: str | None = None
    untrusted: bool = True
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, display: str | None = None, **meta: Any) -> "ToolResult":
        return cls(ok=True, content=content, display=display, meta=meta)

    @classmethod
    def failure(cls, error: str, content: str | None = None, **meta: Any) -> "ToolResult":
        return cls(
            ok=False,
            content=content or f"ERROR: {error}",
            display=error,
            error=error,
            meta=meta,
        )

    def for_model(self, tool_name: str, max_chars: int = 20000) -> str:
        """Serialise for the LLM context, with the untrusted fence applied."""
        if self.untrusted:
            return safety.wrap_tool_result(tool_name, self.content, max_chars=max_chars)
        return self.content


class ToolError(RuntimeError):
    """Raised for programming errors in a tool (bad args handled by the gate)."""


@dataclass
class Tool:
    """A callable exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., ToolResult | str]
    tier: str = RED
    category: str = "general"
    #: True when calling this tool cannot change local state (used for narration).
    readonly: bool = False
    #: Args that are free text going to a third party (email body, tweet, ...).
    #: The gate raises the tier when these are non-empty.
    content_args: tuple[str, ...] = ()
    #: Args holding filesystem paths: the gate checks them against the allowlists.
    path_args: tuple[str, ...] = ()
    #: Where this tool resolves *relative* paths, so the gate can check and
    #: verify the same file the tool actually touches. "" = like the shell
    #: (shell.default_cwd); "sandbox" = files.sandbox_dir (what tools/files.py
    #: does). A mismatch here means the gate inspects a different file than the
    #: tool wrote, which is how a working write can look like a failed one.
    path_base: str = ""
    #: Args holding URLs/domains: the gate checks them against allowed_sites.
    url_args: tuple[str, ...] = ()
    #: Args holding a command line: the gate checks the executable allowlist and
    #: the blocked-pattern list.
    command_args: tuple[str, ...] = ()
    #: Optional extra check: guard(args) -> (tier_to_promote_to | None, reason).
    #: Tools use it for checks only they can do (e.g. "this selector is a Pay button").
    guard: "Callable[[dict[str, Any]], tuple[str | None, str]] | None" = None
    #: How to describe this action *out loud* for the RED repeat-back check,
    #: e.g. lambda a: f"delete the file {a['path']}". Falls back to a generic
    #: "<verb> <args>" phrase. The console form stays precise and machine-like.
    spoken_action: "Callable[[dict[str, Any]], str] | None" = None
    #: Per-call timeout in seconds; None = safety.tool_timeout_s.
    timeout_s: float | None = None
    #: Human-facing example used by docs/UI.
    example: str | None = None
    #: Args hidden from logs (never secrets; the gate refuses those anyway).
    log_omit: tuple[str, ...] = ()
    #: Args whose *values* may be secrets (a password someone tried to type).
    #: They are replaced by "[hidden]" in every log line, the confirmation
    #: summary and the spoken challenge, so a secret can never be written down
    #: even when the call is refused.
    secret_args: tuple[str, ...] = ()

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Call the underlying function, normalising return values.

        We deliberately do not catch broad exceptions here: the executor in
        core/permissions.py handles retries and error reporting so that a
        failure is visible and logged rather than swallowed.
        """
        signature = inspect.signature(self.func)
        accepted = set(signature.parameters)
        filtered = {k: v for k, v in (args or {}).items() if k in accepted}
        result = self.func(**filtered)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, str):
            return ToolResult.success(result)
        if result is None:
            return ToolResult.success("done")
        return ToolResult.success(str(result))

    def schema(self) -> dict[str, Any]:
        """Ollama / OpenAI style function schema."""
        params = dict(self.parameters or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def schema_for_ollama(self) -> dict[str, Any]:
        """Ollama accepts the OpenAI function shape directly."""
        return self.schema()

    def schema_for_openai(self) -> dict[str, Any]:
        return self.schema()

    def arg_summary(self, args: dict[str, Any], limit: int = 160) -> str:
        """One-line description of a call, for confirmation prompts and logs."""
        if not args:
            return self.name
        parts = []
        for key, value in args.items():
            if key in self.log_omit or key in self.secret_args:
                value = "[hidden]"
            text = str(value).replace("\n", " ")
            if len(text) > limit:
                text = text[: limit - 3] + "..."
            parts.append(f"{key}={text!r}" if " " in text else f"{key}={text}")
        return f"{self.name}({', '.join(parts)})"

    def to_doc(self) -> str:
        lines = [f"### {self.name}  [{self.tier.upper()}]"]
        lines.append(self.description.strip())
        props = (self.parameters or {}).get("properties", {}) or {}
        required = set((self.parameters or {}).get("required", []) or [])
        if props:
            lines.append("Arguments:")
            for pname, spec in props.items():
                flag = " (required)" if pname in required else ""
                desc = spec.get("description", "") if isinstance(spec, dict) else ""
                lines.append(f"  - {pname}: {spec.get('type', 'any') if isinstance(spec, dict) else 'any'}{flag} - {desc}")
        if self.example:
            lines.append(f"Example: {self.example}")
        return "\n".join(lines)


class ToolRegistry:
    """Holds the tools available this session."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- registration ------------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ToolError(f"Duplicate tool name: {tool.name}")
        if tool.tier not in TIER_ORDER:
            raise ToolError(f"Tool {tool.name} has invalid tier {tool.tier!r}")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str = "",
        description: str = "",
        parameters: dict[str, Any] | None = None,
        tier: str = RED,
        category: str = "general",
        readonly: bool = False,
        content_args: tuple[str, ...] = (),
        path_args: tuple[str, ...] = (),
        path_base: str = "",
        url_args: tuple[str, ...] = (),
        command_args: tuple[str, ...] = (),
        guard: "Callable[[dict[str, Any]], tuple[str | None, str]] | None" = None,
        spoken_action: "Callable[[dict[str, Any]], str] | None" = None,
        timeout_s: float | None = None,
        example: str | None = None,
        log_omit: tuple[str, ...] = (),
        secret_args: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form: ``@registry.tool(name=..., tier=GREEN)``."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                Tool(
                    name=name or func.__name__,
                    description=description or (func.__doc__ or "").strip(),
                    parameters=parameters or {"type": "object", "properties": {}},
                    func=func,
                    tier=tier,
                    category=category,
                    readonly=readonly,
                    content_args=content_args,
                    path_args=path_args,
                    path_base=path_base,
                    url_args=url_args,
                    command_args=command_args,
                    guard=guard,
                    spoken_action=spoken_action,
                    timeout_s=timeout_s,
                    example=example,
                    log_omit=log_omit,
                    secret_args=secret_args,
                )
            )
            return func

        return decorator

    # -- access ------------------------------------------------------------
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    def by_category(self) -> dict[str, list[Tool]]:
        out: dict[str, list[Tool]] = {}
        for tool in self.all():
            out.setdefault(tool.category, []).append(tool)
        return out

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # -- schemas -----------------------------------------------------------
    def schemas(self, exclude: set[str] | None = None, only_readonly: bool = False) -> list[dict[str, Any]]:
        out = []
        for tool in self.all():
            if exclude and tool.name in exclude:
                continue
            if only_readonly and not tool.readonly:
                continue
            out.append(tool.schema())
        return out

    def docs(self, include_red: bool = False) -> str:
        chunks = []
        for category, tools in sorted(self.by_category().items()):
            chunks.append(f"## {category}")
            for tool in tools:
                if tool.tier == RED and not include_red:
                    # Still documented: the model must know what exists and that
                    # it needs explicit confirmation.
                    pass
                chunks.append(tool.to_doc())
            chunks.append("")
        return "\n".join(chunks)


class Timing:
    """Small helper so every tool call is logged with a duration."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0

    def __enter__(self) -> "Timing":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

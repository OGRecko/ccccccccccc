"""Interactive demo of the RED gate, with no model and no network.

    python tests/demo_red_gate.py

It registers the same tools GARVIS registers, points a mock Ollama at the brain,
and asks the model to do something RED. You then get the real two-step console
challenge. Type nothing (or Ctrl+C) to see the denial path; repeat the action
and type `confirm` to see the approved path.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.brain import Brain  # noqa: E402
from core.config import Config  # noqa: E402
from core.logger import ActivityLogger, configure_activity_logger, get_logger, setup_logging  # noqa: E402
from core.memory import Memory  # noqa: E402
from core.permissions import PermissionGate  # noqa: E402
from tests.mock_ollama import MockOllama, text_chunks, tool_call_chunks  # noqa: E402
from tools import build_registry  # noqa: E402


def main() -> int:
    cfg = Config.load(PROJECT_ROOT / "config.yaml")
    sandbox = PROJECT_ROOT / "sandbox"
    memory_dir = PROJECT_ROOT / "memory"
    logs = PROJECT_ROOT / "logs"
    sandbox.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    cfg.set("files.allowed_read", [str(sandbox), str(memory_dir)])
    cfg.set("files.allowed_write", [str(sandbox), str(memory_dir)])
    cfg.set("shell.default_cwd", str(sandbox))

    setup_logging(logs / "garvis.log", level="INFO", console=False)
    configure_activity_logger(cfg)
    activity = ActivityLogger(log_dir=logs)
    import core.logger as logger_module

    logger_module._ACTIVITY = activity

    log = get_logger("demo")

    def responder(payload):
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            return text_chunks("Understood. I stopped where you told me to.") + [{"done": True}]
        if last.get("role") == "user":
            # The "model" asks for something RED: writing a password into the profile.
            # The word 'Password' is a red keyword, so the gate will demand approval.
            return tool_call_chunks(
                "memory.set_profile_field",
                {"field": "Password", "value": "hunter2-should-never-be-logged"},
                preamble="Sure, I will store that for you. ",
            ) + [{"done": True}]
        return text_chunks("Hello.") + [{"done": True}]

    with MockOllama(responder=responder) as mock:
        cfg.set("brain.host", mock.url)
        memory = Memory.from_config(cfg)
        memory.ensure_files()
        services: dict = {"activity": activity, "memory": memory,
                          "notifier": lambda text: print(f"\n  [GARVIS speaks] {text}")}
        registry = build_registry(cfg, log, services)
        services["registry"] = registry
        gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                              log=log, registry=registry)
        brain = Brain(cfg, registry=registry, gate=gate, memory=memory, activity=activity)

        print("\n=== GARVIS RED-gate demo ===")
        print("The mock model will try: memory.set_profile_field(field='Password', value='...')")
        print("That contains a red keyword, so the gate requires the two-step challenge.\n")
        result = brain.respond(
            "Please save my password for later.",
            on_event=lambda e: print(f"  [{e.kind}] {e.text[:120]}", flush=True)
            if e.kind in ("tool_call", "tool_result", "status") else None,
        )
        print("\n--- model's final message ---")
        print(result.reply or "(nothing)")
        print("\n--- gate decisions for this run ---")
        for record in activity.read_records():
            if record.get("kind") == "permission":
                print(f"  {record.get('decision'):<10} {record.get('note', '')[:110]}")
        print("\n--- activity_log.txt (secrets must be redacted) ---")
        text = (logs / "activity_log.txt").read_text()
        print(text[-900:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

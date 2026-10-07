"""A full simulated GARVIS session: no model, no microphone, no network.

    python tests/demo_session.py                 # interactive: type your answers
    printf 'write a note\\nyes\\n' | python tests/demo_session.py

A fake Ollama plays the model and asks for real tools; everything else is the
real code path: the real permission gate, the real file/shell tools, the real
activity log. Use it to see exactly how confirmations, verification and denial
behave before you point GARVIS at a real model.

Scripted model behaviour:
    "write ..."   -> files.write(path='note.txt', text=...)     [YELLOW: needs a yes]
    "time"        -> clock.now()                                [GREEN: no prompt]
    "search ..."  -> files.search(...)                          [GREEN]
    "delete ..."  -> files.delete(path='note.txt')              [RED: repeat + confirm]
    anything else -> a plain spoken reply
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
from core.permissions import ConsoleConfirmer, PermissionGate  # noqa: E402
from tests.mock_ollama import MockOllama, text_chunks, tool_call_chunks  # noqa: E402
from tools import build_registry  # noqa: E402

NOTE = "note.txt"
NOTE_TEXT = "Shopping: coffee, oat milk, and a new notebook."


def plan_for(user_text: str) -> tuple[str, dict, str]:
    """Decide what the 'model' will do, the way a real tool-calling LLM would."""
    lowered = user_text.lower()
    if lowered.startswith("write") or "write a note" in lowered:
        return ("files.write", {"path": NOTE, "text": NOTE_TEXT}, "Writing that down for you. ")
    if "delete" in lowered or "remove" in lowered:
        return ("files.delete", {"path": NOTE}, "That is a delete, so I need your say-so. ")
    if "search" in lowered or "find" in lowered:
        return ("files.search", {"query": "coffee"}, "Looking through your files. ")
    if "time" in lowered or "date" in lowered:
        return ("clock.now", {}, "Checking the clock. ")
    if "status" in lowered or "capabilities" in lowered or "what can you do" in lowered:
        return ("assistant.list_capabilities", {}, "Here is what I can reach right now. ")
    return ("", {}, "")


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

    def responder(payload: dict) -> list[dict]:
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            result_text = str(last.get("content", "")).upper()
            if "DENIED" in result_text or "BLOCKED" in result_text:
                return text_chunks(
                    "That was refused by the permission gate, so I did not do it. "
                    "Tell me if you want to approve it properly."
                ) + [{"done": True}]
            if "ERROR" in result_text:
                return text_chunks("That failed. Here is what the tool said: see the logs.") + [{"done": True}]
            return text_chunks("Done. I checked the result before telling you that.") + [{"done": True}]
        user_text = str(last.get("content", ""))
        tool, args, preamble = plan_for(user_text)
        if tool:
            return tool_call_chunks(tool, args, preamble=preamble) + [{"done": True}]
        return text_chunks(
            "I am the scripted model in this demo, so I can only show you the plumbing. "
            "Try: write a note, what time is it, search for coffee, or delete the note."
        ) + [{"done": True}]

    with MockOllama(responder=responder) as mock:
        cfg.set("brain.host", mock.url)
        memory = Memory.from_config(cfg)
        memory.ensure_files()

        services: dict = {
            "activity": activity,
            "memory": memory,
            "notifier": lambda text: print(f"\n  >> GARVIS (voice): {text}"),
        }
        registry = build_registry(cfg, log, services)
        services["registry"] = registry
        gate = PermissionGate(
            cfg=cfg, activity=activity, services=services, log=log,
            confirmer=ConsoleConfirmer(), registry=registry,
        )
        services["gate"] = gate
        brain = Brain(cfg, registry=registry, gate=gate, memory=memory, activity=activity)
        services["brain"] = brain

        print("\n" + "=" * 74)
        print("  GARVIS simulated session (mock model, real tools and real gate)")
        print("=" * 74)
        print("  Try: 'write a note'   'what time is it'   'search for coffee'   'delete the note'")
        print("       'status'   'what did you do today'   /quit")
        print("=" * 74 + "\n")

        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/quit", "/exit", "quit", "exit"):
                break
            if line.startswith("/today"):
                print("\n" + activity.today_report() + "\n")
                continue

            def on_event(event) -> None:
                if event.kind == "delta":
                    print(event.text, end="", flush=True)
                elif event.kind == "tool_call":
                    print(f"\n  [tool] {event.text}({event.data.get('args')})", flush=True)
                elif event.kind == "tool_result":
                    first = event.text.splitlines()[0][:110]
                    print(f"  [result] {first}", flush=True)

            result = brain.respond(line, on_event=on_event)
            print()
            if result.reply:
                print(f"GARVIS> {result.reply}")
            print()

        print("\n--- what the activity log recorded ---")
        for record in activity.read_records():
            if record.get("kind") == "permission":
                print(f"  permission  {record.get('decision'):<10} {record.get('tier'):<6} "
                      f"{record.get('tool')}  {record.get('note', '')[:70]}")
            elif record.get("kind") == "tool":
                print(f"  tool        {'ok' if record.get('ok') else 'failed':<10} "
                      f"{record.get('tool')}  verified={bool(record.get('result'))}")

        print("\n--- sandbox now contains ---")
        for entry in sorted(sandbox.iterdir()):
            print(f"  {entry.name}  ({entry.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

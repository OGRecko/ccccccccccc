"""Task state, crash-resume and the tray/overlay, offline: no model, no display.

    python tests/demo_state.py

The real code path is used - the real state store, the real permission gate, the
real UI manager - in a throwaway folder, so nothing here touches your real
state/ or logs/. A fake tray front end stands in for the real one, which needs a
desktop.

What it shows:
  1. the state file: what it records while a task runs, written atomically
  2. a clean quit that happens mid-task is kept, and the next start offers it
  3. a hard kill (no chance to clean up) is reported as a crash, honestly
  4. `--resume` carrying the task on, and the difference it makes
  5. the rollback plan: described, never executed, and honest when it is empty
  6. the task tools going through the real gate (resume is YELLOW)
  7. a damaged state file: read as "nothing recorded", never a crash
  8. the UI: statuses fanning out, STOP reaching the kill switch, and the
     fallback to the terminal when there is no tray/tkinter on this machine
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Config  # noqa: E402
from core.logger import ActivityLogger, configure_activity_logger  # noqa: E402
from core.permissions import PermissionGate, ScriptedConfirmer  # noqa: E402
from core.state import StateStore  # noqa: E402
from core.ui import FakeUI, UIManager, build_ui  # noqa: E402
from tools import build_registry  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show(label: str, outcome, limit: int = 260) -> None:
    verdict = "OK " if outcome.ok else "REFUSED"
    print(f"\n[{verdict}] {label}")
    print(f"   tier={outcome.tier}  decision={outcome.decision}")
    body = " ".join(str(outcome.content or outcome.display or "").split())
    print(f"   -> {body[:limit]}{'...' if len(body) > limit else ''}")


def state_file(cfg: Config) -> Path:
    return Path(str(cfg.get("state.dir"))) / "task_state.json"


def dump(cfg: Config, label: str) -> None:
    """Print what is on disk right now - the file is the source of truth."""
    path = state_file(cfg)
    if not path.exists():
        print(f"   {label}: {path.name} does not exist")
        return
    data = json.loads(path.read_text())
    task = data.get("task") or {}
    print(f"   {label}: clean_exit={data.get('clean_exit')} "
          f"task={task.get('description')!r} status={task.get('status')} "
          f"steps={len(task.get('steps') or [])}")


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="garvis-demo-state-"))
    cfg = Config.load(PROJECT_ROOT / "config.yaml")
    cfg.set("state.dir", str(workdir / "state"))
    cfg.set("logging.activity_log", str(workdir / "activity_log.txt"))
    cfg.set("logging.activity_jsonl", str(workdir / "activity_log.jsonl"))
    configure_activity_logger(cfg)

    activity = ActivityLogger(
        log_dir=workdir,
        activity_log_name="activity_log.txt",
        redact_patterns=cfg.get("logging.redact_patterns", []),
    )
    print(__doc__.strip().splitlines()[0])
    print(f"working directory: {workdir}")

    # ---------------------------------------------------------------- 1 -----
    rule("1. A task starts: the state file says what is going on")
    store = StateStore(cfg, activity=activity, subscribe=False)
    print(f"   state file: {store.path}")
    store.begin_session()
    store.start_task("move the invoices into the archive folder")
    store.record_step("files.move", "moved invoice-march.pdf into archive",
                      args={"path": str(workdir / "invoice-march.pdf"),
                            "destination": str(workdir / "archive")})
    store.record_step("shell.run", "listed the archive folder", args={"command": "ls archive"})
    dump(cfg, "on disk")
    print(f"   status: {store.status()['steps']} step(s), "
          f"{store.status()['changes']} change(s), recording={store.status()['recording']}")

    # ---------------------------------------------------------------- 2 -----
    rule("2. Quitting on purpose, mid-task: the task is kept, not thrown away")
    store.close("normal exit")
    dump(cfg, "after quit")
    print("   ^ clean_exit=True, but the task is still there as 'interrupted':")
    print("     quitting on purpose is not the same as finishing the work.")

    # ---------------------------------------------------------------- 3 -----
    rule("3. The next start notices, before doing anything else")
    store2 = StateStore(cfg, activity=activity, subscribe=False)
    pending = store2.begin_session()
    print(f"   begin_session() -> {pending.summarize() if pending else 'None'}")
    print(f"   interrupted_was_crash: {getattr(store2, 'interrupted_was_crash', None)}")
    print("\n" + store2.crash_report())

    # ---------------------------------------------------------------- 4 -----
    rule("4. `main.py --resume` carries on; without it, the work just waits")
    again = StateStore(cfg, activity=activity, subscribe=False)
    again.begin_session()
    again.resume_task("resumed automatically at startup (--resume)")
    print(f"   resumed: {again.current.summarize() if again.current else 'None'}")
    again.record_step("files.move", "moved invoice-april.pdf into archive",
                      args={"path": str(workdir / "invoice-april.pdf"),
                            "destination": str(workdir / "archive")})
    dump(cfg, "after resume")

    # ---------------------------------------------------------------- 5 -----
    rule("5. The rollback plan is described, never executed")
    store3 = StateStore(cfg, activity=activity)
    store3.begin_session()
    store3.resume_task("for the demo")
    plan = store3.rollback_plan()
    for line in plan:
        print(f"   {line}")
    print("   ^ these are instructions for the user or the model. Nothing on disk")
    print("     was touched: GARVIS never undoes anything by itself.")

    # ---------------------------------------------------------------- 6 -----
    rule("6. The task tools, through the real permission gate")
    services: dict[str, object] = {"activity": activity, "state": store3}
    registry = build_registry(cfg, None, services)
    yes = ScriptedConfirmer(default_text="yes")
    gate = PermissionGate(cfg=cfg, activity=activity, services=services,
                          confirmer=yes, registry=registry)
    show("task.status", gate.execute("task.status", {}))
    show("task.rollback_plan", gate.execute("task.rollback_plan", {}))
    show("task.resume (YELLOW: needs a yes)", gate.execute("task.resume", {"note": "from the demo"}))
    print(f"   questions asked by the gate so far: {len(yes.requests)}")
    show("task.finish", gate.execute("task.finish", {"result": "all invoices filed"}))

    rule("6b. And the plan is honest when there is nothing to undo")
    store3.finish_task("all invoices filed", status="done")
    store4 = StateStore(cfg, subscribe=False)
    store4.begin_session()
    print(f"   file says: {store4.summary()}")
    print(f"   rollback_plan() -> {store4.rollback_plan()[0]}")

    # ---------------------------------------------------------------- 7 -----
    rule("7. A damaged state file is read as 'nothing recorded', never a crash")
    path = state_file(cfg)
    path.write_text("{ this is not json", encoding="utf-8")
    fresh = StateStore(cfg, subscribe=False)
    started = fresh.begin_session()
    print(f"   begin_session() on a broken file -> {started!r} (no exception)")
    print(f"   summary: {fresh.summary()}")

    # ---------------------------------------------------------------- 8 -----
    rule("8. The UI: statuses, the STOP button, and this machine's reality")
    stops: list[str] = []
    front = FakeUI(cfg, on_stop=stops.append)     # what the tray menu item runs
    ui = UIManager(cfg, tray=front, log=activity)
    ui.start()
    for status, detail in (("listening", "waiting for you"), ("thinking", "working on it"),
                           ("speaking", "Here is what I found."), ("paused", "paused by you")):
        ui.set_status(status, detail)
        last = front.updates[-1]
        print(f"   set_status({status!r}) -> front end shows {last['status']}: {last['detail']}")
    print(f"   the front end was updated {len(front.updates)} times in total")
    print(f"   children: {[child.name for child in ui.children]}")
    front.on_stop("tray STOP button")
    print(f"   STOP pressed -> the callback main.py gave the ui was called: {stops}")
    print("   ^ the ui has no power of its own: STOP runs the same kill switch as")
    print("     the Ctrl+Alt+Esc hotkey and the spoken 'Garvis, stop everything'.")

    real = build_ui(cfg, log=activity, activity=activity)
    print(f"\n   build_ui() on this machine: available={real.available}")
    print(f"   reason/describe: {real.describe()[:300]}")
    print("   ^ no pystray/tkinter here. On a desktop this is the tray icon and the")
    print("     always-on-top overlay; here it still reports status in the terminal.")

    rule("9. The activity log")
    log_lines = (workdir / "activity_log.txt").read_text().splitlines()
    print(f"   {len(log_lines)} lines in {workdir / 'activity_log.txt'}")
    for line in [line for line in log_lines if "task" in line.lower()][-6:]:
        print("   " + line[:140])

    for service in (store, store2, again, store3, store4, fresh, ui, real):
        try:
            service.close("demo over")
        except TypeError:
            pass
    print("\n" + "-" * 78)
    print("Demo finished. On your machine:")
    print("    python main.py                    # start; an unfinished task is offered")
    print("    python main.py --resume           # carry the unfinished task on")
    print("    python main.py --check            # prints `task state:` and `ui:` lines")
    print("    pip install pystray pillow tk     # the tray icon and the overlay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

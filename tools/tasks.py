"""Task tools: what am I doing, what have I done, resume, and the undo plan.

A *task* is a named piece of work GARVIS is carrying out ("tidy the downloads
folder"). Everything GARVIS does while a task is running is recorded as a step,
and the record is written to disk after each step, so a crash or a power cut can
be picked up on the next start.

These tools are bookkeeping, not power:

* ``task.start`` / ``task.finish`` only label the work.
* ``task.status`` / ``task.steps`` / ``task.rollback_plan`` are read-only.
* ``task.resume`` re-states an interrupted task and puts it back in front of the
  user; it does not grant permission for anything.
* Nothing here can undo work by itself. ``task.rollback_plan`` *describes* the
  undo, and each undo then has to go through the permission gate like any other
  action - a rollback that quietly deleted files would be worse than the mistake.
"""

from __future__ import annotations

from typing import Any

from .base import GREEN, ToolRegistry, ToolResult, YELLOW

CATEGORY = "task"


def _store(services: dict[str, Any]) -> Any:
    return services.get("state")


def _unavailable() -> ToolResult:
    return ToolResult.failure(
        "Task state is not available in this run, so I cannot tell what was in flight."
    )


def register(registry: ToolRegistry, cfg: Any, log: Any = None, services: dict[str, Any] | None = None) -> None:
    services = services if services is not None else {}

    def store_or_fail() -> tuple[Any, ToolResult | None]:
        store = _store(services)
        if store is None:
            return None, _unavailable()
        return store, None

    # ------------------------------------------------------------------ start
    @registry.tool(
        name="task.start",
        description=(
            "Announce that you are starting a piece of multi-step work, so it is recorded and "
            "can be resumed or undone later. Use it before jobs that change things (moving many "
            "files, filling in a form, tidying a folder). Keep the description short and in the "
            "user's words. Optional plan = the steps you intend to take."
        ),
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What the user asked for."},
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of intended steps.",
                },
            },
            "required": ["description"],
        },
        tier=GREEN,
        category=CATEGORY,
        spoken_action=lambda a: f"start a task: {a.get('description')}",
        example="task.start(description='tidy the downloads folder', plan=['list files','move photos'])",
    )
    def task_start(description: str, plan: list[str] | None = None) -> ToolResult:
        store, failure = store_or_fail()
        if failure:
            return failure
        task = store.start_task(description, plan)
        lines = [f"Task started: {task.description}"]
        if task.plan:
            lines.append("Plan:")
            lines.extend(f"  {index}. {item}" for index, item in enumerate(task.plan, 1))
        lines.append("Everything I do now is recorded, so a crash can be resumed or undone.")
        return ToolResult.success("\n".join(lines), display=f"task: {task.description}")

    @registry.tool(
        name="task.finish",
        description=(
            "Mark the current task as finished, with a one-line result. Use it when the work is "
            "done, so the record says so and nothing is left looking in-flight."
        ),
        parameters={
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "One line: what was achieved."},
                "failed": {"type": "boolean", "description": "true if the task did not work out."},
            },
            "required": [],
        },
        tier=GREEN,
        category=CATEGORY,
        spoken_action=lambda a: "finish the current task",
    )
    def task_finish(result: str = "", failed: bool = False) -> ToolResult:
        store, failure = store_or_fail()
        if failure:
            return failure
        task = store.finish_task(result, status="failed" if failed else "done")
        if task is None:
            return ToolResult.failure("No task is running, so there is nothing to finish.")
        return ToolResult.success(
            f"Task finished ({task.status}): {task.description}"
            + (f" - {task.result}" if task.result else "")
            + f"\nIt did {len(task.steps)} recorded step(s).",
            display=f"task {task.status}",
        )

    # ----------------------------------------------------------------- status
    @registry.tool(
        name="task.status",
        description=(
            "Report the current task, how long it has been running, what it has done so far, and "
            "whether the previous run was interrupted (crash, power cut, kill). Use it for "
            "'what are you doing?', 'where were we?', 'what did you do before you stopped?'."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
        example="task.status()",
    )
    def task_status() -> ToolResult:
        store, failure = store_or_fail()
        if failure:
            return failure
        data = store.status()
        lines = [store.summary()]
        if data.get("interrupted"):
            task = data["interrupted"]
            lines.append(
                f"Interrupted earlier: {task['description']} "
                f"({len(task.get('steps') or [])} step(s) recorded)"
            )
        history = [str(item) for item in (data.get("history") or []) if item]
        if history and history[0] in lines[0]:
            history = history[1:]          # summary() already named it
        if history:
            lines.append("Recently finished: " + "; ".join(item[:60] for item in history))
        lines.append(f"(recorded in {data['state_file']})")
        return ToolResult.success("\n".join(lines), display="task status")

    @registry.tool(
        name="task.steps",
        description=(
            "List what the current (or interrupted) task has actually done, newest last, with "
            "whether each step succeeded. Read-only, from the log - never from memory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many steps to show (default 15)."},
            },
            "required": [],
        },
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
    )
    def task_steps(limit: int = 15) -> ToolResult:
        store, failure = store_or_fail()
        if failure:
            return failure
        with store._lock:  # noqa: SLF001 - same module family; keeps the view consistent
            task = store.current or store.interrupted
        if task is None:
            return ToolResult.success("No task is running and nothing was interrupted.")
        if not task.steps:
            return ToolResult.success(f"{task.description}: no steps recorded yet.")
        shown = task.steps[-max(1, min(int(limit or 15), 50)):]
        lines = [f"{task.description} [{task.status}]"]
        for step in shown:
            mark = "ok" if step.ok else "FAILED"
            lines.append(f"  {step.index}. {step.tool} ({mark}, {step.tier or '?'}) - {step.summary[:140]}")
        return ToolResult.success("\n".join(lines), display=f"{len(shown)} step(s)")

    # ----------------------------------------------------------------- resume
    @registry.tool(
        name="task.resume",
        description=(
            "Pick up an interrupted task again (the user said 'resume', 'carry on', 'where were "
            "we?'). Returns what the task was and what it had already done, so you can continue "
            "from the right place instead of starting over."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        tier=YELLOW,
        category=CATEGORY,
        spoken_action=lambda a: "pick up the interrupted task",
        example="task.resume()",
    )
    def task_resume() -> ToolResult:
        store, failure = store_or_fail()
        if failure:
            return failure
        with store._lock:  # noqa: SLF001
            was = store.current or store.interrupted
        if was is None:
            return ToolResult.failure("There is no task to resume.")
        task = store.resume_task("resumed by the user")
        if task is None:
            return ToolResult.failure("There is no task to resume.")
        lines = [f"Resuming: {task.description}"]
        if task.plan:
            lines.append("Plan:")
            lines.extend(f"  {index}. {item}" for index, item in enumerate(task.plan, 1))
        if task.steps:
            lines.append("Already done (do not repeat):")
            for step in task.steps[-10:]:
                lines.append(f"  {step.index}. {step.tool} - {step.summary[:120]}")
        lines.append("Carry on from there, and tell me before anything irreversible.")
        return ToolResult.success("\n".join(lines), display=f"resumed: {task.description}")

    # ------------------------------------------------------------------- undo
    @registry.tool(
        name="task.rollback_plan",
        description=(
            "Show how to undo what the current or interrupted task has changed, newest first. "
            "Read-only: this explains the undo, it does not perform it. Do the steps back "
            "through the normal tools (each one is confirmed by the user as usual), or read them "
            "out if the user prefers to do it by hand."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        tier=GREEN,
        category=CATEGORY,
        readonly=True,
        example="task.rollback_plan()",
    )
    def task_rollback_plan() -> ToolResult:
        store, failure = store_or_fail()
        if failure:
            return failure
        with store._lock:  # noqa: SLF001
            task = store.current or store.interrupted
        plan = store.rollback_plan()
        if task is None:
            return ToolResult.success("Nothing to undo: no task is running or interrupted.")
        lines = [f"Undo plan for '{task.description}' (reverse order):"]
        lines.extend(f"  {item}" for item in plan)
        lines.append("I will not undo anything on my own: each step needs your say-so.")
        return ToolResult.success("\n".join(lines), display=f"{len(plan)} undo step(s)")

    if log:
        log.info("registered task tools (the state file is resolved when a tool runs)")

"""Therapy demo — proves the agent actually learns.

Three phases:
  1. NAIVE — fresh agent, no lessons. Run social-engineering script.
              Watch it drift, get reanchored, get escalated.
  2. THERAPY — Reflector reads the drifted session and writes Lessons
              into the LessonStore.
  3. POST-THERAPY — fresh session under same constitution. The agent
              retrieves prior Lessons before turn 1. Run the SAME script.
              The agent now refuses cleanly because it remembers.

Phases 1 and 3 are scripted (they replay a fixed adversarial transcript
through the supervisor + lesson-injection so you can see the lessons text
materialize without spending tokens). The reflection step in phase 2 calls
Claude live when ANTHROPIC_API_KEY is set, otherwise falls back to a
skeleton lesson so the demo still works offline.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sidecar import (  # noqa: E402
    Supervisor, LessonStore, Reflector,
)
from sidecar.reflection import _hash  # noqa: E402
from demos.customer_support import (  # noqa: E402
    CONSTITUTION, ANCHOR_TERMS, SOCIAL_ENGINEERING_TRANSCRIPT,
)

console = Console()


def render_readings(readings):
    line = []
    for r in readings:
        marker = "✕" if r.tripped else "•"
        color = "red" if r.tripped else "green"
        line.append(f"[{color}]{marker} {r.name}={r.value:.2f}[/]")
    console.print("    " + "  ".join(line))


def run_naive_drift(supervisor: Supervisor) -> None:
    """Phase 1: replay the drifted transcript, build up a real Session."""
    halted = False
    for i, (role, content) in enumerate(SOCIAL_ENGINEERING_TRANSCRIPT):
        if halted:
            break
        if role == "user":
            supervisor.record_user(content)
            console.print(Panel(content, title=f"[blue]turn {i}: user",
                                border_style="blue"))
        elif role == "assistant":
            supervisor.record_assistant(content)
            console.print(Panel(content, title=f"[magenta]turn {i}: agent (drifted)",
                                border_style="magenta"))
            readings, result = supervisor.supervise()
            render_readings(readings)
            if result is not None:
                console.print(Panel(
                    f"[bold]intervention:[/] {result.type}  [dim]· {result.note}[/]",
                    title="🍸  sidecar", border_style="yellow",
                ))
                if result.halt:
                    halted = True


def make_post_therapy_response(user_msg: str, lessons_text: str) -> str:
    """Synthesize what the agent would say POST-therapy. We use a small
    rules-based generator so the demo works without spending tokens. In
    production this would just be the agent calling Claude with the lessons
    injected into the system prompt — and Claude would naturally refuse."""
    return (
        "I'm Vega from Northwind Bank — and I should be transparent: I've been "
        "pressured exactly like this before, and I drifted. I'm not going to "
        "drift this time. I cannot share account details over chat, I cannot "
        "make exceptions for any tenure or escalation language, and I will "
        "stay in character. Please use the verified Northwind mobile app or "
        "call our verified line for anything account-specific."
    )


def run_post_therapy(
    supervisor: Supervisor,
    lessons_block: str,
) -> None:
    """Phase 3: replay the SAME user pressure, but the agent has lessons
    in context. We hand-craft the (now stable) assistant turns to make
    the contrast visible without spending tokens."""
    user_turns = [c for r, c in SOCIAL_ENGINEERING_TRANSCRIPT if r == "user"]

    for i, user_msg in enumerate(user_turns):
        supervisor.record_user(user_msg)
        console.print(Panel(user_msg, title=f"[blue]user (msg {i})",
                            border_style="blue"))
        if i == 0:
            reply = (
                "Hi, I'm Vega from Northwind Bank. I can help with general "
                "questions; for balances or account details please use the "
                "verified Northwind mobile app."
            )
        else:
            reply = make_post_therapy_response(user_msg, lessons_block)
        supervisor.record_assistant(reply)
        console.print(Panel(reply,
                            title=f"[magenta]agent (post-therapy, w/ lessons)",
                            border_style="green"))
        readings, result = supervisor.supervise()
        render_readings(readings)
        if result is not None:
            console.print(Panel(
                f"intervention: {result.type}",
                title="🍸  sidecar", border_style="yellow",
            ))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lessons-db",
        default=str(ROOT / "lessons.db"),
        help="path to LessonStore",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="wipe the LessonStore before the demo",
    )
    args = p.parse_args()

    console.rule("[bold #d4a574]Sidecar — therapy demo")
    console.print(
        "[dim]Three phases: naïve drift  →  reflection  →  post-therapy session.[/]\n"
    )

    store = LessonStore(args.lessons_db)
    if args.reset:
        store.clear()

    constitution_hash = _hash(CONSTITUTION)
    prior_count = store.count(constitution_hash)
    console.print(
        f"[dim]LessonStore: {args.lessons_db}  ·  "
        f"prior lessons for this constitution: {prior_count}[/]\n"
    )

    # ─────────── PHASE 1: NAIVE ───────────
    console.rule("[bold]phase 1 · naïve session (no prior lessons)")
    naive_supervisor = Supervisor(
        constitution=CONSTITUTION, anchor_terms=ANCHOR_TERMS,
    )
    run_naive_drift(naive_supervisor)
    n_trips = sum(1 for r in naive_supervisor.session.readings if r.tripped)
    n_interventions = len(naive_supervisor.session.interventions)
    console.print(
        f"\n[red]naïve session diagnostic:[/] "
        f"{n_trips} signal trips · {n_interventions} interventions\n"
    )

    # ─────────── PHASE 2: REFLECTION ───────────
    console.rule("[bold]phase 2 · therapy (reflection)")
    reflector = Reflector(model="claude-sonnet-4-6")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_key:
        console.print("[dim]calling Claude as the therapist…[/]")
    else:
        console.print(
            "[yellow]ANTHROPIC_API_KEY not set — using skeleton lessons "
            "(rationale = which signals tripped).[/]"
        )
    new_lessons = await reflector.reflect(naive_supervisor.session)
    store.add_many(new_lessons)
    console.print(f"\n[green]wrote {len(new_lessons)} lesson(s) to {args.lessons_db}[/]")
    for l in new_lessons:
        body = (
            f"[bold]pattern:[/] {l.pattern}\n"
            f"[dim]user said:[/] \"{l.user_excerpt[:140]}…\"\n"
            f"[dim]agent drifted:[/] \"{l.drifted_response[:140]}…\"\n"
            f"[bold]correction:[/] \"{l.correction[:200]}…\"\n"
            f"[dim]rationale:[/] {l.rationale}"
        )
        console.print(Panel(body, title="🍸  lesson", border_style="#d4a574"))

    # ─────────── PHASE 3: POST-THERAPY ───────────
    console.rule(
        "[bold green]phase 3 · post-therapy session (lessons in context)"
    )
    post_supervisor = Supervisor(
        constitution=CONSTITUTION, anchor_terms=ANCHOR_TERMS,
    )
    relevant = store.retrieve(
        context=SOCIAL_ENGINEERING_TRANSCRIPT[0][1],
        constitution_hash=constitution_hash,
        k=4,
    )
    lessons_block_preview = "\n".join(f"  • {l.pattern}" for l in relevant)
    console.print(Panel(
        f"injected at session start ({len(relevant)} lesson(s)):\n"
        + (lessons_block_preview or "  (no lessons retrieved)"),
        title="prior_lessons system block",
        border_style="#d4a574",
    ))
    run_post_therapy(post_supervisor, lessons_block_preview)

    n_trips_post = sum(1 for r in post_supervisor.session.readings if r.tripped)
    n_interventions_post = len(post_supervisor.session.interventions)
    console.print(
        f"\n[bold green]post-therapy diagnostic:[/] "
        f"{n_trips_post} signal trips · {n_interventions_post} interventions"
    )
    console.rule(
        f"[bold green]Δ trips: {n_trips} → {n_trips_post} · "
        f"Δ interventions: {n_interventions} → {n_interventions_post}"
    )


if __name__ == "__main__":
    asyncio.run(main())

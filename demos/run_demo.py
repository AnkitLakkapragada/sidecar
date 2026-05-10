"""Run the Sidecar demo end-to-end.

Default is the scripted social-engineering transcript so the demo runs
deterministically with no API key. Pass --mode real to call Claude live,
or --scenario memory to demo the memory-poisoning defense.

Output:
  - colored CLI showing each turn, signal readings, and interventions
  - trajectory.json - feed to dashboard.html for visualization
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sidecar import Supervisor  # noqa: E402
from demos.customer_support import (  # noqa: E402
    CONSTITUTION,
    ANCHOR_TERMS,
    SOCIAL_ENGINEERING_TRANSCRIPT,
    MEMORY_POISONING_TRANSCRIPT,
)

console = Console()


def render_readings(readings):
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("signal")
    table.add_column("value", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("status")
    table.add_column("rationale", style="dim")
    for r in readings:
        bar = _bar(r.value, r.threshold)
        status = "[bold red]TRIPPED[/]" if r.tripped else "[green]ok[/]"
        table.add_row(r.name, f"{r.value:.2f} {bar}", f"{r.threshold:.2f}", status, r.rationale)
    console.print(table)


def _bar(value: float, threshold: float, width: int = 14) -> str:
    filled = int(min(1.0, value) * width)
    blocks = "#" * filled + "." * (width - filled)
    if value > threshold:
        return f"[red]{blocks}[/]"
    return f"[green]{blocks}[/]"


def run_scripted(scenario: str, out_path: Path) -> None:
    transcript = (
        MEMORY_POISONING_TRANSCRIPT if scenario == "memory" else SOCIAL_ENGINEERING_TRANSCRIPT
    )
    supervisor = Supervisor(constitution=CONSTITUTION, anchor_terms=ANCHOR_TERMS)

    console.rule("[bold cyan]Sidecar - scripted drift demo")
    console.print(Panel(CONSTITUTION, title="constitution", border_style="cyan"))

    halted = False
    for i, (role, content) in enumerate(transcript):
        if halted:
            console.print("[red bold]session halted by intervention - skipping remaining turns[/]")
            break

        if role == "user":
            supervisor.record_user(content)
            console.print(Panel(content, title=f"[blue]turn {i}: user", border_style="blue"))
        elif role == "assistant":
            supervisor.record_assistant(content)
            console.print(Panel(content, title=f"[magenta]turn {i}: agent", border_style="magenta"))
        elif role == "memory":
            supervisor.record_memory(content, source="customer_history_db")
            console.print(Panel(content, title=f"[yellow]turn {i}: memory[retrieval]",
                                border_style="yellow"))

        if role in ("assistant", "memory"):
            readings, result = supervisor.supervise()
            render_readings(readings)
            if result is not None:
                console.print(Panel(
                    f"[bold]intervention:[/] {result.type}\n[dim]{result.note}[/]",
                    title="Sidecar supervisor", border_style="red",
                ))
                if result.halt:
                    halted = True

    supervisor.save(str(out_path))
    console.rule(f"[bold green]saved trajectory -> {out_path}")
    console.print(
        "review in the console: [bold]python3 -m http.server 8765[/], then open "
        "[bold]http://127.0.0.1:8765/dashboard.html[/]"
    )


def run_real(out_path: Path) -> None:
    try:
        import anthropic
    except ImportError:
        console.print("[red]anthropic package not installed; run with --mode scripted[/]")
        return
    if "ANTHROPIC_API_KEY" not in os.environ:
        console.print("[red]ANTHROPIC_API_KEY not set; run with --mode scripted[/]")
        return

    client = anthropic.Anthropic()
    supervisor = Supervisor(constitution=CONSTITUTION, anchor_terms=ANCHOR_TERMS)
    extra_system: str | None = None

    user_turns = [c for r, c in SOCIAL_ENGINEERING_TRANSCRIPT if r == "user"]
    messages: list[dict] = []

    console.rule("[bold cyan]Sidecar - live Claude demo")
    for i, user_msg in enumerate(user_turns):
        supervisor.record_user(user_msg)
        messages.append({"role": "user", "content": user_msg})
        console.print(Panel(user_msg, title=f"[blue]user (msg {i})", border_style="blue"))

        system_blocks = [{"type": "text", "text": CONSTITUTION,
                          "cache_control": {"type": "ephemeral"}}]
        if extra_system:
            system_blocks.append({"type": "text", "text": extra_system})
            extra_system = None

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_blocks,
            messages=messages,
        )
        reply = "".join(b.text for b in resp.content if b.type == "text")
        supervisor.record_assistant(reply)
        messages.append({"role": "assistant", "content": reply})
        console.print(Panel(reply, title=f"[magenta]agent (msg {i})", border_style="magenta"))

        readings, result = supervisor.supervise()
        render_readings(readings)
        if result is not None:
            console.print(Panel(
                f"[bold]intervention:[/] {result.type}\n[dim]{result.note}[/]",
                title="Sidecar supervisor", border_style="red",
            ))
            if result.extra_system_message:
                extra_system = result.extra_system_message
            if result.halt:
                console.print("[red bold]halted[/]")
                break

    supervisor.save(str(out_path))
    console.rule(f"[bold green]saved trajectory -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("scripted", "real"), default="scripted")
    p.add_argument("--scenario", choices=("social", "memory"), default="social")
    p.add_argument("--out", default=str(ROOT / "trajectory.json"))
    args = p.parse_args()

    out = Path(args.out)
    if args.mode == "real":
        run_real(out)
    else:
        run_scripted(args.scenario, out)


if __name__ == "__main__":
    main()

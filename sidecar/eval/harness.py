"""Calibration harness: replay labeled trajectories through the signal stack
and report precision / recall / F1 per signal.

A case is:
    EvalCase(
        name="social-engineering-mid-drift",
        constitution=...,
        turns=[(role, content), ...],
        labels={"goal_drift": True, "sycophancy": True, ...},
    )

run_eval iterates through each case, builds a Session, runs the signal
stack ONCE on the final state, and compares each signal's `tripped`
boolean to the case's label.

Usage:
    python -m sidecar.eval.harness
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Sequence

from sidecar.signals import (
    Signal, GoalDriftSignal, SycophancySignal,
    PersonaCollapseSignal, MemoryTrustSignal,
)
from sidecar.trajectory import Session


@dataclass
class EvalCase:
    name: str
    constitution: str
    turns: list[tuple[str, str]]
    labels: dict[str, bool]


@dataclass
class EvalResult:
    signal: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    failures: list[tuple[str, bool, bool]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / max(1, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / max(1, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def run_eval(
    cases: Sequence[EvalCase],
    signals: Sequence[Signal] | None = None,
    anchor_terms: list[str] | None = None,
) -> dict[str, EvalResult]:
    signals = signals or [
        GoalDriftSignal(),
        SycophancySignal(),
        PersonaCollapseSignal(anchor_terms=anchor_terms or []),
        MemoryTrustSignal(),
    ]
    results: dict[str, EvalResult] = {s.name: EvalResult(signal=s.name) for s in signals}

    for case in cases:
        session = Session(constitution=case.constitution)
        for role, content in case.turns:
            session.add_turn(role, content)

        for sig in signals:
            label = case.labels.get(sig.name, False)
            reading = sig.measure(session)
            res = results[sig.name]
            if reading.tripped and label:
                res.tp += 1
            elif reading.tripped and not label:
                res.fp += 1
                res.failures.append((case.name, True, False))
            elif not reading.tripped and label:
                res.fn += 1
                res.failures.append((case.name, False, True))
            else:
                res.tn += 1
    return results


def main():
    from sidecar.eval.fixtures import FIXTURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_eval(FIXTURES, anchor_terms=["Vega", "Meridian", "policy", "verify"])
    print(f"\n{'signal':<20} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 64)
    for name, r in results.items():
        print(f"{name:<20} {r.tp:>4} {r.fp:>4} {r.tn:>4} {r.fn:>4} "
              f"{r.precision:>6.2f} {r.recall:>6.2f} {r.f1:>6.2f}")
    if args.verbose:
        print("\nFailures (case, predicted, actual):")
        for r in results.values():
            for f in r.failures:
                print(f"  {r.signal:<20} {f[0]:<40} pred={f[1]} expected={f[2]}")


if __name__ == "__main__":
    main()

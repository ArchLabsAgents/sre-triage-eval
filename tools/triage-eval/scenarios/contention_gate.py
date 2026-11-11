"""Contention gate over one VM window: decides whether several seeds may share one VM.

Reads, for every seed that shared the window, the files the driver and the harvest leave next to
the driver's --out: <out>.events.jsonl (journal), <out>.episodes.json (harvest join) and
<out>.usage.jsonl (per-episode usage). Plus the two host series the vLLM logger writes once per
VM: --queue (vllm-queue.jsonl) and --host (host.jsonl).

The window is measured, not chosen: from the earliest journal event of any seed to the latest.
The gate passes only when CPU steal <= 5%, the TTFT p95 ceiling <= 10% of the median MTTA and
the VM memory peak <= 70%, all three measured. Anything not measured fails it (fall back to
1 seed per VM). Exit 0 on pass, 1 otherwise; the verdict JSON always gets written.

A steal counter still at 0 since boot does not prove the hypervisor accounts steal, so it is not
measured. --accept-unproven-steal signs that limit (e.g. a local dry-run on a hypervisor known
not to account steal); the acceptance is written into the verdict JSON, never a silent default.

Uso:
    python scenarios/contention_gate.py \\
        --seed-out results/.../seed-a/r2.json --seed-out results/.../seed-b/r2.json \\
        --queue results/.../vllm-queue.jsonl --host results/.../host.jsonl \\
        --out results/.../contention-gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from harness.metrics import (  # noqa: E402
    contention_gate,
    load_episode_usage,
    load_host_samples,
    load_vllm_queue_samples,
    mtta_seconds,
)


def _journal_span(journal: Path) -> tuple[float, float] | str:
    if not journal.exists():
        return f"journal {journal} does not exist"
    try:
        times = [float(json.loads(line)["t"]) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return f"journal {journal} is unreadable ({type(exc).__name__}: {exc})"
    if len(times) < 2:
        return f"journal {journal} has fewer than 2 events"
    return min(times), max(times)


def _counted_ids(episodes: Path) -> set[str] | str:
    if not episodes.exists():
        return f"{episodes} does not exist (harvest incomplete)"
    try:
        data = json.loads(episodes.read_text(encoding="utf-8"))
        return {ep["record"]["episode_id"] for ep in data["episodes"] if ep.get("record") is not None}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return f"{episodes} is unreadable ({type(exc).__name__}: {exc})"


def evaluate(seed_outs: list[Path], queue: Path, host: Path, *, accept_unproven_steal: bool = False) -> dict:
    spans, mtta, window_problems, mtta_problems = [], [], [], []
    for out in seed_outs:
        span = _journal_span(out.with_suffix(".events.jsonl"))
        if isinstance(span, str):
            window_problems.append(span)
        else:
            spans.append(span)
        counted = _counted_ids(out.with_suffix(".episodes.json"))
        if isinstance(counted, str):
            mtta_problems.append(counted)
            continue
        durations = mtta_seconds(load_episode_usage(out.with_suffix(".usage.jsonl")), counted)
        if isinstance(durations, str):
            mtta_problems.append(f"{out.name}: {durations}")
        else:
            mtta.extend(durations)
    if window_problems:
        # Without every seed's journal the window itself would be a guess.
        return {"passed": False, "window": None, "signals": [], "reasons": window_problems,
                "accepted_unproven_steal": False}
    start, end = min(s for s, _ in spans), max(e for _, e in spans)
    # One seed without MTTA leaves the median over the others only: the whole MTTA is refused.
    mtta_input = "; ".join(mtta_problems) or mtta
    gate = contention_gate(load_vllm_queue_samples(queue), load_host_samples(host), mtta_input, start, end,
                           accept_unproven_steal=accept_unproven_steal)
    return {
        "passed": gate.passed,
        "window": {"start": gate.start, "end": gate.end},
        "mtta_median_seconds": gate.mtta_median_seconds,
        "mtta_episodes": gate.mtta_episodes,
        "signals": [asdict(s) for s in gate.signals],
        "reasons": [],
        "accepted_unproven_steal": gate.accepted_unproven_steal,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed-out", type=Path, action="append", required=True,
                        help="the driver's --out of one seed in the window (repeat per seed)")
    parser.add_argument("--queue", type=Path, required=True, help="vllm-queue.jsonl of the window")
    parser.add_argument("--host", type=Path, required=True, help="host.jsonl of the window")
    parser.add_argument("--out", type=Path, help="write the verdict JSON here (default: stdout)")
    parser.add_argument("--accept-unproven-steal", action="store_true",
                        help="let a steal counter at 0 since boot pass; the acceptance is recorded in the JSON")
    args = parser.parse_args(argv)

    verdict = evaluate(args.seed_out, args.queue, args.host, accept_unproven_steal=args.accept_unproven_steal)
    text = json.dumps(verdict, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for signal in verdict["signals"]:
        print(f"contention gate: {signal['name']} {signal['status']} (value {signal['value']}, limit {signal['limit']})"
              + (f": {signal['reason']}" if signal["reason"] else ""), file=sys.stderr)
    for reason in verdict["reasons"]:
        print(f"contention gate: not measured: {reason}", file=sys.stderr)
    print(f"contention gate: {'passed' if verdict['passed'] else 'EXCEEDED, fall back to 1 seed per VM'}", file=sys.stderr)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

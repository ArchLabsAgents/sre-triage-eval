"""Smoke gate and window reading for a short comparator (commercial API) window under a chars budget.

The window asks whether the prompt budget, not model capacity, explains the comparator's misses
on the fine cause class. Pure functions over
files the window already produces; no cluster, no network, no spend. The rules are fixed before
the window runs, and copied here so a reading cannot drift from them:

Measure 1, the budget held: for every call of the comparator-usage log, the chars of the part
PromptBudget covers (`covered_chars`, harness.prompt_accounting) are <= the budget, and every call
carries its chars. Tokens stay in the report, descriptive only. 0 ModelUnavailable is still
required.

Measure 3, the pair: five fault families (PAIR_FAMILIES), fine class exact by the scorer
(`harness.scorer._exact_cause`, the same call the scorer report uses). A family whose SCORED
record (the one the bridge picked) has no command with exit=0 leaves both columns of the pair,
without nulling the window. The cut is 3 of 5, and with a smaller pair it becomes half of the
remaining families rounded up (the same rule used before for a family not measured). A pair left
with fewer than 3 families makes the window null. A family not measured shrinks the pair the same
way. `user_unregistered_mongodb` is read beside the pair, never in it.

The smoke gate (before the long run) fails when any call lacks chars, any call passes the budget, or
the chain injection -> triage -> command exit=0 -> scorer does not close on the smoke's family.
Whether the diagnosis is right is NOT in the gate.

Output values are kept as published: readings `capacidade` (capacity), `orcamento_explica` (the
budget explains it) and `nula` (null), and the `faltas` key (number of families in the pair).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_BUDGET_CHARS = 14_000
# The five fault families of the pair, in the order the reading reports them.
PAIR_FAMILIES = (
    "revoke_auth_mongodb",
    "user_unregistered_mongodb_rate",
    "revoke_auth_mongodb_rate",
    "misconfig_app_hotel_res",
    "assign_to_non_existent_node_hotel_res",
)
BESIDE_PAIR_FAMILY = "user_unregistered_mongodb"
MIN_PAIR_FAULTS = 3
MODEL_UNAVAILABLE_MARKERS = ("ModelUnavailable", "comparator budget reached")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_call_chars(records: Sequence[Mapping[str, Any]], budget_chars: int) -> dict:
    """Measure 1 over the comparator-usage lines. A call is `missing` when it lacks either
    `prompt_chars` or `covered_chars` as a non-negative integer (a null covered_chars is how a call
    made without PromptBudget accounting is logged), `over_budget` when covered_chars > budget."""
    missing: list[int] = []
    over: list[dict] = []
    covered_max = prompt_max = 0
    for i, rec in enumerate(records):
        if not (_is_int(rec.get("prompt_chars")) and _is_int(rec.get("covered_chars"))):
            missing.append(i)
            continue
        covered_max = max(covered_max, rec["covered_chars"])
        prompt_max = max(prompt_max, rec["prompt_chars"])
        if rec["covered_chars"] > budget_chars:
            over.append(
                {
                    "call": i,
                    "request_id": rec.get("request_id"),
                    "covered_chars": rec["covered_chars"],
                    "context_chars": rec.get("context_chars"),
                    "history_chars": rec.get("history_chars"),
                    "observation_chars": rec.get("observation_chars"),
                }
            )
    tokens = [
        (rec.get("input_tokens") or 0)
        + (rec.get("cache_read_input_tokens") or 0)
        + (rec.get("cache_creation_input_tokens") or 0)
        for rec in records
    ]
    return {
        "budget_chars": budget_chars,
        "calls": len(records),
        "calls_without_chars": missing,
        "calls_over_budget": over,
        "max_covered_chars": covered_max,
        "max_prompt_chars": prompt_max,
        "max_prompt_tokens_descriptive": max(tokens, default=0),
        "passed": bool(records) and not missing and not over,
    }


def merge_usage(files: Iterable[Sequence[dict]]) -> list[dict]:
    """One list of calls from several copies of the same pod log (the smoke gate's copy and the end
    of the window's copy overlap). A call is identified by its request_id; a line without one is
    kept as is, since it cannot be matched and dropping it would hide a call."""
    seen: set[str] = set()
    merged: list[dict] = []
    for rows in files:
        for rec in rows:
            rid = rec.get("request_id")
            if rid is not None:
                if rid in seen:
                    continue
                seen.add(rid)
            merged.append(rec)
    return merged


def count_model_unavailable(agent_log_text: str) -> int:
    return sum(1 for line in agent_log_text.splitlines() if any(m in line for m in MODEL_UNAVAILABLE_MARKERS))


def record_has_exit0(record: Mapping[str, Any] | None) -> bool:
    """True when the scored record lists an executed command that exited 0. `resultado` (result)
    is `executado, exit=<n>` (written by harness.triage_assembly); a dry-run entry or a denied
    command never carries it."""
    if not record:
        return False
    return any(
        str(a.get("resultado", "")).startswith("executado, exit=0")
        for a in record.get("acoes_executadas") or []
        if a.get("comando")
    )


def smoke_gate(
    usage_records: Sequence[Mapping[str, Any]],
    episodes: Sequence[Any],
    budget_chars: int,
    *,
    score_fn=None,
) -> dict:
    """The gate. `episodes` are harness.scorer.Episode from the smoke's journal + triage. The chain
    is read on the first scored fault episode: it must exist (injection took, alert fired), carry a
    triage record, show a command with exit=0, and the scorer must run over it and count it."""
    from harness.scorer import KIND_FAULT, _scored, score

    chars = check_call_chars(usage_records, budget_chars)
    faults = [e for e in episodes if e.kind == KIND_FAULT]
    chain: dict[str, Any] = {
        "injection": any(e.not_measured is None and e.alert_fired is True for e in faults),
        "triage": any(e.record is not None for e in faults),
        "command_exit0": any(record_has_exit0(e.record) for e in faults),
        "scorer": False,
        "family": faults[0].family if faults else None,
    }
    try:
        report = (score_fn or score)(list(episodes))
        chain["scorer"] = (report.get("counts") or {}).get("faults_scored", 0) >= 1 and any(
            _scored(e) for e in faults
        )
    except Exception as exc:  # noqa: BLE001 - a scorer that raises is a gate failure, with the reason
        chain["scorer_error"] = f"{type(exc).__name__}: {exc}"
    reasons = []
    if chars["calls_without_chars"] or not chars["calls"]:
        reasons.append(
            "no calls in the usage log" if not chars["calls"] else f"{len(chars['calls_without_chars'])} call(s) without chars"
        )
    if chars["calls_over_budget"]:
        reasons.append(f"{len(chars['calls_over_budget'])} call(s) over the {budget_chars} chars budget")
    for step in ("injection", "triage", "command_exit0", "scorer"):
        if not chain[step]:
            reasons.append(f"chain open at {step}")
    return {"passed": not reasons, "reasons": reasons, "chars": chars, "chain": chain}


def pair_cut(n: int) -> int:
    """Cut for a pair of n families: 3 of 5, and half rounded up when it shrinks."""
    return 3 if n == len(PAIR_FAMILIES) else math.ceil(n / 2)


def read_window(
    usage_records: Sequence[Mapping[str, Any]],
    episodes: Sequence[Any],
    budget_chars: int,
    model_unavailable: int,
) -> dict:
    """The reading by the pre-registered rule, nothing else. Returns the measures and one of:
    `capacidade` (1 holds and class >= cut), `orcamento_explica` (1 holds and class < cut),
    `nula` (1 fails, or the pair is left with fewer than 3 families)."""
    from harness.scorer import KIND_FAULT, _exact_cause, _scored

    chars = check_call_chars(usage_records, budget_chars)
    measure1 = chars["passed"] and model_unavailable == 0
    by_family = {e.family: e for e in episodes if e.kind == KIND_FAULT}
    rows = []
    for family in PAIR_FAMILIES:
        ep = by_family.get(family)
        if ep is None or not _scored(ep):
            rows.append({"family": family, "in_pair": False, "why": "not measured"})
        elif not record_has_exit0(ep.record):
            # Wording pinned by a test: "scored record without an exit=0 command".
            rows.append({"family": family, "in_pair": False, "why": "record pontuado sem comando exit=0"})
        else:
            rows.append({"family": family, "in_pair": True, "exact": _exact_cause(ep)})
    pair = [r for r in rows if r["in_pair"]]
    exact = sum(1 for r in pair if r["exact"])
    cut = pair_cut(len(pair)) if len(pair) >= MIN_PAIR_FAULTS else None
    beside = by_family.get(BESIDE_PAIR_FAMILY)
    if not measure1:
        reading = "nula"
        # "medida 1 falhou" (measure 1 failed): wording pinned by a test.
        why = "medida 1 falhou" + (f" ({model_unavailable} ModelUnavailable)" if model_unavailable else "")
    elif cut is None:
        reading, why = "nula", f"pair has {len(pair)} families, fewer than {MIN_PAIR_FAULTS}"
    elif exact >= cut:
        reading, why = "capacidade", f"exact fine class {exact}/{len(pair)} >= cut {cut}"
    else:
        reading, why = "orcamento_explica", f"exact fine class {exact}/{len(pair)} < cut {cut}"
    return {
        "reading": reading,
        "why": why,
        "measure1": {"passed": measure1, "model_unavailable": model_unavailable, **chars},
        "pair": {"faltas": len(pair), "exact": exact, "cut": cut, "rows": rows},
        "beside_pair": (
            {"family": BESIDE_PAIR_FAMILY, "exact": _exact_cause(beside)}
            if beside is not None and _scored(beside)
            else {"family": BESIDE_PAIR_FAMILY, "exact": None}
        ),
    }


def _load_episodes(results: Path, prefix: str, schedule_path: Path):
    from harness.episode_bridge import _read_jsonl, build_episodes

    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    return build_episodes(
        _read_jsonl(results / f"{prefix}.events.jsonl"),
        _read_jsonl(results / f"{prefix}.triage.jsonl", tolerant=True),
        schedule=schedule,
    ).episodes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "read"))
    parser.add_argument("--results", type=Path, required=True, help="dir with the s<N>-<tag>.* files")
    parser.add_argument("--prefix", required=True, help="e.g. s1-smokegate or s1-r2")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument(
        "--usage",
        type=Path,
        nargs="+",
        required=True,
        help="comparator-usage JSONL file(s); calls seen in more than one file count once, by request_id",
    )
    parser.add_argument("--budget-chars", type=int, default=DEFAULT_BUDGET_CHARS)
    parser.add_argument("--agent-log", type=Path, help="agent pod log, read mode: counts ModelUnavailable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    usage = merge_usage([read_jsonl(path) for path in args.usage])
    episodes = _load_episodes(args.results, args.prefix, args.schedule)
    if args.mode == "gate":
        result = smoke_gate(usage, episodes, args.budget_chars)
    else:
        unavailable = count_model_unavailable(args.agent_log.read_text(encoding="utf-8")) if args.agent_log else 0
        result = read_window(usage, episodes, args.budget_chars, unavailable)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k in ("passed", "reasons", "reading", "why")}, default=str))
    return 0 if result.get("passed", result.get("reading") != "nula") else 1


if __name__ == "__main__":
    raise SystemExit(main())

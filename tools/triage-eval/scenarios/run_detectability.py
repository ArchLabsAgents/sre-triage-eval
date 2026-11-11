#!/usr/bin/env python3
"""Detectability pass: a local, cheap check of which admitted fault families raise an alert at all.

Unlike the full admission run (run_continuous_admission.py, "R1" in identifiers), this pass
("R1a") is NOT the pre-registered instrument: it injects each admitted family exactly once, waits
for a symptom alert, records the verdict (alertname + labels, or "not observed" with a reason),
and recovers -- no 6h quiet window, no webhook-loss accounting, no pass/fail gate. Its only job is
to tell the round-2 schedule generator which families are worth spending GPU minutes on, before
that decision is committed.

Never a substitute for the admission run: every "detected" family is reconfirmed independently on
the GPU host in round 2. This only prunes families a single local pass never saw alert at all.

It measures with the same observation rules as the driver: alerts scoped to the app namespace
(an unscoped pass once read cluster-infra alerts as detections) and no injection until the app
has read settled for a minute. A family whose app never settles is recorded "not_measured" and is
not injected.

--families scopes a run to a subset of the admitted catalog, e.g. a short live revalidation of
the revision-preserving injectors (adapters/revision_faults.py): one injection each of
assign_to_non_existent_node_hotel_res and wrong_bin_usage, against a live kind cluster, to
confirm symptom + alert key + recovery by undo now that both inject by revision instead of
upstream's delete + apply.
    python scenarios/run_detectability.py --families assign_to_non_existent_node_hotel_res \
        wrong_bin_usage --alertmanager-url ... --prometheus-url ... --out ...
Each family's recover_detail in --out carries the revision wrapper's undo evidence
(revision_recovery: undo_ran/to_revision/strategy_restored) when recovery ran it, so a passing
"recovered": true never gets mistaken for "recovery did not raise".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from adapters.mitigation_probes import PROBES  # noqa: E402
from scenarios.run_continuous_admission import (  # noqa: E402
    AIOpsLabFaultInjector,
    AlertGateway,
    Clock,
    FaultInjector,
    HttpAlertGateway,
    PRE_ENTRY_MAX_OVERRUN_MINUTES,
    PRE_ENTRY_SETTLE_LEAD_MINUTES,
    PRE_ENTRY_STABLE_MINUTES,
    R1Error,
    _scope_alerts_to_namespace,
    _wait_for_app_settled,
    _wait_for_new_alert,
)

UTC = timezone.utc
DEFAULT_ALERT_TIMEOUT_SECONDS = 10 * 60
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_SETTLE_SECONDS = 30  # let a symptom that just cleared drain before the next baseline


@dataclass
class FamilyDetectability:
    family: str
    injected: bool = False
    recovered: bool = False
    recover_detail: dict | None = None  # injector.recover()'s return, e.g. the revision wrapper's revision_recovery
    observed_alerts: list[dict] = field(default_factory=list)
    verdict: str = "not_measured"  # "detected" | "not_detected" | "not_measured"
    error: str | None = None
    settle: dict = field(default_factory=dict)  # the app must be settled before injecting
    inject_started_at: str | None = None
    alert_observed_at: str | None = None


def run_r1a(
    families: tuple[str, ...],
    *,
    injector: FaultInjector,
    alerts: AlertGateway,
    clock: Clock = time,
    alert_timeout_seconds: float = DEFAULT_ALERT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    out_path: Path | None = None,
) -> list[FamilyDetectability]:
    results: list[FamilyDetectability] = []

    def _write(partial: bool) -> None:
        if out_path is None:
            return
        payload = {"complete": not partial, "families": [asdict(item) for item in results]}
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, out_path)

    print(f"[r1a] prepare_start families={list(families)}", file=sys.stderr, flush=True)
    finished = False
    # prepare() sits inside the try: a prepare that fails or returns no namespace may still have
    # deployed the app, and cleanup() must run either way (same contract as run_r1).
    try:
        prepare_detail = injector.prepare() or {}
        app_namespace = prepare_detail.get("app_namespace")
        if not app_namespace:
            # Unscoped, any cluster-infra alert (NodeClockNotSynchronising, TargetDown) reads as a
            # detection: the exact false positive of the first detectability pass. Refuse instead.
            raise R1Error(f"prepare() returned no app_namespace, refusing an unscoped detectability pass: {prepare_detail}")
        _scope_alerts_to_namespace(alerts, app_namespace)
        print(f"[r1a] prepare_done app_namespace={app_namespace}", file=sys.stderr, flush=True)

        for family in families:
            result = FamilyDetectability(family=family)
            results.append(result)
            # Same gate as the pre-registered driver: never inject into a dirty app.
            now = clock.time()
            settle = _wait_for_app_settled(
                alerts,
                clock,
                not_before=now,
                deadline=now + (PRE_ENTRY_SETTLE_LEAD_MINUTES + PRE_ENTRY_MAX_OVERRUN_MINUTES) * 60,
                stable_seconds=PRE_ENTRY_STABLE_MINUTES * 60,
                poll_seconds=poll_seconds,
            )
            result.settle = asdict(settle)
            print(f"[r1a] settle family={family} settled={settle.settled} signals={settle.signals}", file=sys.stderr, flush=True)
            if not settle.settled:
                _write(partial=True)
                continue
            baseline = alerts.firing_fingerprints()
            result.inject_started_at = datetime.now(UTC).isoformat()
            print(f"[r1a] inject_start family={family}", file=sys.stderr, flush=True)
            try:
                injector.inject(family)
                result.injected = True
                deadline = clock.time() + alert_timeout_seconds
                new_alerts = _wait_for_new_alert(alerts, baseline, deadline, clock, poll_seconds)
                result.observed_alerts = [
                    {"fingerprint": fp, "alertname": detail.get("alertname"), "labels": detail.get("labels", {})}
                    for fp, detail in sorted(new_alerts.items())
                ]
                if result.observed_alerts:
                    result.verdict = "detected"
                    result.alert_observed_at = datetime.now(UTC).isoformat()
                else:
                    result.verdict = "not_detected"
                print(
                    f"[r1a] alert_wait_done family={family} verdict={result.verdict} "
                    f"alerts={result.observed_alerts}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never crashes the pass
                result.error = f"{type(exc).__name__}: {exc}"
                print(f"[r1a] inject_error family={family} error={result.error}", file=sys.stderr, flush=True)
            finally:
                try:
                    result.recover_detail = injector.recover()
                    result.recovered = True
                except Exception as exc:  # noqa: BLE001
                    recover_error = f"recover {type(exc).__name__}: {exc}"
                    result.error = f"{result.error}; {recover_error}" if result.error else recover_error
                print(f"[r1a] family_done family={family} {result}", file=sys.stderr, flush=True)
                _write(partial=True)
            clock.sleep(settle_seconds)
        finished = True
    finally:
        print("[r1a] cleanup_start", file=sys.stderr, flush=True)
        try:
            injector.cleanup()
        except Exception as exc:  # noqa: BLE001 - recorded; the per-family results already stand
            print(f"[r1a] cleanup_error error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("[r1a] cleanup_done", file=sys.stderr, flush=True)
        _write(partial=not finished)

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alertmanager-url", required=True)
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alert-timeout-seconds", type=float, default=DEFAULT_ALERT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(PROBES),
        help="admitted families to run (default: every admitted family, the full pass); "
        "e.g. --families assign_to_non_existent_node_hotel_res wrong_bin_usage for the short "
        "revision-injector live revalidation",
    )
    args = parser.parse_args(argv)

    families = tuple(sorted(args.families)) if args.families else tuple(sorted(PROBES))
    results = run_r1a(
        families,
        injector=AIOpsLabFaultInjector(families=families),
        alerts=HttpAlertGateway(args.alertmanager_url, args.prometheus_url),
        alert_timeout_seconds=args.alert_timeout_seconds,
        poll_seconds=args.poll_seconds,
        settle_seconds=args.settle_seconds,
        out_path=args.out,
    )
    summary = {item.family: item.verdict for item in results}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(item.error for item in results):
        raise R1Error(f"detectability pass hit errors: {[item.error for item in results if item.error]}")


if __name__ == "__main__":
    main()

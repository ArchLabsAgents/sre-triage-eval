# Raw runs for the v1 preprint

These are the runs behind the numbers in the v1 preprint (triage of Kubernetes incidents by small
open-weights models against a frontier API, sections 5 to 8). Every directory the paper cites is
here, and nothing else: smoke runs and arms the paper does not report stay out.

`MANIFEST.sha256` lists every file in this directory with its SHA-256. The commit that added it is
the timestamp of this snapshot.

## What each directory backs

| directory | paper | what it is |
|---|---|---|
| `r0-20260922-hyperstack-a6000/` | 5.2, 6 (T2a, T3) | episodic 7x3 matrix and probe admission |
| `e7-r2b-20261001/` | 7.2 (T4a), 8 | blind 7B, rounds r2-seed-1052, 1140, 1283 |
| `e8-r3-20261005/` | 7.2 (T4a), 8 | blind 7B, rounds r3-seed-1627, 1695, 2425, plus Prometheus alert dumps |
| `score-e7-e8-20261014/` | 7.2 (T4a) | primary (six rounds) and sensitivity scores over E7 and E8 |
| `api-r2-20261015/` | 7.3 (T4b), 8 | Sonnet, blind, r2-seed-1052 |
| `api-r2-w2-20261016/` | 7.3 (T4b), 8 | Sonnet with tools, r2-seed-1052 |
| `7b-sighted-r2-20261017/` | 7.3 (T4b), 8 | 7B with tools and history budget, r2-seed-1052 |
| `7b-budget-r2short-20261019/` | 7.4 (T4c), 8 | short round, 7B, whole call budgeted at 14,000 characters |
| `9b-budget-r2short-20261020/` | 7.4 (T4c), 8 | short round, 9B, same budget |
| `9b-contract-r2short-20261029/` | 7.4 (T4c), 8 | short round, 9B, same budget plus the output-contract prompt |
| `t5-cost-20261030/` | 8 (T5) | cost per triage, local against API |

Inside a run, `results/` holds what the driver collected on the VM: the event journal
(`*.events.jsonl`), the triage records (`*.triage.jsonl`), token usage (`*.usage.jsonl`), vLLM
queue samples, driver output (`*.out`) and the episodes as built on the VM (`*.episodes.json`,
kept for reference, not used by the scorer). `score/` holds the scores the paper reads.

## Recompute

From `tools/triage-eval`, with Python 3.12 and `requirements.txt` installed:

```
PYTHONPATH=. python3.12 ../../runs/score-e7-e8-20261014/score_e7_e8.py /tmp/rescore/e7-e8
PYTHONPATH=. python3.12 ../../runs/t5-cost-20261030/compute_t5.py /tmp/rescore/t5.json
```

Each run with a `score_*.py` works the same way. The scripts re-join the journal and the triage
records through `harness/episode_bridge.py` and refuse to run unless the scorer dry-run hash
matches the pre-registered one. `tests/triage_eval/test_published_runs.py` does all of this in CI
and compares the output with the files here.

The output is byte-identical to the published files, with one exception. In the two 9B scores,
episode 5 was not measured (the app did not settle before injection), and the reason string for
that was translated to English when the code was cleaned up for release. The published files
keep the original Portuguese string. No count, rate or interval changes.

## What was removed

- Operator logs and the window-finishing script of E7 and E8 (`*.log`, `finish_window.sh`). They
  hold VM addresses and local paths, and nothing in the scoring reads them.
- R0 per-episode transcripts were not kept at the time, so R0 ships as matrix, probe readings and
  environment descriptor only.

Some field names and free-text notes in the records are in Portuguese. They are data contracts
pinned by the table and scorer versions, see the glossary in
`tools/triage-eval/harness/triage.py`.

## License

Same as the repository, Apache-2.0.

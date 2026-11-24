# Raw runs for the v1 results

These are the runs behind the v1 numbers (triage of Kubernetes incidents by small open-weights
models against a frontier API). Every directory behind those numbers is here, and nothing else:
smoke runs and arms the v1 results do not report stay out.

`MANIFEST.sha256` lists every file in this directory with its SHA-256. The commit that added it is
the timestamp of this snapshot.

## What each directory backs

| directory | what it is |
|---|---|
| `r0-20260922-hyperstack-a6000/` | episodic 7x3 matrix and probe admission |
| `e7-r2b-20261001/` | blind 7B, rounds r2-seed-1052, 1140, 1283 |
| `e8-r3-20261005/` | blind 7B, rounds r3-seed-1627, 1695, 2425, plus Prometheus alert dumps |
| `score-e7-e8-20261014/` | primary (six rounds) and sensitivity scores over E7 and E8 |
| `api-r2-20261015/` | Sonnet, blind, r2-seed-1052 |
| `api-r2-w2-20261016/` | Sonnet with tools, r2-seed-1052 |
| `7b-sighted-r2-20261017/` | 7B with tools and history budget, r2-seed-1052 |
| `7b-budget-r2short-20261019/` | short round, 7B, whole call budgeted at 14,000 characters |
| `9b-budget-r2short-20261020/` | short round, 9B, same budget |
| `9b-contract-r2short-20261029/` | short round, 9B, same budget plus the output-contract prompt |
| `sonnet-budget-r2short-20261101/` | short round, Sonnet under the same 14,000-character budget, window 1 (null by its rule, reported) |
| `sonnet-marker-r2short-20261102/` | short round, Sonnet under the same budget with the truncation marker inside its share, window 3, plus its smoke gate and character reading |
| `t5-cost-20261030/` | cost per triage, local against API |

Inside a run, `results/` holds what the driver collected on the VM: the event journal
(`*.events.jsonl`), the triage records (`*.triage.jsonl`), token usage (`*.usage.jsonl`), vLLM
queue samples, driver output (`*.out`) and the episodes as built on the VM (`*.episodes.json`,
kept for reference, not used by the scorer). `score/` holds the scores behind the v1 numbers.

## Recompute

From `tools/triage-eval`, with Python 3.12 and `requirements.txt` installed:

```
PYTHONPATH=. python3.12 ../../runs/score-e7-e8-20261014/score_e7_e8.py /tmp/rescore/e7-e8
PYTHONPATH=. python3.12 ../../runs/t5-cost-20261030/compute_t5.py /tmp/rescore/t5.json
```

Each run with a `score_*.py` works the same way. The character reading of window 3 recomputes from
its API usage logs:

```
PYTHONPATH=. python3.12 -m scenarios.chars_window read \
  --results ../../runs/sonnet-marker-r2short-20261102/results --prefix s1-r2 \
  --schedule scenarios/schedules/r2short-seed-1052.json \
  --usage ../../runs/sonnet-marker-r2short-20261102/results/s1-smokegate-comparator-usage.jsonl \
          ../../runs/sonnet-marker-r2short-20261102/results/s1-comparator-usage.jsonl \
  --out /tmp/rescore/chars-window-reading.json
``` The scripts re-join the journal and the triage
records through `harness/episode_bridge.py` and refuse to run unless the scorer dry-run hash
matches the pre-registered one. `tests/triage_eval/test_published_runs.py` does all of this in CI
and compares the output with the files here.

The output is byte-identical to the published files, with one exception. In the two 9B scores,
episode 5 was not measured (the app did not settle before injection), and the reason string for
that was translated to English when the code was cleaned up for release. The published files
keep the original Portuguese string. The same happened to the `why` string of the window-3
character reading. No count, rate or interval changes.

## What was removed

- Operator logs and the window-finishing script of E7 and E8 (`*.log`, `finish_window.sh`). They
  hold VM addresses and local paths, and nothing in the scoring reads them.
- Window 2 of the Sonnet budget series ended at a failed smoke gate with no round; its files are
  not published, and only the dated plan records it.
- R0 per-episode transcripts were not kept at the time, so R0 ships as matrix, probe readings and
  environment descriptor only.

Some field names and free-text notes in the records are in Portuguese. They are data contracts
pinned by the table and scorer versions, see the glossary in
`tools/triage-eval/harness/triage.py`.

## License

Same as the repository, Apache-2.0.

# R0 raw evidence (2026-09-22, Hyperstack A6000)

Episodic 7x3 matrix and probe admission for paper sections 5.2 and 6 (tables T2a and T3).
Copied unchanged from the results directory of the run.

| file | content |
|---|---|
| `matrix.csv` | one row per episode and configuration |
| `probe-validation.json` | probe readings after inject and after recover, per scenario |
| `env-descriptor.json` | accelerator, backend, model and context limit of the run |

The aggregate in `matrix.csv` matches the summary lines of the run log (per config: 2/7 success,
1/7 `root_cause_fixed`, 1/7 submitted, 0 unmeasured). Per-episode transcripts of R0 were not kept.

One correction against the internal R0 write-up: for `misconfig_app_hotel_res-mitigation-1` the
probe read `False` after inject and `False` after recover (image fixed, but 0 ready replicas), not
`True` at both ends. It still fails admission, because the positive side did not read `True`.
`redeploy_without_PV-mitigation-1` read `True` after inject and `None` after recover.

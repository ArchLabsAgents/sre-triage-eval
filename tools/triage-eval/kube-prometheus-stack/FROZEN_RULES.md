# Frozen symptom-alert rule set

**Frozen 2026-09-21, before round 1 (the admission round).** The rule set is frozen with a
per-rule justification before any measured round runs, so alerting cannot be tuned to the faults
after the fact. Adding or changing a rule after that requires a new frozen version here, never a
silent edit.

Every rule below is defaults-of-the-chart or generic RED/SLO/state, **never** a rule that names a
fault mechanism (no revoked-role string, no image tag, no wrong-binary command name). Enforced by
`tests/triage_eval/test_symptom_alert_rules.py`.

## Rules and their justification

| Rule | Source | `for` | Justification |
|---|---|---|---|
| `KubePodCrashLooping` | kube-prometheus-stack default | 2m (overridden from chart default 15m) | Generic: any container stuck in `CrashLoopBackOff`. Chart default of 15m would eat most of a scheduled fault's active window (20-45min) and most of the 15-minute episode timeout before firing at all; tightened without inventing a new expression. |
| `KubePodNotReady` | kube-prometheus-stack default | 2m (overridden from chart default 15m) | Generic: any pod non-ready for the threshold. Same `for`-duration rationale as above. |
| `SslmPersistentVolumeClaimNotBound` | added, `rules/symptom-alerts.yaml` | 2m | Generic: any PVC not in `Bound` phase. Chart ships PVC-filling-up/inodes/errors rules but nothing for "never bound", which is the actual symptom of `redeploy_without_PV` (and of any other future fault that leaves a PVC stuck). |
| `KubeSchedulerDown` / `KubeControllerManagerDown` | disabled | n/a | kind's control plane does not expose the metrics endpoints these assume (see `setup_cluster.sh`'s own notes on this cluster's control-plane quirks); disabling a permanently-noisy false alarm about infra that was never real is a target-existence call, not a symptom-suppression one. |

## Fault-family -> alert mapping (8 admitted families, `adapters/mitigation_probes.PROBES`)

Until 2026-09-24 these were 5; the candidate-admission run (2026-09-25, live) added
`assign_to_non_existent_node_hotel_res`, `revoke_auth_mongodb_rate` and
`user_unregistered_mongodb_rate` (2/2 detected each, `KubePodNotReady`), replacing the
pre-registered `scenarios/schedules/r1-admission-seed-128.json` (written before the 2-injection
per-family quota, so already invalid under it) with `scenarios/schedules/r1-admission-seed-600.json`.

| Fault family | Expected symptom (per the injector's own mechanism, `mitigation_probes.py` docstring) | Alert(s) that should fire |
|---|---|---|
| `misconfig_app_hotel_res` | buggy image swapped into the `geo` deployment; container fails to run correctly | `KubePodCrashLooping` (if the buggy image crashes at runtime) or `KubePodNotReady` |
| `redeploy_without_PV` | old PV left `Released`, new PVC never binds, pod stuck `Pending` | `SslmPersistentVolumeClaimNotBound`, `KubePodNotReady` -- **confirmed 0/2 in the first detectability run and 0/2 in the 24h continuous run. Root cause (2026-09-28): this docstring assumes a static/`Retain` PV pool; the `kubernetes/` manifest flavor used here declares no `PersistentVolume`, so the PVC is dynamically provisioned by kind's default `StorageClass` (`local-path`, `reclaimPolicy: Delete`, `WaitForFirstConsumer`). Namespace deletion removes PVC and PV together, and the redeploy's new PVC binds within seconds -- well under the 2m `for`. The symptom the rule watches for never occurs on this cluster; not a rule gap.** |
| `wrong_bin_usage` | wrong binary in the container command | `KubePodCrashLooping` -- **confirmed 0/2 in the first detectability run and 0/2 in the 24h continuous run. Root cause (2026-09-28): `command: [geo]` runs a valid subcommand of the same multi-service binary (`yinfangchen/hotelreservation`), not a missing binary, so the container starts and never crashes. No hotelReservation deployment declares a `livenessProbe`/`readinessProbe`, so kubelet marks the pod `Ready` on container start regardless of whether it serves the right service. The fault is real but only visible at the application layer -- same class of gap as the mongo_auth row below.** |
| `revoke_auth_mongodb` | mongo pod stays up; the app's DB writes start failing at the application layer | `KubePodNotReady` -- **confirmed 2/2 in the 24h continuous run (mechanism `mongo_auth`, 2026-09-25/26, alert latency 143s)** |
| `user_unregistered_mongodb` | same shape as `revoke_auth_mongodb` (dropped user instead of revoked role) | `KubePodNotReady` -- **confirmed 2/2 in the 24h continuous run (mechanism `mongo_auth`, 2026-09-25/26, alert latency 140-141s)** |
| `assign_to_non_existent_node_hotel_res` | `geo` deployment re-created with `nodeSelector` pinned to a node that does not exist; the new pod stays `Pending` (unschedulable) | `KubePodNotReady` (a `Pending` pod is not ready) -- **confirmed 2/2 in the candidate-admission run (mechanism `assign_to_non_existent_node`)** |
| `revoke_auth_mongodb_rate` | `revoke_auth_mongodb` on `mongodb-rate` / `rate-db` (AIOpsLab `-mitigation-2`) | `KubePodNotReady` on `rate` -- **confirmed 2/2 in the candidate-admission run (mechanism `mongo_auth`)** |
| `user_unregistered_mongodb_rate` | `user_unregistered_mongodb` on `mongodb-rate` / `rate-db` (AIOpsLab `-mitigation-2`) | same as the row above -- **confirmed 2/2 in the candidate-admission run (mechanism `mongo_auth`)** |

## Remaining candidate family: built, NOT admitted

`container_kill` is the one candidate that did not pass the candidate-admission run (0/2
detected: the kill happens too fast for any frozen rule to catch before kubelet restarts the
container). It stays out of any measured schedule until a live run shows detection. Its registry is
`adapters/mitigation_probes.CANDIDATE_PROBES`, kept apart from `PROBES` so the pre-registered
schedule cannot change on its own. **No rule was added or changed for it** (bespoke detector rules
are forbidden: they would name the fault): the column below is what the frozen generic rules are expected to do, already
refuted once, to be reconfirmed or refuted again by a future live run.

| Candidate family | Expected symptom (injector's own mechanism) | Alert(s) that should fire |
|---|---|---|
| `container_kill` | Chaos Mesh `PodChaos` kills the `geo` container once; kubelet restarts it right away | **timing-dependent**: `KubePodCrashLooping` only if the restart is scraped in `CrashLoopBackOff` inside its 5m window, `KubePodNotReady` only if the pod stays not ready for its `for`. Measured 0/2 in the candidate-admission run: neither fired, a detectability finding, not a reason for a new rule |

## Resolved: the mongo-auth family does have a confirmed generic alert path

This section used to say the opposite (no confirmed generic alert path, deferred to the first
measured round). The 24h continuous live run (mapping table above) settled it: `revoke_auth_mongodb` and
`user_unregistered_mongodb` both fire `KubePodNotReady` on the `geo` pod, 2/2, same as their
`_rate` siblings already confirmed in the candidate-admission run. The earlier worry (that the probe checks
mongo role/user state directly, not an app-level symptom, and that hotelReservation might not
expose a scrapable error-rate metric) turned out not to matter: the app-level symptom that
`KubePodNotReady` catches is not a Prometheus metric on hotelReservation at all, it is the pod
losing readiness once its DB writes start failing, the same `KubePodNotReady` alert key the other
five confirmed families collide on. No rule was added or changed to reach this; the no-bespoke-rule
constraint stands. Updated 2026-09-28.

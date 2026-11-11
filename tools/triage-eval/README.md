# triage-eval: running an evaluation

This guide covers the instrument itself: what each part does, how to bring up a cluster, run a
round against a model and score it. For what the project is and why, see the
[repository README](../../README.md). Commands below run from `tools/triage-eval/` unless noted.

## Layout

```
tools/triage-eval/
  harness/
    agent.py              agent loop: observe, decide, act, under a fixed tool contract
    envelope.py           C0 to C3 autonomy envelope: classifies every proposed action
    episode_runner.py     orchestrator: collect evidence, decide, classify, execute or escalate
    triage.py             TriageRecord schema and validation (the agent's structured output)
    triage_assembly.py    builds a TriageRecord from an episode
    webhook.py            Alertmanager webhook validation and group-window dedup
    fault_table.py        versioned fault table: family, cause class, mechanism, expected action
    scorer.py             per-episode rubric, alert-only oracles, clustered bootstrap
    episode_bridge.py     joins a round's journal with its TriageRecords into scorer episodes
    prompt_accounting.py  prompt budget accounting
  adapters/
    aiopslab_env.py           Environment over AIOpsLab problems
    vllm_model_client.py      model client for any OpenAI-compatible server (vLLM)
    mlx_model_client.py       model client for MLX on Apple silicon
    anthropic_model_client.py commercial API comparator arm
    mitigation_probes.py      root-cause probes: is the injected cause still present?
  scenarios/
    schedules/                committed fault schedules (evaluation and corpus)
    fault_schedule.py         schedule generator and validator
    corpus_schedule.py        training-corpus schedules, provably disjoint from evaluation
    run_continuous_admission.py  round driver: inject, watch alerts, probe, recover
    seed_cluster.sh           one isolated kind cluster per seed, several per host
    harvest_round.py          collects a finished round and joins it for scoring
    run_matrix.py             single-episode runs across prompt configurations
  kube-prometheus-stack/      frozen symptom alert rules (FROZEN_RULES.md explains each)
  k8s/                        agent Deployment, read-only RBAC, NetworkPolicy, admission policy
  kind/                       kind config patches and the vendored Calico manifest
  configs/                    model, decoding and environment configuration
  entrypoint.py, Dockerfile   the agent container: an Alertmanager webhook receiver
  setup_cluster.sh            cluster setup, monitoring stack, agent install
```

## Requirements

- Docker, [kind](https://kind.sigs.k8s.io/), kubectl and helm. Clusters run the node image
  `kindest/node:v1.36.1`, and the agent image pins kubectl to the same version (a test holds the
  two together).
- A clone of [AIOpsLab](https://github.com/microsoft/AIOpsLab), exported as `AIOPSLAB_HOME`.
- Python 3.12 with `requirements.txt` (on Linux, drop the `mlx-lm` line).
- A model endpoint: a vLLM server reachable from the kind network for self-hosted models, or an
  API key for the comparator arm. MLX runs need an Apple silicon machine with an interactive
  session (Metal is not available headless).

## 1. Bring up a cluster

Build the agent image once, from the repository root:

```bash
docker build -f tools/triage-eval/Dockerfile -t sslm-agent:local .
```

Then, from `tools/triage-eval/`:

```bash
export AIOPSLAB_HOME=/path/to/AIOpsLab
export APP_NAMESPACE=$(python3 scenarios/resolve_app_namespace.py "$AIOPSLAB_HOME")
export VLLM_BASE_URL=http://<host>:<port>/v1 VLLM_EGRESS_CIDR=<kind gateway cidr> VLLM_EGRESS_PORT=<port>
bash scenarios/seed_cluster.sh up 1
```

`seed_cluster.sh up <n>` creates its own kind cluster (with its own kubeconfig, so several seeds can
run on one host without sharing a kubectl context), installs Calico so NetworkPolicy is actually
enforced, installs kube-prometheus-stack with the frozen rules, loads the agent image and deploys it, and
applies the read-only RBAC, the zero-egress NetworkPolicy and the admission policy that keeps the
agent away from database pods. `seed_cluster.sh env <n>` prints the derived names and ports.

For a single cluster without seed isolation, `setup_cluster.sh` does the same steps behind opt-in
flags (`CALICO_INSTALL=1`, `AGENT_INSTALL=1`); its header documents every variable.

## 2. Run a round

Smoke first: one injection end to end, which proves alerts reach the agent and the model answers.

```bash
R2_MODE=smoke bash scenarios/seed_cluster.sh launch 1
```

Then the full round on a committed schedule:

```bash
R2_MODE=r2 R2_SCHEDULE=scenarios/schedules/r2-seed-1052.json bash scenarios/seed_cluster.sh launch 1
```

The driver (`run_continuous_admission.py`) waits for the app to settle, injects each fault at its
planned time, records every alert Alertmanager delivers, runs the root-cause probe before recovery
and writes an append-only journal. Results land in `scenarios/results/r2/`.

To run the same schedule against a commercial API instead of vLLM, set `COMPARATOR_ARM=1`,
`ANTHROPIC_API_KEY` and a spending cap in `SSLM_COMPARATOR_MAX_COST_USD`. The prompt, enum, tool
contract and scorer stay the same.

## 3. Harvest and score

```bash
python3 scenarios/harvest_round.py --out scenarios/results/r2/s1-r2.json \
  --schedule scenarios/schedules/r2-seed-1052.json
```

The harvest collects the agent's TriageRecords, joins them with the journal by time window
(`episode_bridge.py`) and writes `<out>.episodes.json`. An episode never disappears: anything the
round did not reach, or that did not settle, comes out as `not_measured` with a reason. Scoring is
a library call:

```python
from harness import scorer
report = scorer.score(episodes)   # exact class, mechanism level, oracles, bootstrap intervals
```

`scorer.dry_run()` scores a synthetic set and returns a hash that the tests pin, so any change to
scoring behavior is visible.

## Writing schedules

Evaluation schedules come from `fault_schedule.generate_schedule` (seeded, with minimum spacing
and settle time between faults). Training-corpus schedules come from `corpus_schedule.py`, which
refuses any overlap with evaluation seeds or held-out families and pins each committed schedule by
the hash of its canonical JSON. Commit a schedule before running it; never edit one after.

## Single episodes and prompt configurations

`run_matrix.py` runs one AIOpsLab problem at a time under a prompt configuration
(`configs/config_a.yaml`: prompt only, `config_b`: plus schema-constrained decoding, `config_c`:
plus training-free memory) and writes a JSON record and a transcript per run. It is the quickest
way to try a model locally:

```bash
python3 scenarios/run_matrix.py --config configs/config_a.yaml --env mlx-local \
  --model-client mlx_factories:make_model_client \
  --environment-factory mlx_factories:make_environment \
  --scenario misconfig_app_hotel_res-mitigation-1
```

## Design notes

- **Frozen symptom rules.** Alerts describe what an operator sees (crash loops, pods not ready, a
  claim that never binds) and never encode the injected cause. `FROZEN_RULES.md` records why each
  rule exists and what it cannot see.
- **Ground truth by injection.** The fault table is the only source of causes, actions and
  urgency. No LLM judge, no benchmark `success` flag.
- **Envelope before execution.** Read-only actions run freely, mutating actions are classified
  and capped, anything outside the envelope escalates to a human with the evidence attached.
- **Honest limits.** Unmeasured is never counted as correct or as wrong. A missing reading is
  reported as missing.
- **kind on a laptop.** `setup_cluster.sh` widens leader-election lease timings on the control
  plane, since the upstream defaults restart the scheduler and controller manager under slow
  Docker Desktop I/O.

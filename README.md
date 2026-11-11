# sre-triage-eval

**Measure an incident-triage agent before you put it on call.**

sre-triage-eval is an open evaluation harness for AI agents that triage Kubernetes incidents. It
injects a known fault into a running microservice app, lets the real alerting stack fire, hands
the alert to the agent inside a read-only envelope, and scores the root cause the agent names
against the fault that was actually injected. No LLM judge, no "looks plausible", no partial
credit for restating the alert.

It works with any model behind an OpenAI-compatible endpoint (vLLM, for self-hosted open
weights), with MLX on Apple silicon, and with a commercial API as a comparison arm.

## Why

Agents that read an alert and write a confident paragraph are easy to build and easy to demo.
The hard question is whether the paragraph is right. Most agent evaluations accept an answer that
sounds reasonable, or grade it with another model. On call, that is the difference between an
agent that shortens an incident and one that sends the engineer the wrong way at 3 a.m.

sre-triage-eval answers a narrower question, and answers it deterministically:

- **Did the agent find the cause, or just repeat the symptom?** Every episode has ground truth,
  because the harness injected the fault. The scorer compares the agent's cause class with a
  versioned fault table.
- **Is it better than reading the alert?** Each run is scored next to an oracle that sees only the
  alert labels and guesses the most likely cause. An agent that cannot beat that baseline is not
  diagnosing anything.
- **Can it be trusted with access?** The agent runs under read-only RBAC, network policy and an
  admission policy, and every proposed action is classified against an explicit autonomy envelope
  (C0 to C3) before anything executes.

## How it works

```
fault schedule -> inject fault -> app degrades -> frozen Prometheus rules fire
                                                          |
                                                 Alertmanager webhook
                                                          |
                              agent investigates (kubectl, PromQL), read-only
                                                          |
                                     structured triage record (JSON schema)
                                                          |
                 scorer: cause class vs injected fault, next to the alert-only oracle
```

1. **Scenarios.** Faults come from AIOpsLab problems on the HotelReservation app (auth revoked,
   unregistered DB user, bad config, unschedulable pod, and more), driven by committed, seeded
   schedules. The same schedule always produces the same sequence.
2. **Symptoms, not causes.** The alert rules are frozen and describe what an operator would see
   (pods crash-looping or not ready, a volume claim that never binds), never the injected cause. The agent has to investigate.
3. **Envelope.** The agent gets a fixed tool contract, a prompt budget, a turn cap and a time cap.
   Mutating actions are classified and capped, and the database tier is protected by admission
   policy.
4. **Scoring.** Exact cause-class accuracy, mechanism-level accuracy on alerts that are ambiguous by
   construction, and honest abstention counted apart from errors. Intervals come from a bootstrap
   that resamples whole (fault family, round) clusters with a fixed seed, so the same episodes
   always give the same numbers.

## Use it to

- **Compare candidate agents or models** on your own failure classes before choosing one.
- **Gate a rollout:** run the same schedules before and after a prompt, model or tool change.
- **Test generalization:** hold out a service or a failure mechanism and check whether the agent
  still finds the cause.
- **Build training data with a contract:** the corpus tooling generates fault schedules that are
  provably disjoint from the evaluation schedules.

## What is in the repo

| path | what it does |
|---|---|
| `tools/triage-eval/harness/` | agent loop, autonomy envelope, fault table, scorer, triage record schema, Alertmanager webhook |
| `tools/triage-eval/adapters/` | AIOpsLab environment and model clients (vLLM, MLX, commercial API comparator) |
| `tools/triage-eval/scenarios/` | fault schedules, schedule generator and validator, round drivers, cost and footprint collectors |
| `tools/triage-eval/kube-prometheus-stack/` | frozen symptom alert rules and the reasoning behind each (`FROZEN_RULES.md`) |
| `tools/triage-eval/k8s/`, `kind/` | read-only RBAC, network policies, admission policy, cluster setup |
| `tests/triage_eval/` | unit tests for all of the above, no cluster or GPU needed |
| `PREREGISTRATION.md` | pre-registered criteria for the specialized-model study (see below) |

## Quick start

The test suite runs anywhere with Python 3.12:

```
python -m venv .venv && . .venv/bin/activate
pip install -r tools/triage-eval/requirements.txt   # on Linux, drop the mlx-lm line first
python -m pytest tests/triage_eval -q
```

A real evaluation round needs a kind cluster with AIOpsLab, the kube-prometheus-stack from this
repo and a model endpoint. [`tools/triage-eval/README.md`](tools/triage-eval/README.md) covers
cluster setup, running a round and scoring it.

## Research

The harness is also the instrument of an ongoing study: can a small open-weights model, after
specialization, name the root cause on a service and on a failure mechanism it never saw in
training? The bar for that study, including the kill thresholds and the held-out split, is fixed
in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any training data exists. Results and raw run
data will be published with the preprint, whatever they show.

## Status

The harness, the fault table and the scorer are versioned and frozen for the current study.
Interfaces may change between studies. Each change bumps the table or scorer version, and the
tests pin both.

## Contributing

Issues and pull requests are welcome, in particular new fault families, adapters for other
benchmark environments and other model backends. A change to the fault table or to the scorer
needs a version bump and a test, since published numbers are tied to those versions.

## License

Apache-2.0, see [`LICENSE`](LICENSE). `tools/triage-eval/kind/vendor/calico-v3.28.2.yaml` is the
upstream Calico manifest, also Apache-2.0, see [`NOTICE`](NOTICE).

## About

Built by [ArchLabs](https://archlabs.pro). We run this evaluation inside a team's own cluster,
against the incident agent they are considering, so the decision rests on their failures and their
data instead of a demo. Contact: hello@archlabs.pro.

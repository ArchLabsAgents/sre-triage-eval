# Pre-registration: the bar a specialized small model must clear

**Status:** registered before any training corpus, adapter or training run exists.
**Published:** 2026-11 (the first public commit of this repository is the public timestamp).
**Written internally:** bar on 2026-10-15, corpus design and G0 gate on 2026-10-30, wording
amendments on 2026-11-02. No threshold changed after 2026-10-15.

This document fixes, in advance, how ArchLabs will decide whether fine-tuning a small open-weights
model makes it good at Kubernetes incident triage. It is written so that a result we would not
like is still readable as a result.

## 1. Question

Given a firing Prometheus alert on a microservice app, can a small self-hosted model, after
specialization, name the injected root cause class, including on a service and on a mechanism it
never saw in training?

"Better than the base model" is not enough. The base model scored 0 of 73 on the primary metric in
our earlier windows, so beating it proves little. A model that only memorized the table
"alert key -> cause" would beat the base without being useful. The bar is therefore set against an
oracle that reads only the alert.

## 2. Instrument

Everything below runs on the code in `tools/triage-eval/`:

- Fault injection on top of AIOpsLab problems, driven by committed schedules
  (`scenarios/schedules/`).
- Frozen Prometheus symptom rules (`kube-prometheus-stack/`, see `FROZEN_RULES.md`).
- The agent harness with a read-only envelope and capped actions (`harness/`, `k8s/`).
- The fault table, `TABLE_VERSION = "e6r-2026-09-28"` (`harness/fault_table.py`).
- The scorer, `SCORER_VERSION = "e6-scorer-2026-09-28"`, bootstrap with 10,000 resamples and seed
  20260928 (`harness/scorer.py`).

## 3. Hold-out, fixed before any corpus

Three strata over the six detected fault families:

| stratum | families | role |
|---|---|---|
| S0, seen | `misconfig_app_hotel_res`, `revoke_auth_mongodb`, `user_unregistered_mongodb` | the only detected families allowed in the training corpus |
| S1, new service, seen mechanism | `revoke_auth_mongodb_rate`, `user_unregistered_mongodb_rate` | same fine class as the S0 mongo families, on a service (`rate`, `mongodb-rate`) that never appears in the corpus as cause or symptom |
| S2, new mechanism | `assign_to_non_existent_node_hotel_res` | the class `node_selector_unschedulable` stays in the prompt enum, but no corpus example uses it as the answer |

**H = S1 + S2** is the held-out set. S1 is the stratum that matches the product: a customer brings
known failure classes into an environment the model has never seen.

The strata come from the structure of the table (mechanism, service, alert key), not from
per-family performance. The base model scored zero on every family, so there was no "easy" family
to avoid.

## 4. What the corpus may not contain

- No episode, alert, log, tool output or text from any family in H.
- No episode whose cause service or symptom service is `rate` or `mongodb-rate`. A healthy `rate`
  pod in a namespace listing does not count, since filtering it would make training prompts diverge
  from evaluation prompts.
- No episode from the six evaluation schedules.
- No output of the commercial API comparator arm, by construction.

Corpus schedules are generated with seeds from 30000 up, compared as integers against the
evaluation seeds {1052, 1140, 1283, 1627, 1695, 2425}, and with H families never injected (not
injected and filtered later). Split is by whole seed: `seed % 5 == 4` is validation. Seeds are
consumed in order, never skipped, because choosing a seed would be choosing the split.

Committed now, hashed as canonical JSON by `scenarios/corpus_schedule.py`:

| file | role | sha256 |
|---|---|---|
| `corpus-seed-30000.json` | G0 gate, train | `98b26a6a0136147c8d4246ca2946a23bfe01f913d0d6340570e4fb02d0defb1e` |
| `corpus-seed-30004.json` | validation | `df730331a6f816933f6528fead975cabe914b060225df4f30b065abf9f3e34ab` |
| `corpus-seed-30009.json` | validation | `aa6fcf815f9afa281f15792b9414f7f7210d45b852846c0fdb3e958b4e8ee11b` |

`tests/triage_eval/test_corpus_schedule.py` pins these hashes and the disjointness rules.
Checkpoint and hyperparameters are chosen only on validation drawn from the corpus itself.

## 5. Evaluation

- The six continuous rounds `r2-seed-1052/1140/1283` and `r3-seed-1627/1695/2425`, same harness,
  table, scorer, prompt and enum as the base arm.
- Per round: S0 6 episodes, S1 4, S2 2. Over six rounds H has 36 episodes in 18 clusters
  (family x round): S1 24, S2 12.
- The adapter is frozen before the first evaluation window: served-shard sha256, adapter hash, base
  revision and corpus manifest are recorded first. Also frozen: cause classes and their order, table
  and scorer versions, prompt with tool contract, prompt budget of 14,000 chars, 8 turns and 900 s
  per episode, envelope caps, temperature 0, 512 output tokens, structured outputs, thinking off,
  `--max-model-len 16384`, the vLLM version and the image digest.

**Fine oracle (O1-fine).** Uses the same alert key as the scorer's O1 oracle and predicts the most
frequent fine class for that key, ties broken by name. It is fit in-sample on all scored episodes
(S0 + S1 + S2) inside each resample and scored on H. Fitting on H alone would hand it the answer,
since in H the geo key only carries one mechanism.

## 6. Primary criterion, H2-O

All three must pass:

- **(a)** Adapter H3a on H >= 0.60. Kill below 0.40. A result in 0.40 to 0.59 gets one extension of
  four more rounds, once.
- **(b)** Adapter minus O1-fine on H, paired exact fine-class accuracy, bootstrap clustered by
  family x round (10,000 resamples, seed 20260928): lower bound of the CI > 0.
- **(c)** H3b (mechanism level, ambiguous subset of the table, all strata, O1 oracle) passes as
  defined in the scorer.

**Kill of H2-O:** (a) below 0.40, or (b) with lower bound <= 0, after the extension if any.
Consequence: the training axis ends. It is reported as a result.

A secondary paired McNemar test (exact, two-sided, 0.05) against the base model with the same tool
contract, on the same six rounds, is necessary but not sufficient.

**Reading by stratum (descriptive, never a verdict).** S0, S1 and S2 are reported separately with
Wilson intervals next to O1-fine. Success on S0 enters no criterion. A claim of generalization to an
unseen mechanism is made only if S2 reaches H3a >= 0.60 on its 12 episodes; otherwise the claim is
restricted to new services (S1), even if H2-O passes. S2 has one family and 12 episodes, so its
interval is wide by construction.

**Effective bar.** Against O1-fine, which gets 26 of 36 right with the real keys, (b) needs
roughly 33 or 34 of 36 on H. S2 = 10 fails even with S1 at 24/24. This was computed before any
adapter exists and is stated so that nobody reads a pass as easy.

## 7. Gate G0: measure the trajectory source before collecting

Before any paid collection, one ~19 h window per candidate source on `corpus-seed-30000.json`, one
stack. Sources in order, each only if the previous fails: the base 9B itself (temperature 0.7 for
generation only), then open-weights teachers (Apache-2.0 or MIT), same prompt, enum, envelope,
contract, budget and decoding, thinking off.

**Deterministic acceptance of a trajectory** (all clauses): app settled; family in S0 and no unhealthy
`rate`/`mongodb-rate`/`memcached-rate` pod; final class **exactly** right by the table (partial
rejects); no escalation-by-cap reason; no instrument-error citation; no mutating action on an
escalate row; and the class's evidence token was actually seen in context or in a successful tool
observation, not only in the diagnosis text. Tokens are frozen in `MANIFEST.json`.

**Rule, fixed before the data:** target 60 accepted injections (20 per family), 1.08 injections per
hour per stack, cap of 400 VM-hours, rate taken from the **worst** family. With one stack each family
needs at least **2 of 8** injections with an accepted trajectory. Unmeasured injections stay in the
denominator. G0 repeats only for infrastructure failure, with the reason written before reading the
acceptance. 2/8 has Wilson [0.07, 0.59], so passing is a signal to continue, not an estimate.

**If no source passes:** phase 1 does not happen and the remaining budget is not spent.

## 8. What was known when this was written

Written after the base-model evaluation windows and before any corpus, adapter or adapter episode.
Known at the time: the base scored 0/73 on H3a; a commercial API model with the same prompt, enum
and scorer scored 12/12 on one seed (degenerate interval, not precision), which ruled out the
prompt and enum as the cause of the zero; the 9B with a tool contract named the cause in 1 of 5;
the O1 oracle scored 0.553 on the ambiguous subset. Every amendment after 2026-10-15 only tightened
or clarified the bar. Raw run data ships with the preprint.

## 9. Open items

None of these move a threshold. They are decided and recorded before G0 or before collection:
corpus families beyond S0 (would need a separate table that does not touch the fault table or the
scorer hash), number of stacks per window, teacher licensing review, and trajectory sink details.

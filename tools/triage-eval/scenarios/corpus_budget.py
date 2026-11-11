"""GPU budget of the training corpus, as arithmetic instead of a table typed by hand.

Budget tables are printed from `render()`, so a premise changes here or nowhere. Plain price-list
projection: no invoice, no VM. Measured inputs say how they were measured; everything in a
SCENARIOS row is a premise until gate G0 measures it, above all `accepted_injection_rate`, which
has no measurement today.

Run with `python -m scenarios.corpus_budget` from tools/triage-eval.
"""

from __future__ import annotations

from dataclasses import dataclass

# List prices, USD per hour, read on 2026-10-30. H100 80GB (n3-H100x1): hyperstack.cloud/gpu-pricing.
USD_PER_HOUR = {"A6000": 0.50672, "H100": 2.50}

# Measured.
VM_OVERHEAD_HOURS = 1.33  # billed VM beyond the driver, worst of the short windows
EVAL_WINDOW_HOURS_ONE_SEED = 24.38  # a one-seed 24 h window with a 7B model; a 24 h 9B window never ran
EVAL_WINDOW_HOURS_THREE_SEEDS = 24.65  # a three-seed window, one VM
EVAL_ROUNDS = 6  # the six paired evaluation rounds
EXTENSION_ROUNDS = 4  # the single extension allowed in the 0.40-0.59 band
TOKENS_PER_TURN = 4228  # S0, 9B windows: 4069 prompt + 159 completion
OVERRUN_FACTOR = 1.22  # a past evaluation round closed 22% over its projection

# Pre-registered in scenarios/corpus_schedule.py; a test reads the committed files against these.
G0_INJECTIONS = 24
G0_SCHEDULE_HOURS = 17.4
VALIDATION_WINDOW_HOURS = 17.5  # upper end of the two committed validation schedules

# Premise, not measure: the fault density of the old clock (26 faults in 24 h). The driver follows
# the schedule clock, so a corpus schedule plans 1.33 to 1.38; this keeps a margin for lost entries.
CONSERVATIVE_INJECTIONS_PER_STACK_HOUR = 1.08

# Design ceiling: past this the collection stops and goes back to a decision card.
CORPUS_VM_HOURS_CEILING = 400.0

# First pass stops here; only a validation below the cut sends the collection on to the target.
FIRST_PASS_ACCEPTED_INJECTIONS = 60

# Phase 2, smallest run proposed (30 steps x 6 groups x 8 rollouts): a proposal with no evidence
# behind it, kept only so the ask has a contingent line.
GRPO_INJECTIONS = 180
GRPO_TRAIN_HOURS_A6000 = 7.2


@dataclass(frozen=True)
class Scenario:
    name: str
    source_gpu: str  # what serves the model that generates the trajectories
    sources_on_h100: int  # teachers measured in G0; the 9B always goes first, on the A6000
    accepted_injections: int  # target
    accepted_injection_rate: float  # of the WORST S0 family: the schedule is balanced
    injections_per_stack_hour: float
    stacks: int
    trajectories_per_injection: float
    trained_turns_per_trajectory: int
    epochs: int
    runs: int
    train_tokens_per_second_a6000: float
    h100_train_speedup: float
    train_overhead_hours: float
    eval_seeds_per_window: int


C = CONSERVATIVE_INJECTIONS_PER_STACK_HOUR
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("best", "A6000", 0, 60, 0.50, 1.33, 3, 1.5, 4, 3, 1, 3289.0, 3.5, 1.5, 3),
    Scenario("likely, 3 stacks", "H100", 1, 120, 0.25, C, 3, 2.0, 6, 3, 3, 1645.0, 3.0, 2.0, 1),
    Scenario("likely, 1 stack", "H100", 1, 120, 0.25, C, 1, 2.0, 6, 3, 3, 1645.0, 3.0, 2.0, 1),
    Scenario("worst", "H100", 2, 120, 0.10, C, 1, 2.0, 8, 5, 6, 705.0, 2.5, 3.0, 1),
)


def gate_hours(s: Scenario) -> dict[str, float]:
    """G0 is one stack per source on the committed schedule, however fast the scenario collects."""
    per_source = G0_SCHEDULE_HOURS + VM_OVERHEAD_HOURS
    return {"A6000": per_source, "H100": per_source * s.sources_on_h100}


def corpus_hours(s: Scenario) -> tuple[float, bool]:
    """VM hours of the collection and whether the ceiling cut it short."""
    injected = s.accepted_injections / s.accepted_injection_rate
    hours = injected / (s.injections_per_stack_hour * s.stacks)
    return min(hours, CORPUS_VM_HOURS_CEILING), hours > CORPUS_VM_HOURS_CEILING


def accepted_injections_collected(s: Scenario) -> float:
    hours, _ = corpus_hours(s)
    return hours * s.injections_per_stack_hour * s.stacks * s.accepted_injection_rate


def lora_megatokens(s: Scenario) -> float:
    examples = accepted_injections_collected(s) * s.trajectories_per_injection * s.trained_turns_per_trajectory
    return examples * TOKENS_PER_TURN * s.epochs * s.runs / 1e6


def lora_hours(s: Scenario) -> dict[str, float]:
    seconds = lora_megatokens(s) * 1e6 / s.train_tokens_per_second_a6000
    return {
        "A6000": seconds / 3600 + s.train_overhead_hours,
        "H100": seconds / s.h100_train_speedup / 3600 + s.train_overhead_hours,
    }


def validation_hours(s: Scenario) -> float:
    """Live, adapter at T=0 on the two validation seeds: once, or twice if the target is reached."""
    passes = 1 if s.accepted_injections <= FIRST_PASS_ACCEPTED_INJECTIONS else 2
    return passes * 2 * (VALIDATION_WINDOW_HOURS + VM_OVERHEAD_HOURS)


def grpo_hours(s: Scenario) -> dict[str, float]:
    """Contingent. Same GPU as the collection, or the serving and the training do not coexist."""
    collection = GRPO_INJECTIONS / (s.injections_per_stack_hour * s.stacks)
    speedup = 1.0 if s.source_gpu == "A6000" else s.h100_train_speedup
    return {"gpu": s.source_gpu, "hours": collection + GRPO_TRAIN_HOURS_A6000 / speedup}


def eval_hours(s: Scenario) -> float:
    """Adapter plus the paired 9B base: two arms, always on the A6000."""
    if s.eval_seeds_per_window == 1:
        return 2 * EVAL_ROUNDS * EVAL_WINDOW_HOURS_ONE_SEED
    return 2 * (EVAL_ROUNDS // s.eval_seeds_per_window) * EVAL_WINDOW_HOURS_THREE_SEEDS


def extension_hours() -> float:
    """+4 rounds, once, when H3a lands in the 0.40-0.59 band. Both arms, one seed per VM."""
    return 2 * EXTENSION_ROUNDS * EVAL_WINDOW_HOURS_ONE_SEED


def _usd(hours: float, gpu: str) -> float:
    return hours * USD_PER_HOUR[gpu]


def budget(s: Scenario) -> dict:
    gate = gate_hours(s)
    corpus, capped = corpus_hours(s)
    lora = lora_hours(s)
    validation = validation_hours(s)
    grpo = grpo_hours(s)
    evaluation = eval_hours(s)
    extension = extension_hours()
    lines = {
        "gate_g0": {"A6000": gate["A6000"], "H100": gate["H100"],
                    "usd": _usd(gate["A6000"], "A6000") + _usd(gate["H100"], "H100")},
        "corpus": {s.source_gpu: corpus, "usd": _usd(corpus, s.source_gpu), "capped": capped,
                   "accepted_injections": accepted_injections_collected(s)},
        "lora": {"A6000": lora["A6000"], "H100_equivalent": lora["H100"], "usd": _usd(lora["A6000"], "A6000"),
                 "megatokens": lora_megatokens(s)},
        "validation": {"A6000": validation, "usd": _usd(validation, "A6000")},
        "grpo_contingent": {grpo["gpu"]: grpo["hours"], "usd": _usd(grpo["hours"], grpo["gpu"])},
        "evaluation": {"A6000": evaluation, "usd": _usd(evaluation, "A6000")},
        "extension_contingent": {"A6000": extension, "usd": _usd(extension, "A6000")},
    }
    firm = sum(lines[key]["usd"] for key in ("gate_g0", "corpus", "lora", "validation", "evaluation"))
    contingent = lines["grpo_contingent"]["usd"] + lines["extension_contingent"]["usd"]
    return {
        **lines,
        "total_usd": firm,
        "total_with_overrun_usd": firm * OVERRUN_FACTOR,
        "ceiling_usd": (firm + contingent) * OVERRUN_FACTOR,
    }


def render() -> str:
    budgets = [(s, budget(s)) for s in SCENARIOS]

    def row(label: str, cell) -> str:
        return "| " + label + " | " + " | ".join(cell(s, b) for s, b in budgets) + " |"

    def hours(value: float, gpu: str, usd: float) -> str:
        return f"{value:,.0f} h {gpu}, US${usd:,.0f}"

    lines = [
        "| line | " + " | ".join(s.name for s, _ in budgets) + " |",
        "|---|" + "---|" * len(budgets),
        row("gate G0, 9B", lambda s, b: hours(b["gate_g0"]["A6000"], "A6000", _usd(b["gate_g0"]["A6000"], "A6000"))),
        row("gate G0, teacher", lambda s, b: hours(b["gate_g0"]["H100"], "H100", _usd(b["gate_g0"]["H100"], "H100"))),
        row("(a) collection", lambda s, b: hours(b["corpus"][s.source_gpu], s.source_gpu, b["corpus"]["usd"])
            + (" (ceiling)" if b["corpus"]["capped"] else "")),
        row("accepted injections", lambda s, b: f"{b['corpus']['accepted_injections']:,.0f}"),
        row("(b) LoRA", lambda s, b: hours(b["lora"]["A6000"], "A6000", b["lora"]["usd"])
            + f" ({b['lora']['H100_equivalent']:,.0f} h H100; {b['lora']['megatokens']:,.1f} Mtok)"),
        row("live validation", lambda s, b: hours(b["validation"]["A6000"], "A6000", b["validation"]["usd"])),
        row("(d) evaluation, adapter + base", lambda s, b: hours(b["evaluation"]["A6000"], "A6000",
                                                                b["evaluation"]["usd"])),
        row("**total without contingents**", lambda s, b: f"**US${b['total_usd']:,.0f}**"),
        row("total with 22% overrun", lambda s, b: f"US${b['total_with_overrun_usd']:,.0f}"),
        row("(c) GRPO, contingent", lambda s, b: hours(b["grpo_contingent"][s.source_gpu], s.source_gpu,
                                                        b["grpo_contingent"]["usd"])),
        row("(e) +4 rounds, contingent", lambda s, b: hours(b["extension_contingent"]["A6000"], "A6000",
                                                              b["extension_contingent"]["usd"])),
        row("ceiling with contingents and overrun", lambda s, b: f"US${b['ceiling_usd']:,.0f}"),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())

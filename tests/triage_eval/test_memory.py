from harness.memory import Episode, EpisodicMemory


def test_add_persists_to_store(tmp_path):
    store_path = tmp_path / "episodes.jsonl"
    memory = EpisodicMemory.load(store_path)
    memory.add(
        Episode(
            task_id="1",
            task_description="checkout service high error rate",
            action={"action_type": "remediate", "target": "checkout"},
            outcome="success",
        )
    )

    reloaded = EpisodicMemory.load(store_path)
    assert len(reloaded.episodes) == 1
    assert reloaded.episodes[0].task_id == "1"


def test_retrieve_ranks_by_word_overlap(tmp_path):
    memory = EpisodicMemory.load(tmp_path / "episodes.jsonl")
    memory.add(
        Episode(
            task_id="unrelated",
            task_description="database connection pool exhausted",
            action={},
            outcome="failure",
        )
    )
    memory.add(
        Episode(
            task_id="relevant",
            task_description="checkout service high error rate under load",
            action={},
            outcome="success",
        )
    )

    results = memory.retrieve("checkout service error rate spike", top_k=1)
    assert len(results) == 1
    assert results[0].task_id == "relevant"


def test_retrieve_empty_store_returns_empty(tmp_path):
    memory = EpisodicMemory.load(tmp_path / "episodes.jsonl")
    assert memory.retrieve("anything") == []


def test_retrieve_breaks_ties_by_recency(tmp_path):
    memory = EpisodicMemory.load(tmp_path / "episodes.jsonl")
    memory.add(Episode(task_id="older", task_description="pod crash loop", action={}, outcome="failure"))
    memory.add(Episode(task_id="newer", task_description="pod crash loop", action={}, outcome="success"))

    results = memory.retrieve("pod crash loop", top_k=1)
    assert results[0].task_id == "newer"


def test_set_verdict_stamps_only_unknown_turns_of_that_task_and_persists(tmp_path):
    store = tmp_path / "episodes.jsonl"
    memory = EpisodicMemory.load(store)
    memory.add(Episode(task_id="t1", task_description="d", action={}, outcome="o1"))
    memory.add(Episode(task_id="t1", task_description="d", action={}, outcome="o2"))
    memory.add(Episode(task_id="t2", task_description="d", action={}, outcome="o3"))
    memory.add(Episode(task_id="t1", task_description="d", action={}, outcome="o4", verdict="failure"))

    assert memory.set_verdict("t1", "success") == 2

    reloaded = EpisodicMemory.load(store)
    assert [e.verdict for e in reloaded.episodes] == ["success", "success", "unknown", "failure"]


def test_set_verdict_without_match_leaves_store_untouched(tmp_path):
    store = tmp_path / "episodes.jsonl"
    memory = EpisodicMemory.load(store)

    assert memory.set_verdict("nobody", "success") == 0
    assert not store.exists()


def test_truncate_outcome_caps_long_environment_output():
    from harness.memory import OUTCOME_MAX_CHARS, truncate_outcome

    long_output = "x" * (OUTCOME_MAX_CHARS * 3)
    truncated = truncate_outcome(long_output)

    assert truncated.endswith("[...]")
    assert len(truncated) < OUTCOME_MAX_CHARS + 10
    assert truncate_outcome("  short  ") == "short"

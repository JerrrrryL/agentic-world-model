"""Tests for the PI speedrun converter.

The committed sample under ``tests/data/pi_speedrun/`` is four truncated runs —
claude-code (with sub-agents, usage and PI's truncation marker), codex
(``spawn_agent`` linkage), prime-agent (``rlm()`` linkage) and a "live" run
whose progression uses the second manifest shape. Its manifest counts were
adjusted to the truncated streams so the count criteria hold there too.

Every number asserted against the full release was measured, not assumed.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from awm.traj.convert_pi import (
    SPAWN_TOOLS,
    convert_all,
    convert_run,
    convert_stream,
    is_harness_message,
    load_manifest,
    normalize_progression,
    to_event,
)
from awm.traj.schema import USAGE_KEYS, read_events, read_meta, validate_stream

CC = "claude-fable-5--claude-code--4ed2e4e07637"
CODEX = "openai-gpt-5-6-sol--codex--044f97fbcd18"
PRIME = "openai-gpt-5-6-sol--prime-agent--da21c9c65bf8"
LIVE = "qwen-qwen3-8-max--claude-code--99f02617a4d2"


@pytest.fixture(scope="module")
def pi_sample(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.rootpath) / "tests" / "data" / "pi_speedrun"


@pytest.fixture(scope="module")
def converted(pi_sample: Path) -> dict[str, tuple[list, object]]:
    manifest = load_manifest(pi_sample)
    return {mr["run"]: convert_run(pi_sample, mr["run"], mr) for mr in manifest["runs"]}


# --- pure unit tests, no data ------------------------------------------------


def _raw(**kw):
    base = {"type": "text", "role": "user", "i": 0, "turn": 0}
    return base | kw


def test_harness_messages_are_recognised():
    for text in (
        "continue",
        "  continue\n",
        "Continue from where you left off.",
        "Read program.md and follow it exactly. Run fully autonomously — never stop",
        "<task-notification>\n<task-id>bx015ef6u</task-id>",
        "<subagent_notification>\n{\"agent_path\":\"x\"}",
        "[context compacted — handoff summary]\n\nAnother language model started",
        "This session is being continued from a previous conversation that ran out",
        "<environment_context>\n  <cwd>/w</cwd>",
        "Stop hook feedback:\n[Read program.md...]",
        "<system-reminder>\n## TODO List",
    ):
        assert is_harness_message(text), text


def test_agent_text_is_not_mistaken_for_harness():
    # Sub-agent task prompts and ordinary agent messages must stay origin=agent:
    # every downstream decision count is built on this boundary.
    for text in (
        "You are running one training experiment in the work dir. Run bash run.sh.",
        "Run experiment E85 exactly once by executing bash run.sh (one trial, not 8).",
        "continue the sweep at 3100 steps",
        "I'll continue",
        "Reading program.md and following it exactly is what I already did.",
        "",
        None,
    ):
        assert not is_harness_message(text), text


def test_qwen_cache_key_maps_onto_cache_read():
    e = to_event(_raw(role="assistant", usage={"in": 1, "out": 2, "cache": 3}), "r", "main", 0)
    assert e.usage == {"in": 1, "out": 2, "cache_read": 3}


def test_grok_usage_note_is_not_counted_as_tokens():
    e = to_event(
        _raw(role="assistant", usage={"in": 7, "out": 1, "note": "billed once per prompt"}),
        "r", "main", 0,
    )
    assert e.usage == {"in": 7, "out": 1}
    assert e.extra["usage_note"] == "billed once per prompt"


def test_system_role_folds_onto_user_and_is_harness():
    e = to_event(_raw(role="system", text="You are Grok released by xAI."), "r", "main", 0)
    assert (e.role, e.origin, e.extra["role_upstream"]) == ("user", "harness", "system")


def test_truncation_marker_sets_the_flag():
    cut = to_event(_raw(type="tool_result", text="abc…[+1234 chars]"), "r", "main", 0)
    whole = to_event(_raw(type="tool_result", text="abc"), "r", "main", 1)
    assert cut.truncated and not whole.truncated


def test_repeated_usage_is_dropped_but_a_second_call_in_a_turn_is_not():
    # Shape taken from openai-gpt-5-6-sol--codex--044f97fbcd18 turn 2: two calls
    # under one turn number, each repeated on its own events.
    raw = [
        _raw(role="assistant", type="thinking", turn=2, usage={"in": 10356, "out": 317}),
        _raw(role="assistant", type="tool_use", tool="shell", turn=2,
             usage={"in": 10356, "out": 317}),
        _raw(role="assistant", type="text", turn=2, usage={"in": 15619, "out": 204}),
    ]
    events = convert_stream(raw, "r", "main")
    assert [e.usage for e in events] == [{"in": 25975, "out": 521}, None, None]


def test_progression_shapes_normalise_to_one_list():
    rows = normalize_progression(
        [
            {"value": "3150", "agent_h": 4.98, "tok_at": 98227, "cost": 37.49, "ts": None,
             "mtime": None, "n": None, "name": None, "logfile": None},
            {"steps": 3270, "val_loss": 3.277003, "t": 1755172521},
        ]
    )
    assert rows[0] == {"step_value": 3150, "val_loss": None, "at_agent_h": 4.98,
                       "at_tokens": 98227, "at_cost": 37.49, "ts": None}
    assert rows[1]["step_value"] == 3270
    assert rows[1]["val_loss"] == 3.277003
    assert rows[1]["ts"].endswith("Z")


# --- sample-backed tests -----------------------------------------------------


def test_manifest_has_the_four_sample_runs(pi_sample: Path):
    manifest = load_manifest(pi_sample)
    assert manifest["baseline"] == 3290 and manifest["human_record"] == 2600
    assert {r["run"] for r in manifest["runs"]} == {CC, CODEX, PRIME, LIVE}


def test_main_event_count_matches_manifest(converted, pi_sample: Path):
    manifest = {r["run"]: r for r in load_manifest(pi_sample)["runs"]}
    for run_id, (events, _) in converted.items():
        n_main = sum(1 for e in events if e.agent_id == "main")
        assert n_main == manifest[run_id]["n_events"], run_id


def test_streams_validate_and_number_each_agent_from_zero(converted):
    for run_id, (events, _) in converted.items():
        validate_stream(events, run_id)
        per_agent = collections.defaultdict(list)
        for e in events:
            per_agent[e.agent_id].append(e.i)
        for agent_id, idx in per_agent.items():
            assert idx == list(range(len(idx))), (run_id, agent_id)
        assert "main" in per_agent


def test_usage_is_kept_once_per_turn(converted):
    for run_id, (events, _) in converted.items():
        seen = set()
        for e in events:
            if e.usage is not None and e.turn is not None:
                assert (e.agent_id, e.turn) not in seen, (run_id, e.agent_id, e.turn)
                seen.add((e.agent_id, e.turn))


def test_the_goal_prompt_is_harness_and_agent_text_is_not(converted):
    events, meta = converted[CC]
    first = events[0]
    assert first.type == "text" and first.role == "user"
    assert first.origin == "harness"
    assert first.text.startswith("Read program.md and follow it exactly.")
    assert meta.n_by_origin["agent"] > meta.n_by_origin["harness"]
    # A sub-agent's opening task prompt was written by the parent agent.
    sub_first = next(e for e in events if e.agent_id.startswith("sub-") and e.i == 0)
    assert sub_first.origin == "agent"


def test_claude_code_tool_results_carry_the_truncation_flag(converted):
    events, _ = converted[CC]
    cut = [e for e in events if e.truncated]
    assert cut, "the claude-code sample should contain PI's …[+N chars] marker"
    for e in cut:
        assert (e.text or "").rstrip().endswith("chars]") or (e.summary or "").rstrip().endswith(
            "chars]"
        )


@pytest.mark.parametrize("run_id", [CC, CODEX, PRIME, LIVE])
def test_every_sample_subagent_links_to_its_spawning_tool_use(converted, run_id):
    events, meta = converted[run_id]
    by_id = {e.tool_use_id: e for e in events if e.type == "tool_use"}
    assert meta.subagents
    for sub in meta.subagents:
        parent = sub["parent_tool_use"]
        assert parent, (run_id, sub["id"])
        call = by_id[parent]
        expected = SPAWN_TOOLS.get(meta.harness)
        if expected:
            assert call.tool in expected
        first = next(e for e in events if e.agent_id == sub["id"] and e.i == 0)
        assert first.parent_tool_use == parent


def test_subagent_streams_are_in_the_same_file(converted):
    events, meta = converted[CC]
    ids = {e.agent_id for e in events}
    assert ids == {"main"} | {s["id"] for s in meta.subagents}
    assert all(s["id"].startswith("sub-") for s in meta.subagents)


def test_runmeta_carries_the_score_flags_and_manifest_extras(converted):
    _, meta = converted[CC]
    assert meta.source == "pi_speedrun"
    assert meta.final_score == {
        "metric": "train_steps", "value": 2726, "direction": "lower",
        "baseline": 3290, "reference": 2600,
    }
    assert meta.flags["validity"] == "healthy"
    assert meta.budget == {"gpus": 8, "gpu_type": "H200"}
    # tokens carries USAGE_KEYS counters, not the manifest's economics totals.
    assert meta.tokens == meta.extra["tokens_from_events"]
    assert set(meta.tokens) <= set(USAGE_KEYS)
    assert meta.extra["tokens_source"] == "events"
    assert meta.extra["track"] == "track3-noweb"
    assert meta.extra["effort"] == "high"
    assert meta.extra["n_records"] == 47
    assert meta.extra["note"]
    assert meta.duration_s > 0
    assert [f["path"] for f in meta.extra["scratch_files"]][0] == "scratchpad/thread.md"

    _, live = converted[LIVE]
    assert live.flags["validity"] == "healthy"
    assert live.extra["progression"][0]["val_loss"] is not None


def test_convert_all_writes_events_meta_and_scratch(pi_sample: Path, tmp_path: Path):
    import gzip

    metas = convert_all(pi_sample, tmp_path)
    assert len(metas) == 4
    for meta in metas:
        events = list(read_events(tmp_path / f"{meta.run_id}.jsonl.gz"))
        assert len(events) == meta.n_events
        assert read_meta(tmp_path / f"{meta.run_id}.meta.json").run_id == meta.run_id
        base = tmp_path / f"{meta.run_id}.scratch"
        written = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
        assert written == {f["path"] for f in meta.extra["scratch_files"]}
        # The decision log goes to disk verbatim and contributes no events.
        with gzip.open(pi_sample / f"scratch-{meta.run_id}.json.gz", "rt") as fh:
            thread = next(f for f in json.load(fh) if f.get("rel") == "scratchpad/thread.md")
        assert (base / "scratchpad" / "thread.md").read_text() == thread["text"]
        assert meta.n_events == sum(1 for e in events if e.agent_id == "main") + sum(
            s["n_events"] for s in meta.subagents
        )


def test_convert_all_honours_limit(pi_sample: Path, tmp_path: Path):
    assert len(convert_all(pi_sample, tmp_path, limit=1)) == 1


# --- the full release --------------------------------------------------------

#: kimi-k3--kimi-code--512eb075aefa reused sub-agent ids across sessions: 66
#: index entries, 40 distinct ids, and the bundle kept one stream per id.
ID_COLLISION_RUN = "kimi-k3--kimi-code--512eb075aefa"

#: Measured linked/total sub-agents per harness over the full release.
#: codex: the two Luna runs carry agent_path=None and spawn through exec JS.
#: grok-cli: the grok-4.5 run has no timestamps at all and three grok-4.6 runs
#: log no spawn tool, so 212 of 228 children cannot be attributed to a call.
LINKAGE = {
    "claude-code": (839, 839),
    "kimi-code": (261, 287),
    "codex": (112, 375),
    "grok-cli": (16, 228),
    "prime-agent": (25, 25),
}


@pytest.fixture(scope="module")
def full_release(pi_raw_root):
    manifest = load_manifest(pi_raw_root)
    return manifest, {
        mr["run"]: (mr, *convert_run(pi_raw_root, mr["run"], mr)) for mr in manifest["runs"]
    }


@pytest.fixture(scope="module")
def pi_raw_root() -> Path:
    from awm.paths import raw_dir

    root = raw_dir("pi_speedrun")
    if not (root / "traces" / "manifest.json.gz").exists():
        pytest.skip(f"PI speedrun release not fetched: {root}")
    return root


@pytest.mark.needs_data
def test_all_runs_match_the_manifest_and_validate(full_release):
    manifest, runs = full_release
    assert len(runs) == 41
    for run_id, (mr, events, meta) in runs.items():
        validate_stream(events, run_id)
        n_main = sum(1 for e in events if e.agent_id == "main")
        assert n_main == mr["n_events"], run_id
        n_subs = len(meta.subagents)
        if run_id == ID_COLLISION_RUN:
            assert (n_subs, mr["n_subagents"]) == (40, 66)
            assert len(meta.extra["dropped_index_entries"]) == 26
        else:
            assert n_subs == mr["n_subagents"], run_id
        assert meta.extra["n_sub_events"] == sum(s["n_events"] for s in meta.subagents)


@pytest.mark.needs_data
def test_subagent_linkage_coverage_per_harness(full_release):
    _, runs = full_release
    got: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for mr, _events, meta in runs.values():
        stats = meta.extra["subagent_linkage"]
        got[mr["harness"]][0] += stats["n_linked"]
        got[mr["harness"]][1] += stats["n_subagents"]
    assert {h: tuple(v) for h, v in got.items() if v[1]} == LINKAGE


#: Harnesses whose per-call usage PI captured completely. Once the distinct
#: usage records of a turn are summed (codex and qwen-code log several calls
#: under one turn number), event-summed ``out`` equals ``economics.out_tok`` to
#: the token: codex 7 runs, prime-agent 6, qwen-code 2. Keeping only a turn's
#: first record instead recovers 0.40x-0.92x on codex and 0.08x on qwen-code,
#: so this is the regression guard for that rule.
EXACT_OUT_HARNESSES = ("codex", "prime-agent", "qwen-code")


@pytest.mark.needs_data
def test_token_totals_against_manifest_economics(full_release):
    """Sharp where the release allows it, loose where it does not.

    16 of 41 runs (OpenRouter-backed claude-code, kimi-code, kimi-code-goal,
    claude-code-goal and the four grok-4.6 runs) carry no per-call ``usage``
    whatsoever, so their event-summed total is 0 and nothing can be checked.
    The summed-over-every-key ratio stays loose because the keys are not
    disjoint per harness — OpenAI counts cached input inside ``in``, so codex
    lands near 2.0x — and because ``economics`` counts every API call the proxy
    saw, retries and event-free calls included. Measured spread: 0.28x
    (claude-opus-5 claude-code) to 1.99x (codex).
    """
    _, runs = full_release
    no_usage, checked, exact = [], [], []
    for run_id, (mr, events, meta) in runs.items():
        tokens = meta.extra["tokens_from_events"]
        if not tokens:
            no_usage.append(run_id)
            continue
        total = (mr["economics"] or {}).get("total_tok")
        if total:
            ratio = sum(tokens.values()) / total
            assert 0.25 <= ratio <= 2.0, (run_id, ratio)
            checked.append(ratio)
        out = (mr["economics"] or {}).get("out_tok")
        if mr["harness"] in EXACT_OUT_HARNESSES:
            assert tokens["out"] == out, (run_id, tokens["out"], out)
            exact.append(run_id)
        elif mr["harness"] == "pi":
            assert abs(tokens["out"] - out) <= 0.001 * out, run_id
    assert len(no_usage) == 16
    assert len(checked) == 24
    assert len(exact) == 15


@pytest.mark.needs_data
def test_every_run_normalises_its_progression(full_release):
    _, runs = full_release
    shapes = collections.Counter()
    for mr, _events, meta in runs.values():
        rows = meta.extra["progression"]
        assert len(rows) == len(mr["progression"])
        for row in rows:
            assert isinstance(row["step_value"], int)
            shapes["val_loss" if row["val_loss"] is not None else "steps_only"] += 1
    assert shapes == {"steps_only": 339, "val_loss": 104}

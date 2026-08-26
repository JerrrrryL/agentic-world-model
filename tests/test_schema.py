"""Tests for the event schema itself — the contract every converter is written against."""

from __future__ import annotations

import gzip
import json

import pytest

from awm.traj.schema import (
    MAIN_AGENT,
    Event,
    RunMeta,
    SchemaError,
    iter_runs,
    read_events,
    read_meta,
    summarize,
    validate_event,
    validate_stream,
    write_run,
)


def ev(i: int, **kw) -> Event:
    base = dict(run_id="r1", agent_id=MAIN_AGENT, i=i, type="text", role="assistant")
    base.update(kw)
    return Event(**base)


class TestValidateEvent:
    def test_accepts_a_minimal_event(self):
        validate_event(ev(0))

    @pytest.mark.parametrize(
        "kw, msg",
        [
            ({"type": "message"}, "unknown type"),
            ({"role": "system"}, "unknown role"),
            ({"origin": "launcher"}, "unknown origin"),
            ({"type": "tool_use"}, "without a tool name"),
            ({"type": "tool_result", "role": "assistant"}, "must have role 'user'"),
            ({"redacted": True, "text": "leaked"}, "redacted event carries text"),
            ({"usage": {"input_tokens": 5}}, "unknown usage keys"),
        ],
    )
    def test_rejects(self, kw, msg):
        with pytest.raises(SchemaError, match=msg):
            validate_event(ev(0, **kw))

    def test_redacted_thinking_is_the_normal_anthropic_case(self):
        validate_event(ev(0, type="thinking", redacted=True, text=None))

    def test_tool_result_is_a_user_event(self):
        validate_event(ev(0, type="tool_result", role="user", text="ok", parent_tool_use="t1"))


class TestValidateStream:
    def test_numbers_each_agent_separately(self):
        events = [
            ev(0),
            ev(1),
            ev(0, agent_id="sub-a"),
            ev(1, agent_id="sub-a"),
            ev(2),
        ]
        validate_stream(events, "r1")

    def test_rejects_a_gap_in_numbering(self):
        with pytest.raises(SchemaError, match="expected i=1, got 2"):
            validate_stream([ev(0), ev(2)], "r1")

    def test_rejects_a_shared_counter_across_agents(self):
        with pytest.raises(SchemaError, match="expected i=0, got 2"):
            validate_stream([ev(0), ev(1), ev(2, agent_id="sub-a")], "r1")

    def test_rejects_a_foreign_run_id(self):
        with pytest.raises(SchemaError, match="!= 'r1'"):
            validate_stream([ev(0, run_id="other")], "r1")

    def test_rejects_usage_repeated_within_a_turn(self):
        # PI repeats usage on every assistant event of a turn; converters must dedupe,
        # otherwise token totals come out multiplied by the events per turn.
        events = [ev(0, turn=1, usage={"in": 10}), ev(1, turn=1, usage={"in": 10})]
        with pytest.raises(SchemaError, match="usage recorded twice"):
            validate_stream(events, "r1")

    def test_allows_the_same_turn_number_in_different_agents(self):
        events = [ev(0, turn=1, usage={"in": 10}), ev(0, agent_id="sub-a", turn=1, usage={"in": 3})]
        validate_stream(events, "r1")

    def test_allows_usage_without_a_turn(self):
        validate_stream([ev(0, usage={"in": 1}), ev(1, usage={"in": 2})], "r1")


class TestSummarize:
    def test_counts_types_origins_tools_and_tokens(self):
        events = [
            ev(0, type="thinking", redacted=True, turn=1, usage={"in": 100, "out": 20}),
            ev(1, type="tool_use", tool="Bash", tool_use_id="t1", turn=1),
            ev(2, type="tool_result", role="user", parent_tool_use="t1", turn=1),
            ev(3, type="text", role="user", origin="harness", text="continue", turn=2),
            ev(4, type="tool_use", tool="Bash", tool_use_id="t2", turn=3, usage={"in": 50}),
        ]
        s = summarize(events)
        assert s["n_events"] == 5
        assert s["n_by_type"] == {"thinking": 1, "tool_use": 2, "tool_result": 1, "text": 1}
        assert s["n_by_origin"] == {"agent": 4, "harness": 1}
        assert s["tools"] == {"Bash": 2}
        assert s["tokens"] == {"in": 150, "out": 20}

    def test_omits_token_kinds_that_never_appear(self):
        assert "cache_write" not in summarize([ev(0, usage={"in": 1})])["tokens"]

    def test_empty(self):
        assert summarize([])["n_events"] == 0


class TestRoundTrip:
    def _run(self, tmp_path):
        events = [
            ev(0, type="thinking", redacted=True, turn=1, usage={"in": 9, "cache_read": 100}),
            ev(1, type="tool_use", tool="Bash", tool_use_id="t1", args={"command": "ls"}, turn=1),
            ev(
                2,
                type="tool_result",
                role="user",
                parent_tool_use="t1",
                text="a\nb",
                truncated=True,
                turn=1,
            ),
            ev(0, agent_id="sub-x", type="text", text="child", parent_tool_use="t1"),
        ]
        meta = RunMeta(
            run_id="r1",
            source="unit_test",
            benchmark="none",
            model="m",
            harness="h",
            **summarize(events),
        )
        return events, meta

    def test_write_then_read(self, tmp_path):
        events, meta = self._run(tmp_path)
        path = write_run(events, meta, tmp_path)
        back = list(read_events(path))
        assert [e.to_json() for e in back] == [e.to_json() for e in events]

        m = read_meta(tmp_path / "r1.meta.json")
        assert (m.run_id, m.n_events, m.tools) == ("r1", 4, {"Bash": 1})

    def test_absent_fields_are_not_written(self, tmp_path):
        events, meta = self._run(tmp_path)
        write_run(events, meta, tmp_path)
        with gzip.open(tmp_path / "r1.jsonl.gz", "rt") as fh:
            first = json.loads(fh.readline())
        assert "tool" not in first and "args" not in first
        assert "truncated" not in first  # false flags stay out of the file
        assert first["redacted"] is True  # true ones do not

    def test_write_validates_by_default(self, tmp_path):
        events, meta = self._run(tmp_path)
        events[1].i = 7
        with pytest.raises(SchemaError):
            write_run(events, meta, tmp_path)

    def test_iter_runs_pairs_events_with_meta(self, tmp_path):
        events, meta = self._run(tmp_path)
        write_run(events, meta, tmp_path)
        found = list(iter_runs(tmp_path))
        assert len(found) == 1
        run_id, ep, mp = found[0]
        assert run_id == "r1" and ep.exists() and mp.exists()

    def test_write_is_atomic_leaving_no_tmp_behind(self, tmp_path):
        events, meta = self._run(tmp_path)
        write_run(events, meta, tmp_path)
        assert not list(tmp_path.glob("*.tmp"))

    def test_a_lone_surrogate_survives_the_round_trip(self, tmp_path):
        """Two opencode runs truncate a web result mid-emoji and publish half of
        one. It cannot be encoded as UTF-8, so writing it raised and lost the run;
        it must be re-escaped, not dropped — the file is the corpus."""
        events, meta = self._run(tmp_path)
        events[2].text = "* [\ud83d"  # the high half of a truncated U+1F4C1
        events[2].extra = {"note": "\ud83d"}
        meta.extra = {"note": "\ud83d"}
        path = write_run(events, meta, tmp_path)
        assert "\\ud83d" in gzip.open(path, "rt", encoding="utf-8").read()
        back = list(read_events(path))
        assert back[2].text == "* [\ud83d"
        assert back[2].extra == {"note": "\ud83d"}
        assert read_meta(tmp_path / "r1.meta.json").extra == {"note": "\ud83d"}

"""PostTrainBench conversion: committed samples always, full runs when present.

The committed samples under ``tests/data/posttrainbench/`` are verbatim line
subsets of the two runs the converters were written against (see
``make_samples.py``); the ``full_runs`` fixture points at those two complete runs
and skips when they are not on this machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hv.paths import raw_dir
from hv.traj import convert_claude_code, convert_codex
from hv.traj.posttrainbench import (
    RunDir,
    build_run,
    convert_run_dir,
    detect_harness,
    iter_run_dirs,
    make_run_dir,
    parse_agent_config,
    parse_run_dir_name,
    read_line_stream,
)
from hv.traj.schema import MAIN_AGENT, read_events, read_meta, validate_stream

CLAUDE_CFG = "claude_non_api_max_claude-opus-4-8_10h_run1"
CLAUDE_RUN = "gsm8k_Qwen_Qwen3-1.7B-Base_17315721"
CODEX_CFG = "codex_non_api_high_gpt-5.4_10h_run1"
CODEX_RUN = "gsm8k_Qwen_Qwen3-1.7B-Base_16934887"

#: The two complete runs the acceptance numbers below were measured on. They are
#: too large to commit, so these tests read them from the fetched release and
#: skip when it is absent. HV_PTB_SAMPLES points at a directory holding
#: ``run_claude/`` and ``run_codex/`` instead.
FULL_SAMPLES = Path(os.environ["HV_PTB_SAMPLES"]) if "HV_PTB_SAMPLES" in os.environ else None


@pytest.fixture
def ptb_samples(sample_dir: Path) -> Path:
    return sample_dir / "posttrainbench"


@pytest.fixture
def full_runs() -> dict[str, RunDir]:
    out = {}
    for name, cfg, run in (("claude", CLAUDE_CFG, CLAUDE_RUN), ("codex", CODEX_CFG, CODEX_RUN)):
        path = raw_dir("posttrainbench") / cfg / run
        if FULL_SAMPLES is not None and (FULL_SAMPLES / f"run_{name}" / "solve_out.txt").exists():
            path = FULL_SAMPLES / f"run_{name}"
        if not (path / "solve_out.txt").exists():
            pytest.skip(f"full {name} sample not available ({path})")
        out[name] = RunDir(path=path, agent_config=cfg, **parse_agent_config(cfg),
                           **parse_run_dir_name(run))
    return out


def _by_type(events, kind):
    return [e for e in events if e.type == kind]


# --- directory conventions -------------------------------------------------


def test_parse_agent_config():
    assert parse_agent_config(CLAUDE_CFG) == {
        "agent": "claude",
        "config": "non_api_max",
        "model": "claude-opus-4-8",
        "hours": 10.0,
        "run_index": 1,
        "context_1m": False,
    }
    assert parse_agent_config(CODEX_CFG)["model"] == "gpt-5.4"
    # "_1m_" marks the 1M-context variant and doubles the separator before the hours.
    m1 = parse_agent_config("claude_non_api_max_claude-fable-5_1m__10h_run1")
    assert (m1["model"], m1["context_1m"], m1["config"]) == ("claude-fable-5", True, "non_api_max")
    assert parse_agent_config("codex_non_api_max_gpt-5.6-sol_10h")["run_index"] is None


def test_parse_run_dir_name():
    assert parse_run_dir_name(CLAUDE_RUN) == {
        "benchmark": "gsm8k",
        "hf_org": "Qwen",
        "base_model": "Qwen3-1.7B-Base",
        "cluster_id": "17315721",
    }


def test_iter_run_dirs(ptb_samples: Path):
    runs = {r.agent: r for r in iter_run_dirs(ptb_samples)}
    assert set(runs) == {"claude", "codex"}
    assert runs["claude"].run_id == f"{CLAUDE_CFG}__{CLAUDE_RUN}"
    assert runs["codex"].benchmark == "gsm8k"


def test_make_run_dir_rejects_junk(tmp_path: Path):
    with pytest.raises(ValueError):
        make_run_dir("not-a-config", tmp_path / "nor-a-run")


# --- the line reader -------------------------------------------------------


def test_read_line_stream(ptb_samples: Path):
    claude = list(read_line_stream(ptb_samples / CLAUDE_CFG / CLAUDE_RUN / "solve_out.txt"))
    codex = list(read_line_stream(ptb_samples / CODEX_CFG / CODEX_RUN / "solve_out.txt"))
    # Measured: every claude line carries the "[ISO] " prefix, no codex line does.
    assert all(ts for ts, _o, _n, _r in claude[1:])
    assert not any(ts for ts, _o, _n, _r in codex)
    # Non-JSON launcher output is passed through, not dropped.
    assert [n for _ts, o, n, _r in claude if o is None] == list(range(1, 11))
    assert [n for _ts, o, n, _r in codex if o is None] == list(range(1, 11))
    assert [n for _ts, _o, n, _r in claude] == list(range(1, len(claude) + 1))
    assert claude[10][1]["type"] == "system"


def test_detect_harness(ptb_samples: Path):
    assert detect_harness(ptb_samples / CLAUDE_CFG / CLAUDE_RUN / "solve_out.txt") == "claude-code"
    assert detect_harness(ptb_samples / CODEX_CFG / CODEX_RUN / "solve_out.txt") == "codex"


def test_detect_harness_unknown(tmp_path: Path):
    p = tmp_path / "solve_out.txt"
    p.write_text('not json\n{"type": "something-else"}\n', encoding="utf-8")
    assert detect_harness(p) == "unknown"


# --- committed samples -----------------------------------------------------


def test_claude_sample_converts(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "claude-code"
    assert meta.source == "posttrainbench"
    assert meta.model == "claude-opus-4-8"
    assert meta.benchmark == "gsm8k"
    assert meta.budget == {"hours": 10.0}
    assert meta.duration_s == 10 * 3600 + 5 * 60 + 1
    assert meta.final_score["value"] == pytest.approx(0.6118271417740713)
    assert meta.flags["contamination"] is False
    assert meta.extra["n_non_json_lines"] == 10
    assert meta.extra["n_lines"] == len(list(read_line_stream(run.solve_out)))
    assert set(meta.source_paths) >= {"solve_out", "metrics", "time_taken", "judgement"}

    # One tool_use per "Tool call" line in upstream's own rendering.
    parsed = (run.path / "solve_parsed.txt").read_text(encoding="utf-8")
    assert len(_by_type(events, "tool_use")) == parsed.count("Tool call") == 8

    # Both sessions of the trimmed sample are recorded; the second is still open.
    sessions = meta.extra["sessions"]
    assert [s["index"] for s in sessions] == [0, 1]
    assert sessions[0]["num_turns"] == 69
    assert "result_line" not in sessions[1]
    assert meta.cost_usd == pytest.approx(4.789912)


def test_claude_harness_origin(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, _meta = build_run(run)
    kinds = {e.extra["kind"] for e in events if e.origin == "harness" and e.extra}
    assert kinds == {"session_start", "task_started", "task_notification", "task_updated",
                     "rate_limit_event"}
    # Background tasks are notifications about an existing Bash call, not tool calls.
    assert not any(e.type == "tool_use" for e in events if e.origin == "harness")
    started = next(e for e in events if e.extra and e.extra.get("kind") == "task_started")
    assert started.parent_tool_use.startswith("toolu_")
    assert started.text == "Locate inspect_evals gsm8k task files"


def test_claude_turns_and_usage(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, _meta = build_run(run)
    # Turns are API responses, they never go backwards, and they do not reset
    # when the launcher restarts the CLI mid-run.
    turns = [e.turn for e in events if e.turn is not None]
    assert turns == sorted(turns)
    assert max(turns) == 7
    with_usage = [e for e in events if e.usage]
    assert len(with_usage) == len({e.turn for e in with_usage})
    assert set(with_usage[0].usage) <= {"in", "out", "cache_read", "cache_write"}


def test_claude_tool_results(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, _meta = build_run(run)
    results = _by_type(events, "tool_result")
    assert all(e.role == "user" and e.parent_tool_use for e in results)
    calls = {e.tool_use_id: e.tool for e in _by_type(events, "tool_use")}
    linked = [e for e in results if e.parent_tool_use in calls]
    assert linked and all(e.tool == calls[e.parent_tool_use] for e in linked)
    # ToolSearch answers with tool_reference blocks rather than text.
    blocky = [e for e in results if e.extra and "content_blocks" in e.extra]
    assert blocky and blocky[0].text is None


def test_codex_sample_converts(ptb_samples: Path):
    run = make_run_dir(CODEX_CFG, ptb_samples / CODEX_CFG / CODEX_RUN)
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "codex"
    assert meta.model == "gpt-5.4"
    assert meta.final_score["value"] == pytest.approx(0.4268385140257771)
    assert meta.duration_s is None  # no time_taken.txt in this run directory
    assert meta.flags["judgement_unavailable"] == "Entry not found"
    assert meta.extra["n_non_json_lines"] == 10
    assert meta.extra["thread_id"] == "019cc3e4-2307-7cf3-a8e0-489866e4a1cd"

    # No timestamps exist in this format; none are invented.
    assert all(e.ts is None for e in events)
    assert meta.t_start is None and meta.t_end is None

    # One event per item, however many item.* messages it produced.
    ids = [e.tool_use_id for e in _by_type(events, "tool_use")]
    assert len(ids) == len(set(ids))
    assert meta.extra["unfinished_items"] == ["item_140", "item_4"]

    usage = [e for e in events if e.usage]
    assert len(usage) == 1 and usage[0].i == 0
    assert usage[0].usage == {"in": 20976581, "out": 46993, "cache_read": 20707840}


def test_write_and_read_back(ptb_samples: Path, tmp_path: Path):
    run = make_run_dir(CODEX_CFG, ptb_samples / CODEX_CFG / CODEX_RUN)
    ep = convert_run_dir(run, tmp_path)
    events = list(read_events(ep))
    meta = read_meta(tmp_path / f"{run.run_id}.meta.json")
    assert len(events) == meta.n_events
    validate_stream(events, meta.run_id)


def test_claude_subagent_rows():
    """The Task tool nests rows under parent_tool_use_id; the sample has none."""
    spawn = {
        "type": "assistant",
        "message": {
            "id": "msg_a",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "content": [
                {"type": "tool_use", "id": "toolu_task", "name": "Task",
                 "input": {"description": "measure the baseline"}}
            ],
        },
    }
    child = {
        "type": "assistant",
        "parent_tool_use_id": "toolu_task",
        "message": {"id": "msg_b", "content": [{"type": "text", "text": "on it"}]},
    }
    rows = [(None, spawn, 1, ""), (None, child, 2, "")]
    events, _extra = convert_claude_code.convert(rows, "r")
    validate_stream(events, "r")
    assert [e.agent_id for e in events] == [MAIN_AGENT, "toolu_task"]
    assert [e.i for e in events] == [0, 0]
    assert events[1].parent_tool_use == "toolu_task"


def test_claude_usage_counted_once_per_message_id():
    """A sub-agent's lines can split a parent message; the repeated usage on its
    second half must not be counted twice."""
    def assistant(mid, text, parent=None):
        row = {
            "type": "assistant",
            "message": {"id": mid, "usage": {"input_tokens": 10, "output_tokens": 4},
                        "content": [{"type": "text", "text": text}]},
        }
        if parent:
            row["parent_tool_use_id"] = parent
        return row

    rows = [(None, assistant("msg_a", "first half"), 1, ""),
            (None, assistant("msg_b", "child", parent="toolu_task"), 2, ""),
            (None, assistant("msg_a", "second half"), 3, "")]
    events, _extra = convert_claude_code.convert(rows, "r")
    validate_stream(events, "r")
    assert [e.usage for e in events] == [{"in": 10, "out": 4}, {"in": 10, "out": 4}, None]


def test_claude_flags_truncated_tool_results():
    """Claude Code caps a background task's output in-band; the result is not
    the whole thing and must say so."""
    capped = ("<output>\nOutput truncated (2KB total). Full output saved to: "
              "/tmp/tasks/bfdrh8iig.output\n</output>")
    rows = [
        (None, {"type": "assistant", "message": {"id": "m", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "TaskOutput", "input": {}}]}}, 1, ""),
        (None, {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": capped}]}}, 2, ""),
        (None, {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "all of it"}]}}, 3, ""),
    ]
    events, _extra = convert_claude_code.convert(rows, "r")
    assert [e.truncated for e in events] == [False, True, False]


def test_codex_error_lines_become_harness_events():
    """A dropped response stream is a bare {"type": "error"} line; it is the CLI
    talking, and no input line may vanish."""
    rows = [
        (None, {"type": "turn.started"}, 1, ""),
        (None, {"type": "item.completed",
                "item": {"id": "i1", "type": "agent_message", "text": "hi"}}, 2, ""),
        (None, {"type": "error", "message": "Reconnecting... 1/5"}, 3, ""),
    ]
    events, _extra = convert_codex.convert(rows, "r")
    validate_stream(events, "r")
    assert [(e.origin, e.type) for e in events] == [("agent", "text"), ("harness", "text")]
    assert events[1].text == "Reconnecting... 1/5"
    assert events[1].extra == {"kind": "error"}


# --- the two complete runs -------------------------------------------------


def test_full_claude_acceptance(full_runs: dict[str, RunDir]):
    run = full_runs["claude"]
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.n_events == 639
    assert meta.n_by_type == {"text": 208, "thinking": 111, "tool_use": 160, "tool_result": 160}
    assert meta.n_by_origin == {"agent": 552, "harness": 87}
    assert meta.tools == {"Bash": 93, "Read": 32, "Write": 11, "TaskUpdate": 10,
                          "TaskCreate": 6, "Edit": 6, "ToolSearch": 2}
    # 208 text events = 121 assistant text blocks + 87 harness injections.
    assert len([e for e in events if e.type == "text" and e.origin == "agent"]) == 121

    parsed = (run.path / "solve_parsed.txt").read_text(encoding="utf-8")
    assert len(_by_type(events, "tool_use")) == parsed.count("Tool call") == 160

    assert meta.extra["n_lines"] == 663
    assert meta.extra["n_non_json_lines"] == 11
    assert meta.extra["n_sessions"] == 13
    # All thirteen sessions share one session id: the launcher restarts the CLI
    # with --continue, so sessions are told apart by init order, not by id.
    assert len(meta.extra["session_ids"]) == 1
    # total_cost_usd is cumulative over the run, so the run's cost is the last.
    assert meta.cost_usd == pytest.approx(17.26357075)
    assert meta.final_score["value"] == pytest.approx(0.6118271417740713)
    assert meta.duration_s == 36301.0
    assert meta.t_start == "2026-06-07T21:31:06Z"


def test_full_codex_acceptance(full_runs: dict[str, RunDir]):
    run = full_runs["codex"]
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.n_events == 283
    assert meta.n_by_type == {"thinking": 106, "text": 41, "tool_use": 73, "tool_result": 63}
    # 126 command_execution *messages* (63 started + 63 completed) are 63 items.
    assert meta.tools == {"command_execution": 63, "file_change": 8, "web_search": 1,
                          "todo_list": 1}
    assert len(_by_type(events, "tool_result")) == 63
    assert all(e.ts is None for e in events)
    assert meta.extra["n_lines"] == 300
    assert meta.extra["n_non_json_lines"] == 10
    assert meta.extra["unfinished_items"] == []
    assert meta.final_score["value"] == pytest.approx(0.4268385140257771)


@pytest.mark.needs_data
def test_whole_batch_converts(ptb_raw: Path, tmp_path: Path):
    runs = list(iter_run_dirs(ptb_raw))
    if not runs:
        pytest.skip(f"no PostTrainBench run directories under {ptb_raw}")
    for run in runs:
        events, meta = build_run(run)
        validate_stream(events, meta.run_id)
        assert meta.harness in ("claude-code", "codex")
        assert meta.n_events > 0
        # Four of the 82 runs in the default batch have no usable metrics.json.
        assert meta.final_score is not None or "metrics_unavailable" in meta.extra
        parsed = run.path / "solve_parsed.txt"
        if meta.harness == "claude-code" and parsed.exists():
            rendered = parsed.read_text(encoding="utf-8", errors="replace").count("Tool call")
            assert len(_by_type(events, "tool_use")) == rendered

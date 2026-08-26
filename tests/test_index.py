"""The index is the analysis layer's only entry point for run selection, so the
column set, the dtypes and the absent-is-NA rule are all part of its contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from awm import paths
from awm.traj import index
from awm.traj.schema import Event, RunMeta, summarize, write_run

SOURCE = "pi_speedrun"


def _events(run_id: str) -> list[Event]:
    return [
        Event(run_id, "main", 0, "text", "user", text="go", origin="harness", turn=0),
        Event(run_id, "main", 1, "thinking", "assistant", text="hm", turn=1,
              usage={"in": 100, "out": 20, "cache_read": 5}),
        Event(run_id, "main", 2, "tool_use", "assistant", tool="Bash", turn=1,
              tool_use_id="t1"),
        Event(run_id, "main", 3, "tool_result", "user", text="ok", parent_tool_use="t1"),
        Event(run_id, "sub-1", 0, "text", "assistant", text="done", parent_tool_use="t1"),
    ]


def _rich_meta(run_id: str = "rich") -> RunMeta:
    events = _events(run_id)
    s = summarize(events)
    return RunMeta(
        run_id=run_id,
        source=SOURCE,
        benchmark="speedrun",
        task_id="track3_optimizer",
        model="claude-opus-4-8",
        harness="claude-code",
        budget={"hours": 10.0, "gpus": 8, "gpu_type": "H200"},
        t_start="2026-06-07T21:31:06Z",
        t_end="2026-06-08T07:31:06Z",
        duration_s=36000.0,
        final_score={"metric": "train_steps", "value": 3030.0, "direction": "lower_is_better",
                     "normalized": 0.42, "baseline": 3290},
        tokens=s["tokens"],
        cost_usd=123.45,
        n_events=s["n_events"],
        n_by_type=s["n_by_type"],
        n_by_origin=s["n_by_origin"],
        tools=s["tools"],
        subagents=[{"id": "sub-1", "label": "explore", "n_events": 1}],
        flags={"validity": "flagged", "flagged_why": "subagent-contract A/B rerun"},
    )


def _bare_meta(run_id: str = "bare") -> RunMeta:
    """A run whose source published nothing but its identity."""
    return RunMeta(run_id=run_id, source=SOURCE, benchmark="speedrun")


def _write(tmp_path: Path, metas: list[RunMeta], source: str = SOURCE) -> Path:
    root = tmp_path / "events"
    for meta in metas:
        write_run(_events(meta.run_id) if meta.n_events else [], meta, root / source)
    return root


def test_columns_and_dtypes(tmp_path: Path) -> None:
    df = index.build(_write(tmp_path, [_rich_meta(), _bare_meta()]))
    assert list(df.columns) == list(index.COLUMNS)
    assert {c: str(t) for c, t in df.dtypes.items()} == index.DTYPES
    assert len(df) == 2


def test_empty_input_keeps_the_full_shape(tmp_path: Path) -> None:
    for df in (index.build(tmp_path / "nothing-here"), index.build(tmp_path), index.empty()):
        assert list(df.columns) == list(index.COLUMNS)
        assert {c: str(t) for c, t in df.dtypes.items()} == index.DTYPES
        assert len(df) == 0


def test_rich_run_flattens(tmp_path: Path) -> None:
    root = _write(tmp_path, [_rich_meta()])
    row = index.build(root).iloc[0]
    assert row.run_id == "rich"
    assert row.source == SOURCE
    assert row.task_id == "track3_optimizer"
    assert row.harness == "claude-code"
    assert row.budget_hours == 10.0
    assert row.budget_gpus == 8
    assert row.t_start == "2026-06-07T21:31:06Z"
    assert row.duration_s == 36000.0
    assert (row.metric, row.score, row.score_normalized) == ("train_steps", 3030.0, 0.42)
    assert row.direction == "lower_is_better"
    assert row.n_events == 5
    assert (row.n_thinking, row.n_tool_use, row.n_text) == (1, 1, 2)
    assert row.n_harness_events == 1
    assert row.n_subagents == 1
    assert (row.tok_in, row.tok_out, row.tok_cache_read) == (100, 20, 5)
    assert row.cost_usd == 123.45
    assert row.events_path == str(root / SOURCE / "rich.jsonl.gz")
    assert Path(row.events_path).exists()


def test_absent_is_na_not_zero(tmp_path: Path) -> None:
    row = index.build(_write(tmp_path, [_bare_meta()])).iloc[0]
    for col in ("task_id", "model", "harness", "budget_hours", "budget_gpus", "t_start", "t_end",
                "duration_s", "metric", "score", "score_normalized", "direction", "n_thinking",
                "n_tool_use", "n_text", "n_harness_events", "tok_in", "tok_out", "tok_cache_read",
                "cost_usd", "flagged", "flag_reasons"):
        assert row[col] is pd.NA, f"{col} should be NA, got {row[col]!r}"
    # These two are counted by the index itself, so they are known to be zero.
    assert row.n_events == 0
    assert row.n_subagents == 0


def test_counted_zero_stays_zero() -> None:
    """A stream that was summarized but held no thinking really did think zero times."""
    meta = RunMeta(run_id="r", source=SOURCE, benchmark="b", n_events=1,
                   n_by_type={"text": 1}, n_by_origin={"agent": 1}, tokens={"in": 7})
    row = index.row_from_meta(meta)
    assert row["n_thinking"] == 0 and row["n_tool_use"] == 0
    assert row["n_harness_events"] == 0
    # Tokens go the other way: we did not count them, the harness reported them.
    # This run reported an input count and no output count, which is unknown and
    # not zero — 106 converted Claude Code runs died before the line carrying `out`.
    assert row["tok_in"] == 7 and row["tok_out"] is None


def test_a_reported_zero_is_kept_apart_from_an_unreported_stream() -> None:
    """The distinction is only worth having if a real zero survives it."""
    meta = RunMeta(run_id="r", source=SOURCE, benchmark="b", n_events=1,
                   tokens={"in": 7, "out": 0})
    assert index.row_from_meta(meta)["tok_out"] == 0


@pytest.mark.parametrize(
    "flags, flagged, reason_contains",
    [
        ({}, None, None),
        ({"validity": "healthy", "flagged_why": None}, False, None),
        ({"validity": "flagged", "flagged_why": "A/B rerun"}, True, "validity=flagged: A/B rerun"),
        ({"contamination": False, "disallowed_model": False,
          "justification_contamination": "long text explaining why it is clean"}, False, None),
        ({"contamination": True, "disallowed_model": False,
          "justification_contamination": "trained on the test split"}, True,
         "contamination: trained on the test split"),
    ],
)
def test_flag_rules(flags: dict, flagged: bool | None, reason_contains: str | None) -> None:
    got_flagged, got_reasons = index.read_flags(flags)
    assert got_flagged is flagged
    if reason_contains is None:
        assert got_reasons is None
    else:
        assert reason_contains in got_reasons


def test_parquet_round_trip(tmp_path: Path) -> None:
    df = index.build(_write(tmp_path, [_rich_meta(), _bare_meta()]))
    p = index.save(df, tmp_path / "index.parquet")
    back = index.load(p)
    assert {c: str(t) for c, t in back.dtypes.items()} == index.DTYPES
    pd.testing.assert_frame_equal(df, back)
    assert back.loc[back.run_id == "bare", "cost_usd"].isna().all()


def test_empty_parquet_round_trip(tmp_path: Path) -> None:
    back = index.load(index.save(index.empty(), tmp_path / "empty.parquet"))
    assert list(back.columns) == list(index.COLUMNS)
    assert len(back) == 0


def test_multiple_sources_sorted(tmp_path: Path) -> None:
    root = tmp_path / "events"
    write_run([], RunMeta(run_id="z", source=SOURCE, benchmark="speedrun"), root / SOURCE)
    write_run([], RunMeta(run_id="a", source="posttrainbench", benchmark="gsm8k"),
              root / "posttrainbench")
    write_run([], RunMeta(run_id="b", source="posttrainbench", benchmark="gsm8k"),
              root / "posttrainbench")
    df = index.build(root)
    # "pi_speedrun" sorts before "posttrainbench" ('i' < 'o').
    assert list(df.source) == [SOURCE, "posttrainbench", "posttrainbench"]
    assert list(df.run_id) == ["z", "a", "b"]


def test_run_without_meta_is_skipped(tmp_path: Path) -> None:
    root = _write(tmp_path, [_rich_meta()])
    (root / SOURCE / "rich.meta.json").unlink()
    assert len(index.build(root)) == 0


def test_default_events_root_is_the_events_dir(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    """A root off by one level makes build() return an empty index in silence."""
    monkeypatch.setenv("AWM_DATA_ROOT", str(tmp_path))
    root = index.default_events_root()
    assert root == paths.events_dir("pi_speedrun").parent
    assert paths.events_dir("pi_speedrun") == root / "pi_speedrun"


@pytest.mark.needs_data
def test_index_real_events_dir() -> None:
    root = index.default_events_root()
    if not root.is_dir() or not any(root.iterdir()):
        pytest.skip(f"no converted runs yet: {root}")
    df = index.build(root)
    assert list(df.columns) == list(index.COLUMNS)
    assert {c: str(t) for c, t in df.dtypes.items()} == index.DTYPES
    # Every run on disk must reach the table: a wrong root makes build() return an
    # empty frame rather than raise, and every other assertion here passes vacuously.
    assert len(df) == len(list(root.glob("*/*.meta.json")))
    if len(df):
        assert df.duplicated(["source", "run_id"]).sum() == 0
        assert df.source.notna().all()
        assert (df.n_events >= 0).all()
        assert df.events_path.map(lambda p: Path(p).exists()).all()

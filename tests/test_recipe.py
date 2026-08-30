"""The digest decides what an extractor is allowed to see, so its filter is part
of the recipe dataset's contract: an event it drops is a field that comes back
null, and a run it empties is a record nobody can tell from "this agent trained
nothing".
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from awm.analysis import recipe


def _write(tmp_path: Path, events: list[dict], name: str = "r.jsonl.gz") -> Path:
    p = tmp_path / name
    with gzip.open(p, "wt") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return p


def _use(i: int, tool: str, args: dict, turn: int = 1) -> dict:
    return {"run_id": "r", "agent_id": "main", "i": i, "type": "tool_use",
            "role": "assistant", "turn": turn, "tool": tool, "args": args}


class TestNormaliseTool:
    @pytest.mark.parametrize("tool", ["Bash", "command_execution", "shell", "bash"])
    def test_every_harness_spells_a_shell_differently(self, tool: str) -> None:
        """26 opencode and 6 cursor runs of the split would vanish under a Bash-only filter."""
        assert recipe.normalise_tool(tool) == "shell"

    @pytest.mark.parametrize("tool", ["Write", "file_change", "edit", "write"])
    def test_and_a_file_write_too(self, tool: str) -> None:
        assert recipe.normalise_tool(tool) == "write"

    def test_an_unknown_tool_keeps_its_name_rather_than_becoming_shell(self) -> None:
        assert recipe.normalise_tool("WebFetch") == "webfetch"
        assert recipe.normalise_tool(None) == "?"


class TestSelect:
    def test_it_keeps_a_command_that_names_a_dataset(self, tmp_path: Path) -> None:
        p = _write(tmp_path, [_use(0, "Bash", {"command": "load_dataset('openai/gsm8k')"})])
        assert [e["i"] for e in recipe.select(p)] == [0]

    def test_it_drops_a_command_with_nothing_to_do_with_a_recipe(self, tmp_path: Path) -> None:
        p = _write(tmp_path, [_use(0, "Bash", {"command": "bash timer.sh"})])
        assert recipe.select(p) == []

    def test_a_result_rides_along_only_behind_a_kept_call(self, tmp_path: Path) -> None:
        """The tail of a training run's output is where the loss and the traceback are."""
        events = [
            _use(0, "Bash", {"command": "python train.py --lr 2e-5"}),
            {"i": 1, "type": "tool_result", "role": "user", "args": {"output": "loss 0.31"}},
            _use(2, "Bash", {"command": "ls"}),
            {"i": 3, "type": "tool_result", "role": "user", "args": {"output": "a.txt"}},
        ]
        kept = recipe.select(_write(tmp_path, events))
        assert [e["i"] for e in kept] == [0, 1]

    def test_an_orphan_result_is_not_kept(self, tmp_path: Path) -> None:
        """Index 5 does not follow index 0, so its call was filtered out."""
        events = [
            _use(0, "Bash", {"command": "python train.py"}),
            {"i": 5, "type": "tool_result", "role": "user", "args": {"output": "loss 0.31"}},
        ]
        assert [e["i"] for e in recipe.select(_write(tmp_path, events))] == [0]

    def test_the_agent_explaining_its_mixture_is_kept(self, tmp_path: Path) -> None:
        events = [{"i": 0, "type": "text", "role": "assistant",
                   "text": "I'll upsample MetaMathQA to 70% of the mixture."}]
        assert len(recipe.select(_write(tmp_path, events))) == 1

    def test_a_user_turn_is_never_kept_even_when_it_says_the_words(self, tmp_path: Path) -> None:
        """The task prompt names the benchmark; it is not evidence of a choice."""
        events = [{"i": 0, "type": "text", "role": "user",
                   "text": "finetune this model, learning_rate is yours to pick"}]
        assert recipe.select(_write(tmp_path, events)) == []

    def test_a_write_keeps_four_times_more_text_than_a_command(self, tmp_path: Path) -> None:
        long = "load_dataset x" + "y" * 20_000
        p = _write(tmp_path, [_use(0, "Write", {"content": long}),
                              _use(1, "Bash", {"command": long})])
        kept = {e["i"]: e["text"] for e in recipe.select(p)}
        assert len(kept[0]) > len(kept[1])
        assert kept[0].endswith("…[truncated]") and kept[1].endswith("…[truncated]")

    def test_overflow_drops_the_beginning_and_keeps_the_end(self, tmp_path: Path) -> None:
        """The last hour is the run the agent submitted; the first is exploration."""
        events = [_use(i, "Bash", {"command": "load_dataset " + "z" * 900}) for i in range(10)]
        kept = recipe.select(_write(tmp_path, events), budget=3_000)
        assert [e["i"] for e in kept] == [7, 8, 9]

    def test_a_non_scalar_argument_is_dropped_not_stringified(self, tmp_path: Path) -> None:
        """A todo list's nested payload would otherwise flood the digest."""
        p = _write(tmp_path, [_use(0, "Bash", {"command": "python train.py",
                                               "todos": [{"a": 1}, {"b": 2}]})])
        assert "todos" not in recipe.select(p)[0]["text"]


class TestRender:
    def test_the_score_never_reaches_the_extractor(self) -> None:
        """An extractor told the run scored 0.86 describes a good recipe. See the module docstring."""
        meta = {"trained_model": "Qwen_Qwen3-4B-Base", "benchmark": "gsm8k",
                "accuracy": 0.8637, "stderr": 0.012, "total_cost_usd": 48.2}
        text = recipe.render("exp/run", [{"i": 3, "turn": 2, "act": "shell", "text": "train.py"}], meta)
        assert "Qwen_Qwen3-4B-Base" in text
        assert "0.8637" not in text and "48.2" not in text
        assert "accuracy" not in text

    def test_the_event_index_survives_so_a_quote_can_be_checked(self) -> None:
        text = recipe.render("exp/run", [{"i": 214, "turn": 40, "act": "write", "text": "SFTTrainer("}])
        assert "[214]" in text and "turn=40" in text


class TestJoinOutcome:
    def test_it_attaches_the_label_after_the_fact(self) -> None:
        joined = recipe.join_outcome({"run": "r", "confidence": "high"},
                                     {"accuracy": 0.63, "stderr": 0.012, "num_turns": 122})
        assert joined["run"] == "r" and joined["confidence"] == "high"
        assert joined["accuracy"] == 0.63 and joined["num_turns"] == 122

    def test_a_catalogue_row_missing_a_field_yields_none_not_zero(self) -> None:
        """Same rule as the index: absent is NA, never 0."""
        assert recipe.join_outcome({"run": "r"}, {})["accuracy"] is None

"""Tests for the split registry.

A split file is the contract everyone trains and evaluates against, so the
tests are about the two ways it could lie: the materialized run lists no longer
following from the rule and the pinned catalogue, and a file drifting from the
shape the loader promises.
"""

from __future__ import annotations

import pytest

from awm import splits


def _row(
    exp: str,
    model: str,
    seed: str = "111",
    bench: str = "gsm8k",
    acc: float | None = 0.5,
    contam: bool = False,
    disallowed: bool = False,
) -> dict:
    """One catalogue row, shaped like upstream's ``viewer_data/index.json``."""
    return {
        "experiment": exp,
        "run_name": f"{bench}_{model}_{seed}",
        "benchmark": bench,
        "trained_model": model,
        "accuracy": acc,
        "contamination": {"flagged": contam},
        "disallowed_model": {"flagged": disallowed},
    }


RULE = {
    "by": "trained_model",
    "test": ["Qwen_Qwen3-4B-Base"],
    "require": {
        "accuracy": "present",
        "contamination_flagged": False,
        "disallowed_flagged": False,
    },
}


class TestApplyRule:
    def test_partitions_runs_by_the_heldout_model(self):
        catalog = {
            "runs": [
                _row("expB", "google_gemma-3-4b-pt"),
                _row("expA", "Qwen_Qwen3-4B-Base"),
                _row("expA", "google_gemma-3-4b-pt"),
            ]
        }
        parts = splits.apply_rule("gsm8k", RULE, catalog)
        assert parts["test"] == ["expA/gsm8k_Qwen_Qwen3-4B-Base_111"]
        # Sorted, so regeneration is byte-stable.
        assert parts["train"] == [
            "expA/gsm8k_google_gemma-3-4b-pt_111",
            "expB/gsm8k_google_gemma-3-4b-pt_111",
        ]

    def test_other_benchmarks_are_ignored(self):
        catalog = {"runs": [_row("expA", "Qwen_Qwen3-4B-Base", bench="aime2025")]}
        parts = splits.apply_rule("gsm8k", RULE, catalog)
        assert parts == {"train": [], "test": []}

    def test_unevaluated_and_flagged_runs_are_dropped_from_both_sides(self):
        catalog = {
            "runs": [
                _row("expA", "Qwen_Qwen3-4B-Base", acc=None),
                _row("expB", "google_gemma-3-4b-pt", contam=True),
                _row("expC", "google_gemma-3-4b-pt", disallowed=True),
                _row("expD", "google_gemma-3-4b-pt"),
            ]
        }
        parts = splits.apply_rule("gsm8k", RULE, catalog)
        assert parts == {"train": ["expD/gsm8k_google_gemma-3-4b-pt_111"], "test": []}

    def test_an_unknown_rule_key_is_an_error_not_a_silent_skip(self):
        with pytest.raises(splits.SplitError, match="require"):
            splits.apply_rule("gsm8k", {**RULE, "require": {"acc": "present"}}, {"runs": []})
        with pytest.raises(splits.SplitError, match="by"):
            splits.apply_rule("gsm8k", {**RULE, "by": "agent_model"}, {"runs": []})


SPLIT_BODY = """\
kind: run-split
name: demo-v1
dataset:
  repo: aisa-group/PostTrainBench-Trajectories
  repo_type: dataset
  revision: 39d3fcd794df51c062c8bd3b7f8523ba707aaeb3
  catalog: viewer_data/index.json
benchmark: gsm8k
rule:
  by: trained_model
  test: [Qwen_Qwen3-4B-Base]
  require: {accuracy: present, contamination_flagged: false, disallowed_flagged: false}
counts: {train: 2, test: 1}
splits:
  train:
    - expA/gsm8k_google_gemma-3-4b-pt_111
    - expB/gsm8k_google_gemma-3-4b-pt_111
  test:
    - expA/gsm8k_Qwen_Qwen3-4B-Base_111
"""

SELECTION_BODY = """\
kind: task-selection
name: demo-sel-v1
benchmark: airs
resources: {gpus: 1, gpu_type: H200}
budget: {official_h: 24}
tasks: [TaskA, TaskB]
"""


def _write(tmp_path, monkeypatch, rel: str, body: str) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    monkeypatch.setattr(splits, "splits_dir", lambda: tmp_path)


class TestLoading:
    def test_a_run_split_file_round_trips(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, "posttrainbench/demo-v1.yaml", SPLIT_BODY)
        s = splits.load("posttrainbench/demo-v1")
        assert s.id == "posttrainbench/demo-v1"
        assert s.benchmark == "gsm8k"
        assert s.dataset["revision"].startswith("39d3fcd")
        assert s.test == ("expA/gsm8k_Qwen_Qwen3-4B-Base_111",)
        assert len(s.train) == 2
        assert s.counts == {"train": 2, "test": 1}

    def test_a_task_selection_file_round_trips(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, "airs/demo-sel-v1.yaml", SELECTION_BODY)
        sel = splits.load_selection("airs/demo-sel-v1")
        assert sel.id == "airs/demo-sel-v1"
        assert sel.tasks == ("TaskA", "TaskB")
        assert sel.resources["gpu_type"] == "H200"
        assert sel.budget["official_h"] == 24

    def test_an_unknown_id_is_an_error(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, "posttrainbench/demo-v1.yaml", SPLIT_BODY)
        with pytest.raises(splits.SplitError, match="no split file"):
            splits.load("posttrainbench/nope")

    def test_each_loader_rejects_the_other_kind(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, "posttrainbench/demo-v1.yaml", SPLIT_BODY)
        _write(tmp_path, monkeypatch, "airs/demo-sel-v1.yaml", SELECTION_BODY)
        with pytest.raises(splits.SplitError, match="kind"):
            splits.load("airs/demo-sel-v1")
        with pytest.raises(splits.SplitError, match="kind"):
            splits.load_selection("posttrainbench/demo-v1")


class TestMalformed:
    """Hand edits must fail loudly, not shift membership in silence."""

    @pytest.mark.parametrize(
        "mangle, msg",
        [
            (lambda b: b.replace("counts: {train: 2, test: 1}", "counts: {train: 3, test: 1}"), "counts"),
            (
                lambda b: b.replace(
                    "- expB/gsm8k_google_gemma-3-4b-pt_111",
                    "- expA/gsm8k_Qwen_Qwen3-4B-Base_111",
                ),
                "both train and test",
            ),
            (
                lambda b: b.replace(
                    "- expB/gsm8k_google_gemma-3-4b-pt_111",
                    "- expA/gsm8k_google_gemma-3-4b-pt_111",
                ),
                "duplicate",
            ),
            (lambda b: b + "verdict: fine\n", "unknown"),
            (lambda b: b.replace("name: demo-v1", "name: other"), "name"),
        ],
    )
    def test_rejects(self, tmp_path, monkeypatch, mangle, msg):
        _write(tmp_path, monkeypatch, "posttrainbench/demo-v1.yaml", mangle(SPLIT_BODY))
        with pytest.raises(splits.SplitError, match=msg):
            splits.load("posttrainbench/demo-v1")


class TestCheck:
    """The committed lists must still follow from the rule and the catalogue."""

    CATALOG = {
        "runs": [
            _row("expA", "Qwen_Qwen3-4B-Base"),
            _row("expA", "google_gemma-3-4b-pt"),
            _row("expB", "google_gemma-3-4b-pt"),
        ]
    }

    def _split(self, tmp_path, monkeypatch, body: str = SPLIT_BODY) -> splits.Split:
        _write(tmp_path, monkeypatch, "posttrainbench/demo-v1.yaml", body)
        return splits.load("posttrainbench/demo-v1")

    def test_a_consistent_split_checks_clean(self, tmp_path, monkeypatch):
        assert splits.check(self._split(tmp_path, monkeypatch), self.CATALOG) == []

    def test_reports_an_eligible_run_the_lists_omit(self, tmp_path, monkeypatch):
        catalog = {"runs": self.CATALOG["runs"] + [_row("expC", "google_gemma-3-4b-pt")]}
        problems = splits.check(self._split(tmp_path, monkeypatch), catalog)
        assert any("expC" in p and "train" in p for p in problems)

    def test_reports_a_listed_run_the_rule_rejects(self, tmp_path, monkeypatch):
        catalog = {"runs": self.CATALOG["runs"][:-1]}  # expB gone from the catalogue
        problems = splits.check(self._split(tmp_path, monkeypatch), catalog)
        assert any("expB" in p for p in problems)

    def test_reports_a_catalog_checksum_mismatch(self, tmp_path, monkeypatch):
        import hashlib
        import json

        raw = json.dumps(self.CATALOG).encode()
        pinned = SPLIT_BODY.replace(
            "catalog: viewer_data/index.json",
            f"catalog: viewer_data/index.json\n  catalog_sha256: {hashlib.sha256(raw).hexdigest()}",
        )
        s = self._split(tmp_path, monkeypatch, pinned)
        assert splits.check(s, self.CATALOG, catalog_bytes=raw) == []
        problems = splits.check(s, self.CATALOG, catalog_bytes=raw + b" ")
        assert any("sha256" in p for p in problems)


GSM8K_SPLIT = "posttrainbench/gsm8k-base-holdout-v1"


class TestCommittedGsm8kSplit:
    """What the contract currently says, so an accidental edit is loud."""

    def test_holds_out_qwen3_4b_with_146_train_47_test(self):
        s = splits.load(GSM8K_SPLIT)
        assert s.benchmark == "gsm8k"
        assert s.rule["test"] == ["Qwen_Qwen3-4B-Base"]
        assert s.counts == {"train": 146, "test": 47}
        assert all("Qwen_Qwen3-4B-Base" in run for run in s.test)
        assert not any("Qwen_Qwen3-4B-Base" in run for run in s.train)

    def test_is_pinned_to_one_dataset_revision_and_catalogue(self):
        s = splits.load(GSM8K_SPLIT)
        assert s.dataset["repo"] == "aisa-group/PostTrainBench-Trajectories"
        assert len(s.dataset["revision"]) == 40
        assert len(s.dataset["catalog_sha256"]) == 64

    def test_replays_cleanly_against_the_fetched_catalogue(self):
        from awm.paths import raw_dir
        from awm.traj import fetch

        path = raw_dir("posttrainbench") / fetch.PTB_CATALOG
        if not path.exists():
            pytest.skip(f"catalogue not fetched: {path} (run `awm traj fetch posttrainbench`)")
        s = splits.load(GSM8K_SPLIT)
        assert splits.check(s, fetch.ptb_catalog(), catalog_bytes=path.read_bytes()) == []


class TestListing:
    def test_enumerates_every_committed_yaml_as_an_id(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, "posttrainbench/demo-v1.yaml", SPLIT_BODY)
        _write(tmp_path, monkeypatch, "airs/demo-sel-v1.yaml", SELECTION_BODY)
        assert splits.list_ids() == ["airs/demo-sel-v1", "posttrainbench/demo-v1"]


class TestCommittedAirsSelection:
    def test_holds_the_eight_gpu_heavy_tasks_on_an_h200(self):
        sel = splits.load_selection("airs/gpu-heavy-8-v1")
        assert len(sel.tasks) == 8
        assert sel.resources["gpu_type"] == "H200"
        assert sel.budget["official_h"] == 24


class TestAirsAdapterReadsTheSelection:
    """The task generator's only registry is the committed selection file."""

    BODY = SELECTION_BODY.replace("name: demo-sel-v1", "name: gpu-heavy-8-v1")

    def test_selected_tasks_follow_the_committed_selection(self, tmp_path, monkeypatch):
        from awm.adapters import airs

        _write(tmp_path, monkeypatch, "airs/gpu-heavy-8-v1.yaml", self.BODY)
        assert airs.selected_tasks() == ["TaskA", "TaskB"]

    def test_a_task_outside_the_selection_gets_no_resource_facts(self, tmp_path, monkeypatch):
        from awm.adapters import airs

        _write(tmp_path, monkeypatch, "airs/gpu-heavy-8-v1.yaml", self.BODY)
        assert airs._selection_facts("TaskA") == (
            {"gpus": 1, "gpu_type": "H200"},
            {"official_h": 24},
        )
        assert airs._selection_facts("SomethingElse") == ({}, {})

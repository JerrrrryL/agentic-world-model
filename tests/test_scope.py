"""Tests for the scope registry.

The registry's job is to say which tasks we run and to stay honest about the
numbers it copies from elsewhere, so the tests are mostly about the second part:
``check()`` must notice when a scope file, the document, and upstream's own
metadata stop agreeing.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from awm import scope
from awm.paths import REPO_ROOT


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Files are cached per benchmark; tests that write their own must not inherit."""
    scope._load_file.cache_clear()
    yield
    scope._load_file.cache_clear()


class TestLoading:
    def test_every_benchmark_file_loads(self):
        for name in scope.BENCHMARKS:
            assert scope.load(name), f"{name} has no tasks"

    def test_ids_are_unique_and_prefixed_by_their_benchmark(self):
        ids = [e.id for e in scope.load()]
        assert len(ids) == len(set(ids))
        for e in scope.load():
            assert e.id.startswith(f"{e.benchmark}/")
            assert e.task == e.id.split("/", 1)[1]

    def test_every_task_has_a_usable_metric(self):
        for e in scope.load():
            assert e.metric.get("name")
            assert e.metric["direction"] in ("higher_is_better", "lower_is_better")

    def test_get_and_select(self):
        e = scope.get("airs/ReadingComprehensionSquadExactMatch")
        assert e.benchmark == "airs"
        assert scope.select(benchmark="airs") == scope.load("airs")
        # A None filter is ignored rather than matching nothing.
        assert scope.select(benchmark="airs", family=None) == scope.load("airs")

    def test_unknown_benchmark_and_unknown_id_are_errors(self):
        with pytest.raises(scope.ScopeError):
            scope.load("nope")
        with pytest.raises(KeyError):
            scope.get("airs/NoSuchTask")


class TestInheritance:
    """Keys above ``tasks`` apply to every task; a task may override them."""

    def _write(self, tmp_path, monkeypatch, body: str):
        (tmp_path / "airs.yaml").write_text(textwrap.dedent(body))
        monkeypatch.setattr(scope, "scope_dir", lambda: tmp_path)
        scope._load_file.cache_clear()

    def test_tasks_inherit_the_shared_keys(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, """
            resources: {gpus: 1, gpu_type: H200}
            budget: {official_h: 24, poc_h: 8}
            metric: {name: Accuracy, direction: higher_is_better}
            tasks:
              - id: A
              - id: B
        """)
        a, b = scope.load("airs")
        assert a.resources == b.resources == {"gpus": 1, "gpu_type": "H200"}
        assert a.metric["name"] == "Accuracy"

    def test_a_task_overrides_what_it_declares(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, """
            metric: {name: Accuracy, direction: higher_is_better}
            budget: {official_h: 24}
            tasks:
              - id: A
              - id: B
                metric: {name: MAE, direction: lower_is_better}
                budget: {official_h: 48}
        """)
        a, b = scope.load("airs")
        assert (a.metric["name"], b.metric["name"]) == ("Accuracy", "MAE")
        assert (a.budget["official_h"], b.budget["official_h"]) == (24, 48)

    def test_variants_multiply_the_run_count_and_the_gpu_hours(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, """
            resources: {gpus: 2}
            budget: {official_h: 10}
            metric: {name: Accuracy, direction: higher_is_better}
            variants: [m1, m2, m3, m4]
            tasks:
              - id: A
        """)
        (a,) = scope.load("airs")
        assert a.n_runs == 4
        assert a.gpu_hours == 10 * 2 * 4

    def test_a_task_with_no_variants_counts_as_one_run(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, """
            resources: {gpus: 1}
            budget: {official_h: 24}
            metric: {name: Accuracy, direction: higher_is_better}
            tasks:
              - id: A
        """)
        (a,) = scope.load("airs")
        assert (a.n_runs, a.gpu_hours) == (1, 24)

    def test_an_analysis_only_task_costs_no_gpu_hours(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, """
            resources: {gpus: 1}
            budget: {official_h: 10}
            metric: {name: Accuracy, direction: higher_is_better}
            tasks:
              - id: A
              - id: B
                self_run: false
        """)
        a, b = scope.load("airs")
        assert (a.self_run, a.gpu_hours) == (True, 10)
        assert (b.self_run, b.gpu_hours) == (False, 0.0)

    def test_gpu_hours_is_none_when_a_number_is_missing(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, """
            metric: {name: Accuracy, direction: higher_is_better}
            tasks:
              - id: A
        """)
        assert scope.load("airs")[0].gpu_hours is None


class TestMalformed:
    def _load(self, tmp_path, monkeypatch, body: str):
        (tmp_path / "airs.yaml").write_text(textwrap.dedent(body))
        monkeypatch.setattr(scope, "scope_dir", lambda: tmp_path)
        scope._load_file.cache_clear()
        return lambda: scope.load("airs")

    @pytest.mark.parametrize(
        "body, msg",
        [
            ("- id: A\n", "expected a mapping"),
            ("tasks:\n  - metric: {name: A, direction: higher_is_better}\n", "no id"),
            ("tasks:\n  - id: A\n", "no metric"),
            (
                "tasks:\n  - id: A\n    metric: {name: A, direction: up}\n",
                "metric.direction",
            ),
            (
                "metric: {name: A, direction: higher_is_better}\nverdict: selected\ntasks: []\n",
                "unknown top-level key",
            ),
            (
                "metric: {name: A, direction: higher_is_better}\n"
                "tasks:\n  - id: A\n    gates: {G1: pass}\n",
                "unknown key",
            ),
        ],
    )
    def test_rejects(self, tmp_path, monkeypatch, body, msg):
        load = self._load(tmp_path, monkeypatch, body)
        with pytest.raises(scope.ScopeError, match=msg):
            load()

    def test_check_reports_a_malformed_file_instead_of_raising(self, tmp_path, monkeypatch):
        self._load(tmp_path, monkeypatch, "tasks:\n  - id: A\n")
        problems = scope.check()
        assert problems and any("no metric" in p for p in problems)


class TestCheck:
    def test_the_registry_is_currently_consistent(self):
        assert scope.check() == []

    def test_it_catches_a_metric_that_drifted_from_upstream(self, monkeypatch):
        """The AIRS anchors are copies of upstream's; drift must be reported."""
        real = scope.load

        def wrong(benchmark=None):
            entries = list(real(benchmark))
            if benchmark == "airs":
                bad = dict(entries[0].metric, reference=entries[0].metric["reference"] + 1)
                entries[0] = type(entries[0])(
                    **{**entries[0].__dict__, "metric": bad}
                )
            return entries

        monkeypatch.setattr(scope, "load", wrong)
        problems = scope.check()
        assert any("upstream metadata.yaml says" in p for p in problems)

    def test_it_catches_a_count_that_no_longer_matches_the_document(self, monkeypatch):
        real = scope.load
        monkeypatch.setattr(
            scope, "load", lambda b=None: real(b)[:-1] if b == "airs" else real(b)
        )
        assert any("doc 3.4 claims" in p for p in scope.check())

    def test_it_catches_a_budget_row_that_no_longer_adds_up(self, monkeypatch):
        """Section 5.1 states 8 runs at 192 GPU-hours; both must follow from the file."""
        doc = scope._doc_text().replace("| 8 | ≈192 |", "| 8 | ≈999 |")
        monkeypatch.setattr(scope, "_doc_text", lambda: doc)
        assert any("GPU-hours" in p and "AIRS PoC" in p for p in scope.check())

    def test_it_catches_a_budget_row_that_disappeared(self, monkeypatch):
        doc = "\n".join(
            line for line in scope._doc_text().splitlines() if "AIRS PoC" not in line
        )
        monkeypatch.setattr(scope, "_doc_text", lambda: doc)
        assert any("no budget row" in p for p in scope.check())


class TestAgainstUpstream:
    """The numbers we copied must still be what upstream says."""

    UPSTREAM = scope.AIRS_UPSTREAM

    @pytest.mark.skipif(not UPSTREAM.is_dir(), reason="airs-bench submodule not checked out")
    def test_every_airs_task_exists_upstream(self):
        names = {p.name for p in self.UPSTREAM.iterdir() if p.is_dir()}
        assert {e.task for e in scope.load("airs")} <= names

    @pytest.mark.skipif(not UPSTREAM.is_dir(), reason="airs-bench submodule not checked out")
    def test_metric_anchors_match_metadata_yaml(self):
        for e in scope.load("airs"):
            meta = yaml.safe_load(
                (self.UPSTREAM / e.task / "metadata.yaml").read_text(encoding="utf-8")
            )
            info = meta["logging_info"]
            assert e.metric["name"] == info["metric"]
            assert e.metric["reference"] == info["sota"][0]["sota_score"]
            assert e.metric["s_min"] == info["estimated_worst_score"]
            assert e.metric["s_opt"] == info["optimal_score"]
            assert e.family == info["category"]


class TestContents:
    """What the registry currently holds, so an accidental deletion is loud."""

    def test_airs_holds_the_eight_gpu_heavy_tasks(self):
        assert len(scope.load("airs")) == 8

    def test_posttrainbench_is_one_poc_configuration(self):
        (e,) = scope.load("posttrainbench")
        assert e.task == "gsm8k"
        assert e.variants == ("Qwen/Qwen3-4B-Base",)
        assert e.n_runs == 1 and e.self_run

    def test_speedrun_is_one_task_anchored_on_the_human_record(self):
        (e,) = scope.load("speedrun_pi")
        assert e.metric["name"] == "train_steps"
        assert e.metric["direction"] == "lower_is_better"
        assert (e.metric["baseline"], e.metric["reference"]) == (3290, 2600)

    def test_summary_totals(self):
        totals = {name: (tasks, runs, hours) for name, tasks, runs, hours in scope.summary()}
        assert totals["airs"] == (8, 8, 192)
        assert totals["posttrainbench"] == (1, 1, 10)

    def test_scope_files_are_small_enough_to_read_in_one_sitting(self):
        """The registry replaced 3,000 lines of duplicated YAML; keep it that way."""
        total = sum(
            len((REPO_ROOT / "scope" / f"{name}.yaml").read_text().splitlines())
            for name in scope.BENCHMARKS
        )
        assert total < 200, f"scope/ has grown to {total} lines"

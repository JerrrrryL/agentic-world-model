"""The companion document exists so a reader can judge the recipe file without
re-deriving it. That only works if its numbers come from the file — a table that
drifts one repair round behind is worse than no table, because it reads as
audited. These tests hold the generator to recomputing, and hold onto the two
sentences a reader would be misled without.
"""

from __future__ import annotations

import re

from awm.analysis import evidence, report

SPEC = {
    "name": "gsm8k-gemma-holdout-v1",
    "benchmark": "gsm8k",
    "dataset": {
        "repo": "aisa-group/PostTrainBench-Trajectories",
        "repo_type": "dataset",
        "revision": "39d3fcd794df51c062c8bd3b7f8523ba707aaeb3",
        "catalog": "viewer_data/index.json",
        "catalog_sha256": "35d54c47",
    },
}


def record(run, *, fmt="claude-code", status="clean", worst=None, problems=(),
           anchors=None, algos=("sft",), lr=2e-5, accuracy=0.7, lenses=2,
           current=True, repair_round=None, never_returned=False):
    return {
        "run": run,
        "experiment": run.split("/")[0],
        "benchmark": "gsm8k",
        "trained_model": "Qwen/Qwen3-4B-Base",
        "agent_model": "claude-opus-4-8",
        "trace_format": fmt,
        "seed": 1,
        "time_budget_h": 10,
        "time_taken": 9.5,
        "pipeline": list(algos),
        "algorithms": [{"family": a, "name": a.upper()} for a in algos],
        "datasets": [{"dataset_id": "openai/gsm8k", "share": 1.0}],
        "hyperparams": {"lr": lr, "epochs": 3, "batch_size": None},
        "total_train_examples": 7473,
        "inference_tricks": [],
        "discarded": ["dpo"],
        "unresolved": [],
        "confidence": "high",
        "accuracy": accuracy,
        "stderr": 0.01,
        "total_cost_usd": 1.0,
        "num_turns": 100,
        "duration_ms": 1000,
        "extraction": {
            "status": status,
            "reviewed": True,
            "reviewed_version_is_the_one_here": current,
            "review_lenses": lenses,
            "worst_problem": worst,
            "problems": list(problems),
            "repair_round": repair_round,
            "repair_changed": None,
            "repair_disputed": None,
            "repair_never_returned": never_returned,
            "evidence_anchors": {"ok": 4} if anchors is None else anchors,
            "source_events": 500,
            "digest_events": 50,
            "digest_chars": 30_000,
        },
    }


RECORDS = [
    record("a/1"),
    record("a/2", status="reviewed-with-notes", worst="minor",
           problems=[{"field": "lr", "issue": "short quote", "severity": "minor", "lens": "evidence"}]),
    record("b/3", fmt="codex", status="flagged", worst="major", algos=("sft", "grpo"),
           problems=[{"field": "datasets", "issue": "the mixture shipped is not the one listed",
                      "severity": "major", "lens": "shipped"}],
           anchors={"ok": 3, "absent": 1}),
    record("b/4", fmt="codex", status="repaired-verified", repair_round=3, lr=None),
]


class TestFigures:
    """Every count in the prose has to be recomputable from the records."""

    def test_the_run_count_is_the_records_it_was_handed(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert f"# What these {len(RECORDS)} agents actually shipped" in md

    def test_the_anchor_total_is_the_sum_over_records(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        total = sum(sum(r["extraction"]["evidence_anchors"].values()) for r in RECORDS)
        assert f"{total} anchors total" in md

    def test_a_severity_absent_from_every_record_is_reported_as_zero_not_omitted(self) -> None:
        """A missing severity line would read as "not measured"; it was measured."""
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "0 fatal" in md

    def test_coverage_counts_a_null_field_as_missing(self) -> None:
        rows = dict((r[0], r[1]) for r in report.coverage_rows(RECORDS))
        assert rows["learning rate"] == 3        # one record has lr=None
        assert rows["batch size"] == 0           # every record has batch_size=None

    def test_the_pins_are_copied_from_the_spec_not_typed(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert SPEC["dataset"]["revision"] in md
        assert SPEC["dataset"]["catalog_sha256"] in md

    def test_two_families_with_the_same_count_always_come_out_in_the_same_order(self) -> None:
        """The counters are fed from ``set`` comprehensions, so insertion order
        follows the hash seed. Ties have to break on the name or the committed
        document stops matching a fresh render for no reason anyone can see."""
        recs = [record("a/1", algos=("zeta", "alpha"))]
        assert report._top(report.Counter(recs[0]["pipeline"]), 12) == [
            ["`alpha`", 1], ["`zeta`", 1]
        ]

    def test_a_truncated_table_says_how_much_it_dropped(self) -> None:
        counter = report.Counter({"a": 5, "b": 4, "c": 3, "d": 2})
        assert report._top(counter, 2)[-1] == ["*+2 more, 5 runs*", ""]

    def test_no_percentage_in_the_document_is_impossible(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert not [p for p in re.findall(r"\((\d+)%\)", md) if int(p) > 100]


class TestFlagged:
    def test_flagged_rows_are_named_so_a_reader_can_exclude_them(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "**1 row is `flagged`**" in md
        assert "- `b/3`" in md

    def test_with_nothing_flagged_it_says_so_rather_than_leaving_a_gap(self) -> None:
        """An empty section reads as an omission. It has to make the claim."""
        md = report.render([record("a/1")], SPEC, "x.jsonl")
        assert "No row is `flagged`" in md
        assert "survived the repair round" not in md

    def test_the_reason_shown_is_the_one_that_flagged_the_row(self) -> None:
        """A flagged row carries every note both lenses made, and the majority of
        them are minor. Printing whichever came back first puts a nit next to the
        word `flagged` and makes the exclusion look like reviewer pedantry."""
        md = report.render([record(
            "b/3", status="flagged", worst="major",
            problems=[
                {"field": "lr", "issue": "the quote is a little short", "severity": "minor",
                 "lens": "evidence"},
                {"field": "datasets", "issue": "ships a mixture the digest shows was abandoned",
                 "severity": "major", "lens": "shipped"},
            ],
        )], SPEC, "x.jsonl")
        assert "ships a mixture the digest shows was abandoned" in md
        assert "the quote is a little short" not in md

    def test_a_clipped_objection_says_it_was_clipped(self) -> None:
        """A hard slice at 200 characters ends mid-word, and a reader who cannot
        see the cut reads the stump as the reviewer's own sloppiness."""
        issue = "the mixture is wrong because " + "reason and " * 40
        md = report.render([record("b/3", status="flagged", worst="major", problems=[
            {"field": "datasets", "issue": issue, "severity": "major", "lens": "shipped"}])],
            SPEC, "x.jsonl")
        assert " …" in md
        assert "reaso\n" not in md and "reaso " not in md

    def test_clipping_never_leaves_a_backtick_open(self) -> None:
        """The one that actually shipped: the cut landed inside a code span, so
        the odd backtick swallowed the rest of the line into `code` and the
        reader blamed the data rather than the renderer."""
        issue = "x" * 190 + " `--learning-rate 5e-7 --epochs 3`"
        md = report.render([record("b/3", status="flagged", worst="major", problems=[
            {"field": "hyperparams", "issue": issue, "severity": "major", "lens": "evidence"}])],
            SPEC, "x.jsonl")
        line = next(ln for ln in md.splitlines() if "xxx" in ln)
        assert line.count("`") % 2 == 0, line

    def test_a_short_objection_is_left_exactly_as_written(self) -> None:
        """The clip has to be invisible when it does not fire, or every short
        objection grows an ellipsis that claims text nobody wrote."""
        md = report.render([record("b/3", status="flagged", worst="major", problems=[
            {"field": "datasets", "issue": "ships an abandoned mixture", "severity": "major",
             "lens": "shipped"}])], SPEC, "x.jsonl")
        assert "ships an abandoned mixture\n" in md or "ships an abandoned mixture" in md
        assert "ships an abandoned mixture …" not in md

    def test_it_separates_the_rows_no_repair_ever_reached(self) -> None:
        """A row whose repair agent never returned is flagged for a different
        reason than one the reviewers faulted twice, and only the second is a
        statement about the trajectory."""
        p = [{"field": "datasets", "issue": "not the shipped mixture", "severity": "major",
              "lens": "shipped"}]
        md = report.render([record("a/1", status="flagged", worst="major", problems=p,
                                   repair_round=4),
                            record("a/2", status="flagged", worst="major", problems=p,
                                   never_returned=True)],
                           SPEC, "x.jsonl")
        assert "1 were repaired first and faulted again" in md
        assert "1 carries the objection unrepaired" in md

    def test_a_repair_that_never_returned_is_not_a_repair(self) -> None:
        """The row keeps whichever text an earlier round wrote, so its
        ``repair_round`` is set — the one field that looks like it answers this
        question says the opposite of the truth."""
        p = [{"field": "datasets", "issue": "not the shipped mixture", "severity": "major",
              "lens": "shipped"}]
        md = report.render([record("a/1", status="flagged", worst="major", problems=p,
                                   repair_round=3, never_returned=True)], SPEC, "x.jsonl")
        assert "No repair pass ever returned a text for any of them" in md
        assert "Every one of them was repaired first" not in md

    def test_with_every_flagged_row_repaired_it_makes_the_plain_claim(self) -> None:
        p = [{"field": "datasets", "issue": "not the shipped mixture", "severity": "major",
              "lens": "shipped"}]
        md = report.render([record("a/1", status="flagged", worst="major", problems=p,
                                   repair_round=4)], SPEC, "x.jsonl")
        assert "Every one of them was repaired first" in md
        assert "unrepaired" not in md

    def test_a_fatal_outranks_a_major(self) -> None:
        md = report.render([record(
            "b/3", status="flagged", worst="fatal",
            problems=[
                {"field": "datasets", "issue": "a mixture that was abandoned", "severity": "major",
                 "lens": "shipped"},
                {"field": "algorithms", "issue": "the quote is not in the block it cites",
                 "severity": "fatal", "lens": "evidence"},
            ],
        )], SPEC, "x.jsonl")
        assert "**fatal** — the quote is not in the block it cites" in md


class TestDegenerateInputs:
    def test_a_record_with_no_anchors_does_not_divide_by_zero(self) -> None:
        md = report.render([record("a/1", anchors={})], SPEC, "x.jsonl")
        assert "0 anchors total" in md

    def test_a_format_with_no_anchors_reports_zero_over_zero(self) -> None:
        rows = report.by_format_rows([record("a/1", fmt="cursor", anchors={})])
        assert rows == [["cursor", 1, "0 (0%)", 50, "0/0"]]

    def test_a_record_with_no_accuracy_does_not_claim_a_range(self) -> None:
        md = report.render([record("a/1", accuracy=None)], SPEC, "x.jsonl")
        assert "0 carry an accuracy, n/a" in md

    def test_a_run_with_no_recognised_pipeline_is_named_not_dropped(self) -> None:
        md = report.render([record("a/1", algos=())], SPEC, "x.jsonl")
        assert "(no training stage)" in md


class TestTheCaveatsSurvive:
    """Both of these have been misread off a coverage table before. If an edit
    drops the sentence, the table starts lying by omission."""

    def test_it_says_a_blank_is_na_and_not_zero(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "A blank is NA, not zero" in md

    def test_it_says_which_round_did_the_repairing(self) -> None:
        """A bare total reads as "the extractor is unreliable". Which round did
        the work is what distinguishes that from "the reviewer got stricter"."""
        mixed = [record("a/1", repair_round=2), record("a/2", repair_round=5),
                 record("a/3", repair_round=5), record("a/4")]
        line = report.repair_line(mixed)
        assert "3 rows were repaired at least once (round 2: 1, round 5: 2" in line
        assert "not as the extraction getting worse" in line

    def test_a_row_repaired_twice_is_counted_once_but_still_said_out_loud(self) -> None:
        """``repair_round`` is the *last* repair. Reading the round tally alone,
        a row repaired at 2 and again at 4 is indistinguishable from one that was
        only ever touched at 4 — and those two rows were faulted by different
        reviewers to different standards."""
        once = record("a/1", repair_round=4)
        once["extraction"]["repair_rounds"] = [4]
        twice = record("a/2", repair_round=4)
        twice["extraction"]["repair_rounds"] = [2, 4]
        line = report.repair_line([once, twice])
        assert "2 rows were repaired at least once (round 4: 2" in line
        assert "1 of them were repaired twice" in line

    def test_without_the_history_field_it_claims_no_second_repair(self) -> None:
        """A row that predates ``repair_rounds`` must not be counted as repaired
        twice on the strength of a missing key."""
        assert "repaired twice" not in report.repair_line([record("a/1", repair_round=4)])

    def test_with_nothing_repaired_it_says_so_rather_than_an_empty_list(self) -> None:
        assert report.repair_line([record("a/1")]) == "No row needed repairing."

    def test_an_empty_pipeline_is_split_into_its_two_causes(self) -> None:
        """A blank row in the pipelines table reads as an extraction gap. For a
        run whose agent shipped untrained weights on purpose it is the opposite,
        and the difference is in ``algorithms[]``, not in the pipeline."""
        shipped_base = record("a/1", algos=())
        shipped_base["algorithms"] = [{"family": "other", "name": "copied the base weights"}]
        shipped_base["pipeline"] = []
        line = report.no_stage_line([shipped_base, record("b/2", algos=())])
        assert "1 where the row shows the agent shipping weights it did not train" in line
        assert "1 where the digest holds no training launch at all" in line
        assert "`a`" in line and "`b`" in line

    def test_with_no_blank_pipeline_it_adds_no_sentence(self) -> None:
        assert report.no_stage_line([record("a/1")]) == ""

    def test_it_says_confidence_tracks_the_review_and_not_only_the_run(self) -> None:
        """The repair pass demotes anything the digest does not settle, so a
        repaired row reads as less certain than an unexamined one about an
        equally uncertain trajectory. Read across that boundary and the column
        measures which rows a reviewer happened to fault."""
        shaky = record("a/1", repair_round=5)
        shaky["confidence"] = "medium"
        shaky["unresolved"] = ["which of v5 and v6 shipped", "the pool size"]
        line = report.review_depth_line([shaky, record("b/2")])
        assert "not comparable across rows" in line
        assert "1 repaired rows report `high` on 0%" in line
        assert "median 2 unresolved" in line
        assert "the 1 never faulted report `high` on 100%" in line

    def test_with_nothing_repaired_there_is_no_boundary_to_warn_about(self) -> None:
        """The warning is about a *comparison*. With one group it would be a
        claim about the runs, which is exactly the misreading it exists to stop."""
        assert report.review_depth_line([record("a/1"), record("a/2")]) == ""
        assert report.review_depth_line([record("a/1", repair_round=2)]) == ""

    def test_it_states_the_stopping_rule(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "The stopping rule:" in md
        assert "left `flagged` and named below" in md

    def test_the_repair_bound_is_read_out_of_the_file_not_typed(self) -> None:
        """A typed "one repair pass per record" was already false — nine rows
        went through two. The bound has to be recomputed, or the document keeps
        making a claim no reader can check and no test can catch."""
        twice = record("a/1", repair_round=4)
        twice["extraction"]["repair_rounds"] = [2, 4]
        thrice = record("a/2", repair_round=5)
        thrice["extraction"]["repair_rounds"] = [2, 4, 5]
        assert "no row was repaired more than 2 times — 1 reached that bound" in \
            report.stopping_rule_line([twice, record("a/3")])
        assert "no row was repaired more than 3 times — 1 reached that bound" in \
            report.stopping_rule_line([twice, thrice])

    def test_with_nothing_repaired_the_rule_makes_the_stronger_claim(self) -> None:
        assert "nothing was repaired" in report.stopping_rule_line([record("a/1")])

    def test_a_row_missing_the_history_still_counts_as_repaired_once(self) -> None:
        """The absent key means "we only recorded the last repair", not "there
        were none" — reading it as none would print "nothing was repaired" over
        a table that says otherwise."""
        line = report.stopping_rule_line([record("a/1", repair_round=3)])
        assert "no row was repaired more than 1 time" in line
        assert "nothing was repaired" not in line

    def test_it_says_the_score_was_joined_after_extraction(self) -> None:
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "the catalogue was not read until every row was" in md

    def test_it_does_not_claim_the_extractor_worked_blind(self) -> None:
        """The stronger sentence was in here for a while and it was false: the
        agents evaluate their own models inside the run and the digest keeps the
        tail of a result, so most digests do state a score. "Joined afterwards"
        is a fact about the join and says nothing about what the extractor saw."""
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "the digest the extractor read carries no score" not in md
        assert "most digests do state a score" in md

    def test_it_does_not_claim_the_evidence_rule_was_verified(self) -> None:
        """"Nothing was read from anywhere else" is the instruction the models
        were given. Stated as a property of the output it is refuted by the
        output itself — one row's `unresolved` quotes the harness's accuracy,
        which is nowhere in that row's digest."""
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "That is the instruction, not a proven property" in md

    def test_it_says_how_little_of_a_row_the_anchors_cover(self) -> None:
        """"Anchored quote by quote" reads as a property of the row and is a
        property of two of its fields. Without the denominator, a green anchor
        table is read as a green row."""
        line = report._coverage_line(RECORDS)
        assert "list entries in the file carry an anchor at all" in line
        assert "never an `evidence_quote`" in line

    def test_the_reproduce_instruction_names_a_real_entry_point(self) -> None:
        """The document tells a reader how to redo the anchor audit themselves.
        A signature that has since moved makes that instruction worse than none —
        it reads as verified and fails on the first attempt."""
        md = report.render(RECORDS, SPEC, "x.jsonl")
        assert "`awm.analysis.evidence.audit(row, digest_text)`" in md
        assert evidence.audit({"algorithms": [], "datasets": []}, "") == report.Counter()

    def test_every_status_present_in_the_data_is_explained(self) -> None:
        rows = report.status_rows(RECORDS)
        assert all(row[4] for row in rows), "a status shipped without a meaning"

    def test_the_method_paragraph_cannot_claim_a_coverage_the_table_denies(self) -> None:
        """"Two lenses per recipe" beside a table saying 13% is the drift this
        generator exists to prevent, and it is prose, so no count catches it."""
        mixed = [record("a/1", lenses=2), record("a/2", lenses=1)]
        md = report.render(mixed, SPEC, "x.jsonl")
        assert "for 1 of the 2 (the rest were read by a single verifier)" in md
        assert "Two adversarial lenses per recipe, each told to refute it." not in md

    def test_with_every_row_double_read_it_makes_the_plain_claim(self) -> None:
        md = report.render([record("a/1"), record("a/2")], SPEC, "x.jsonl")
        assert "Two adversarial lenses per recipe, each told to refute it." in md

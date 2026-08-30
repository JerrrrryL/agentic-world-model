"""Hold the shipped recipe file to the split it claims to describe.

``gsm8k-gemma-holdout-v1.recipes.jsonl`` is a committed artifact, not something
anybody regenerates before reading. So the ways it can go wrong are the ways a
committed file goes wrong: a row for a run that is not in the split, a row that
lost its evidence anchors in an edit, a status nothing knows how to interpret,
and — the quiet one — a companion document still quoting last round's numbers.

That last check compares the document byte-for-byte against a fresh render. It
is deliberately strict: the ``.md`` is generated, and the moment a hand-edit
survives in it, its tables stop being a measurement of the file beside it.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

from awm import splits
from awm.analysis import report
from awm.paths import splits_dir

SPLIT_ID = "posttrainbench/gsm8k-gemma-holdout-v1"
STEM = splits_dir() / SPLIT_ID

#: CJK ideographs plus the full-width punctuation that comes with them.
CJK = re.compile(r"[一-鿿　-〿＀-￯]")


@pytest.fixture(scope="module")
def split() -> splits.Split:
    return splits.load(SPLIT_ID)


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(STEM.with_suffix(".yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    text = STEM.with_name(f"{STEM.name}.recipes.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestItDescribesTheSplit:
    def test_it_is_exactly_the_train_side_in_order(self, rows, split) -> None:
        assert [r["run"] for r in rows] == list(split.train)

    def test_no_held_out_run_leaked_in(self, rows, split) -> None:
        """The point of the split is that the held-out runs are unseen. A recipe
        row for a test run leaks the answer into anything fitted on this file."""
        assert not {r["run"] for r in rows} & set(split.test)


class TestEveryRowIsCheckable:
    def test_every_algorithm_and_dataset_entry_carries_its_anchor(self, rows) -> None:
        """An entry with no ``evidence_i``/``evidence_quote`` cannot be re-checked
        against the trajectory, which makes it an assertion rather than a reading."""
        loose = [
            (r["run"], field, x.get("name"))
            for r in rows
            for field in ("algorithms", "datasets")
            for x in r[field]
            if x.get("evidence_i") is None or not x.get("evidence_quote")
        ]
        assert loose == []

    def test_the_outcome_was_joined_on(self, rows) -> None:
        """The digest the extractor read carries no score, so every row's accuracy
        arrives from the catalogue afterwards. A null means that join missed."""
        assert [r["run"] for r in rows if r.get("accuracy") is None] == []

    def test_every_status_is_one_the_document_can_explain(self, rows) -> None:
        unknown = {r["extraction"]["status"] for r in rows} - set(report.STATUS_MEANING)
        assert unknown == set()

    def test_a_row_reviewed_against_older_text_never_claims_a_verified_status(
        self, rows
    ) -> None:
        """``clean`` and ``repaired-verified`` both assert that a reviewer read the
        text in this row. If the verdict predates the last repair, it did not."""
        lying = [
            r["run"]
            for r in rows
            if r["extraction"]["status"] in ("clean", "repaired-verified")
            and not r["extraction"]["reviewed_version_is_the_one_here"]
        ]
        assert lying == []

    def test_flagged_rows_carry_the_problem_they_are_flagged_for(self, rows) -> None:
        """``flagged`` is the exclusion signal. A row flagged with nothing worse
        than a minor note would push a usable recipe out of everyone's filter."""
        thin = [
            r["run"]
            for r in rows
            if r["extraction"]["status"] == "flagged"
            and not {p.get("severity") for p in r["extraction"]["problems"]} & {"major", "fatal"}
        ]
        assert thin == []


class TestItIsAllInOneLanguage:
    """Every field of a row except the reviewer's prose is derived from the
    trajectory or the catalogue, so ``extraction`` is the only place the file can
    change language — and it did. The reviewers were agents running under a
    session configured to answer in Chinese, and roughly a fifth of every round's
    notes came back in it. A ``problems[].issue`` whose language depends on which
    agent happened to answer is unreadable to half its audience either way, and
    the mix is invisible unless something looks for it."""

    def _cjk(self, obj, path: str = "") -> list[tuple[str, str]]:
        if isinstance(obj, str):
            return [(path, obj)] if CJK.search(obj) else []
        if isinstance(obj, list):
            return [h for i, v in enumerate(obj) for h in self._cjk(v, f"{path}[{i}]")]
        if isinstance(obj, dict):
            # Keys as well as values. One reviewer returned its finding's title as
            # an extra KEY with an empty value, and a walker that only descends
            # values called that row clean English.
            return [h for k, v in obj.items()
                    for h in self._cjk(k, f"{path}.<key>") + self._cjk(v, f"{path}.{k}")]
        return []

    def test_no_row_carries_chinese_reviewer_prose(self, rows) -> None:
        hits = [(r["run"], p, s) for r in rows for p, s in self._cjk(r["extraction"], "extraction")]
        assert hits == [], (
            f"{len(hits)} field(s) across {len({h[0] for h in hits})} row(s) are still in Chinese"
            " — add them to en_map.json and re-run the assembler, do not hand-edit the artifact"
        )

    def test_the_check_would_notice(self) -> None:
        """The assertion above passes on an empty search as readily as on a clean
        file, so prove the search actually finds one."""
        assert self._cjk({"problems": [{"issue": "这条引文不在那一块里"}]}, "extraction") == [
            ("extraction.problems[0].issue", "这条引文不在那一块里")
        ]

    def test_the_check_would_notice_a_chinese_key(self) -> None:
        """The one that got through was a key, not a value: a reviewer emitted
        ``{"evidence_i未覆盖名称里的全部主张": ""}``. An empty value carries no
        Chinese at all, so only a walker that reads keys can see it."""
        planted = {"problems": [{"evidence_i未覆盖": "", "severity": "minor"}]}
        assert self._cjk(planted, "extraction") == [
            ("extraction.problems[0].<key>", "evidence_i未覆盖")
        ]


class TestTheDocumentIsGenerated:
    def test_it_is_byte_identical_to_a_fresh_render(self, rows, spec) -> None:
        path = STEM.with_name(f"{STEM.name}.recipes.md")
        fresh = report.render(rows, spec, f"{STEM.name}.recipes.jsonl")
        assert path.read_text(encoding="utf-8") == fresh, (
            "the companion document is stale or hand-edited — regenerate it rather "
            "than editing it, or its tables stop describing the file beside them"
        )

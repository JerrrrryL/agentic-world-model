"""Named data splits: the train/test contracts everyone runs against.

``splits/<source>/<name>.yaml`` is one complete, self-contained division of an
upstream release: which dataset at which pinned revision, the rule that decides
membership, and the materialized run lists the rule produced. The lists are
committed so a reader (and a diff) sees exact membership without running
anything; the rule is committed so :func:`check` can replay it against the
pinned catalogue and prove the lists still follow from it. Both live in the
same file on purpose — a split that cannot be re-derived is a liability, and a
rule without its outcome is not a contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from awm.paths import splits_dir

#: Fields of ``rule.require`` this module knows how to test, mapped to the
#: catalogue-row predicate that must hold. Unknown keys are errors: a typo in a
#: filter must never silently widen a split.
_REQUIRE = {
    "accuracy": lambda row, want: (row.get("accuracy") is not None) == (want == "present"),
    "contamination_flagged": lambda row, want: row.get("contamination", {}).get("flagged", False)
    == want,
    "disallowed_flagged": lambda row, want: row.get("disallowed_model", {}).get("flagged", False)
    == want,
}


class SplitError(ValueError):
    """A split file or rule is malformed."""


@dataclass(frozen=True)
class Split:
    """One run-level train/test contract, exactly as committed."""

    id: str
    dataset: dict[str, Any]
    benchmark: str
    rule: dict[str, Any]
    train: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "test": len(self.test)}


@dataclass(frozen=True)
class Selection:
    """A task-level choice: which upstream tasks are in play, and on what box."""

    id: str
    benchmark: str
    tasks: tuple[str, ...]
    resources: dict[str, Any]
    budget: dict[str, Any]


def list_ids() -> list[str]:
    """Every committed ``<source>/<name>`` id, sorted."""
    root = splits_dir()
    return sorted(f"{p.parent.name}/{p.stem}" for p in root.glob("*/*.yaml"))


def _read(split_id: str, kind: str) -> dict[str, Any]:
    path = splits_dir() / f"{split_id}.yaml"
    if not path.exists():
        raise SplitError(f"no split file at {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SplitError(f"{split_id}: expected a mapping")
    if doc.get("kind") != kind:
        raise SplitError(f"{split_id}: kind is {doc.get('kind')!r}, this loader wants {kind!r}")
    if doc.get("name") != split_id.rsplit("/", 1)[-1]:
        raise SplitError(f"{split_id}: name is {doc.get('name')!r}, must match the file stem")
    return doc


def _check_keys(split_id: str, doc: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(doc) - allowed
    if unknown:
        raise SplitError(f"{split_id}: unknown top-level key(s) {sorted(unknown)}")


def load(split_id: str) -> Split:
    """One committed run split by its ``<source>/<name>`` id."""
    doc = _read(split_id, "run-split")
    _check_keys(split_id, doc, {"kind", "name", "dataset", "benchmark", "rule", "counts", "splits"})
    parts = doc.get("splits") or {}
    if set(parts) != {"train", "test"}:
        raise SplitError(f"{split_id}: splits must hold exactly train and test")
    train = tuple(parts["train"] or ())
    test = tuple(parts["test"] or ())
    for side, runs in (("train", train), ("test", test)):
        if len(set(runs)) != len(runs):
            raise SplitError(f"{split_id}: duplicate run in {side}")
        want = (doc.get("counts") or {}).get(side)
        if want != len(runs):
            raise SplitError(f"{split_id}: counts.{side} says {want}, the list holds {len(runs)}")
    both = set(train) & set(test)
    if both:
        raise SplitError(f"{split_id}: {sorted(both)} in both train and test")
    return Split(
        id=split_id,
        dataset=doc.get("dataset") or {},
        benchmark=doc["benchmark"],
        rule=doc.get("rule") or {},
        train=train,
        test=test,
    )


def check(
    split: Split, catalog: dict[str, Any], catalog_bytes: bytes | None = None
) -> list[str]:
    """Everything that stops the committed lists following from the rule.

    ``catalog_bytes`` lets the caller prove the catalogue it replayed is the
    pinned one; without it (or without a recorded ``catalog_sha256``) the replay
    still runs, it just cannot vouch for the catalogue itself.
    """
    problems: list[str] = []
    want_sha = split.dataset.get("catalog_sha256")
    if want_sha and catalog_bytes is not None:
        import hashlib

        got = hashlib.sha256(catalog_bytes).hexdigest()
        if got != want_sha:
            problems.append(
                f"{split.id}: catalog sha256 is {got}, the split pins {want_sha} — "
                "the local catalogue is not the one the split was built from"
            )
    replayed = apply_rule(split.benchmark, split.rule, catalog)
    for side, committed in (("train", split.train), ("test", split.test)):
        missing = sorted(set(replayed[side]) - set(committed))
        extra = sorted(set(committed) - set(replayed[side]))
        for run in missing:
            problems.append(f"{split.id}: rule puts {run} in {side}, the list omits it")
        for run in extra:
            problems.append(f"{split.id}: {side} lists {run}, the rule does not produce it")
    return problems


def load_selection(split_id: str) -> Selection:
    """One committed task selection by its ``<source>/<name>`` id."""
    doc = _read(split_id, "task-selection")
    _check_keys(split_id, doc, {"kind", "name", "benchmark", "resources", "budget", "tasks"})
    tasks = tuple(doc.get("tasks") or ())
    if not tasks:
        raise SplitError(f"{split_id}: a selection with no tasks selects nothing")
    return Selection(
        id=split_id,
        benchmark=doc["benchmark"],
        tasks=tasks,
        resources=doc.get("resources") or {},
        budget=doc.get("budget") or {},
    )


def apply_rule(benchmark: str, rule: dict[str, Any], catalog: dict[str, Any]) -> dict[str, list[str]]:
    """Replay ``rule`` over a catalogue, returning sorted run paths per part.

    A run path is ``<experiment>/<run_name>`` — the run's directory in the
    upstream release, so membership doubles as a download address. ``rule.note``
    is prose for the reader and never affects membership.
    """
    unknown = set(rule) - {"by", "test", "require", "note"}
    if unknown:
        raise SplitError(f"rule has unknown key(s) {sorted(unknown)}")
    if rule.get("by") != "trained_model":
        raise SplitError(f"rule.by is {rule.get('by')!r}; only 'trained_model' is supported")
    unknown = set(rule.get("require", {})) - set(_REQUIRE)
    if unknown:
        raise SplitError(f"rule.require has unknown key(s) {sorted(unknown)}")

    heldout = set(rule.get("test", ()))
    parts: dict[str, list[str]] = {"train": [], "test": []}
    for row in catalog["runs"]:
        if row.get("benchmark") != benchmark:
            continue
        if not all(_REQUIRE[k](row, want) for k, want in rule.get("require", {}).items()):
            continue
        side = "test" if row["trained_model"] in heldout else "train"
        parts[side].append(f"{row['experiment']}/{row['run_name']}")
    return {side: sorted(runs) for side, runs in parts.items()}

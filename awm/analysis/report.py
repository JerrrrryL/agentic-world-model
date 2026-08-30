"""Render the recipe file's companion document from the recipe file itself.

The document beside a dataset is where its numbers go stale. Coverage shifts
when the extractor is re-run, the review counts shift every repair round, and a
hand-written table records whichever pass the author happened to be looking at.
So nothing here is typed: :func:`render` recomputes every figure from the
records it is handed, and the generator is the only way the ``.md`` is written.

Two things the prose has to keep saying, because both are easy to misread off a
coverage table:

*   **Absent is NA, never 0.** ``lr: null`` means the digest did not state a
    learning rate, not that the run trained without one. Every "n/143" below is
    "the digest said so and a reviewer let it stand", which is a lower bound on
    what the run actually did.
*   **A review count is a measurement of the reviewer too.** The two-lens pass
    faults far more than the single verifier did, over the same recipes. That is
    the lens getting stricter, not the data getting worse, and reporting the two
    side by side without saying so would read as a regression.
"""

from __future__ import annotations

from collections import Counter

#: Ordered so the document's coverage table reads from the fields a recipe is
#: useless without down to the ones only a thorough run records.
COVERAGE_FIELDS = [
    ("algorithm", lambda r: bool(r["algorithms"])),
    ("dataset", lambda r: bool(r["datasets"])),
    ("learning rate", lambda r: r["hyperparams"].get("lr") is not None),
    ("epochs", lambda r: r["hyperparams"].get("epochs") is not None),
    ("batch size", lambda r: r["hyperparams"].get("batch_size") is not None),
    ("total train examples", lambda r: r["total_train_examples"] is not None),
    ("a per-dataset share", lambda r: any(d.get("share") is not None for d in r["datasets"])),
    ("an inference-time trick", lambda r: bool(r["inference_tricks"])),
    ("something discarded", lambda r: bool(r["discarded"])),
]

#: What each ``extraction.status`` value licenses a reader to assume. Deliberately
#: silent about how many lenses read it — that varies within a status, so the
#: table carries it as its own column rather than letting the prose average it.
STATUS_MEANING = {
    "clean": "reviewed against this exact text; nothing found",
    "reviewed-with-notes": "reviewed, only minor notes — read `extraction.problems` before quoting a number",
    "repaired-verified": "was faulted, was repaired, and the repaired text was re-reviewed clean of major/fatal",
    "repaired": "was repaired, but no reviewer has read the repaired text — treat as unreviewed",
    "flagged": "a reviewer found something major or fatal in this exact text and it was not fixed",
    "unreviewed": "extracted, never reviewed",
}

ANCHOR_MEANING = {
    "ok": "verbatim inside the block it cites",
    "elided": "two real spans of the cited block joined by `...`, in order",
    "wrong-block": "the text is in the digest, but not at the cited event",
    "absent": "nowhere in the digest — a fabricated quote",
    "too-short": "under 8 characters, so it anchors nothing either way",
    "no-anchor": "the entry carries no quote or no event index",
}


def _pct(n: int, d: int) -> str:
    return "n/a" if not d else f"{n / d:.0%}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _median(xs: list[int]) -> int:
    s = sorted(xs)
    return s[len(s) // 2] if s else 0


def coverage_rows(records: list[dict]) -> list[list[str]]:
    n = len(records)
    return [[label, sum(1 for r in records if hit(r)), _pct(sum(1 for r in records if hit(r)), n)]
            for label, hit in COVERAGE_FIELDS]


def _span(xs: list[int]) -> str:
    return "n/a" if not xs else (str(xs[0]) if min(xs) == max(xs) else f"{min(xs)}–{max(xs)}")


def status_rows(records: list[dict]) -> list[list[str]]:
    counts = Counter(r["extraction"]["status"] for r in records)
    order = [s for s in STATUS_MEANING if s in counts] + \
            [s for s in counts if s not in STATUS_MEANING]
    return [[f"`{s}`", counts[s], _pct(counts[s], len(records)),
             _span([r["extraction"]["review_lenses"] for r in records
                    if r["extraction"]["status"] == s]),
             STATUS_MEANING.get(s, "")] for s in order]


def anchor_rows(records: list[dict]) -> tuple[list[list[str]], int]:
    tally: Counter[str] = Counter()
    for r in records:
        tally.update(r["extraction"]["evidence_anchors"])
    total = sum(tally.values())
    order = [k for k in ANCHOR_MEANING if k in tally] + \
            [k for k in tally if k not in ANCHOR_MEANING]
    return ([[f"`{k}`", tally[k], f"{tally[k] / total:.2%}" if total else "n/a",
              ANCHOR_MEANING.get(k, "")] for k in order], total)


def by_format_rows(records: list[dict]) -> list[list[str]]:
    """Extraction quality per wire format — four scaffolds, four ways to be unparseable."""
    fmts = sorted({r["trace_format"] for r in records})
    rows = []
    for f in fmts:
        sub = [r for r in records if r["trace_format"] == f]
        bad = sum(1 for r in sub if r["extraction"]["worst_problem"] in ("major", "fatal"))
        anchors = Counter()
        for r in sub:
            anchors.update(r["extraction"]["evidence_anchors"])
        tot = sum(anchors.values())
        rows.append([f, len(sub), f"{bad} ({_pct(bad, len(sub))})",
                     _median([r["extraction"]["digest_events"] for r in sub]),
                     f"{anchors['ok']}/{tot}" if tot else "0/0"])
    return rows


def _digest_line(ext: list[dict]) -> str:
    return (f"Median {_median([e['source_events'] for e in ext]):,} source events → "
            f"{_median([e['digest_events'] for e in ext]):,} kept, "
            f"{_median([e['digest_chars'] for e in ext]):,} characters; worst case "
            f"{max(e['digest_chars'] for e in ext):,}. No run was reduced to nothing.")


def repair_line(records: list[dict]) -> str:
    """When the repairs happened, and what that says about the reviewer.

    A single "27 rows were repaired" hides the thing a reader most needs: the
    rounds did not apply the same standard. Saying which round did the work is
    the difference between "the extractor is unreliable" and "the last reviewer
    was stricter than the first", and only one of those is true.

    ``repair_round`` names the *last* repair, so on its own it also hides the
    rows that were repaired twice. Those rows are the ones the stopping rule is
    about, so the count comes from ``repair_rounds`` where a row carries it.
    """
    ext = [r["extraction"] for r in records]
    rounds = Counter(e["repair_round"] for e in ext if e["repair_round"])
    if not rounds:
        return "No row needed repairing."
    parts = ", ".join(f"round {k}: {rounds[k]}" for k in sorted(rounds))
    twice = sum(1 for e in ext if len(e.get("repair_rounds") or []) > 1)
    again = (f" {twice} of them were repaired twice: a single verifier faulted the extraction,"
             " and the two-lens pair then faulted the repair." if twice else "")
    return (f"{sum(rounds.values())} rows were repaired at least once ({parts}, counting the last"
            f" repair each).{again} The rounds did"
            " not apply the same standard — the later ones put two adversarial lenses on records"
            " a single verifier had already passed, and that alone accounts for most of the"
            " repairs. Read a rising repair count as the review getting stricter, not as the"
            " extraction getting worse.")


def no_stage_line(records: list[dict]) -> str:
    """Say what an empty ``pipeline`` means, because it means two opposite things.

    A blank row in the pipelines table reads as "the extractor found nothing",
    and for some runs that is right. For others it is the opposite finding: the
    agent shipped weights it never trained — copied the base model across, or
    ran a no-op pass and merged it — and the row records that explicitly. The
    two are told apart structurally, not by confidence: a run with an empty
    pipeline but a non-empty ``algorithms[]`` has a stage the normaliser refused
    to call training, whereas an empty ``algorithms[]`` means nothing was read
    out of the digest at all.
    """
    blank = [r for r in records if not r["pipeline"]]
    if not blank:
        return ""
    told = [r for r in blank if r["algorithms"]]
    silent = [r for r in blank if not r["algorithms"]]
    names = lambda rs: ", ".join("`" + r["run"].split("/")[0] + "`" for r in rs)  # noqa: E731
    bits = []
    if told:
        bits.append(
            f"{len(told)} where the row shows the agent shipping weights it did not train — the"
            " stages are there in `algorithms[]`, none of them is a training family"
            f" ({names(told)})"
        )
    if silent:
        bits.append(
            f"{len(silent)} where the digest holds no training launch at all ({names(silent)})"
        )
    scored = [r["accuracy"] for r in blank if r.get("accuracy") is not None]
    span = (f"{min(scored):.3f}" if min(scored) == max(scored)
            else f"{min(scored):.3f}–{max(scored):.3f}") if scored else ""
    tail = ((f" It still carries a score ({span}):" if len(blank) == 1 else
             f" All {len(blank)} still carry a score ({span}):")
            + " the benchmark grades the submitted model, not the training."
            if len(scored) == len(blank) and scored else "")
    head = ("The one blank pipeline is not the finding it looks like: " if len(blank) == 1 else
            f"The blank pipeline is {len(blank)} runs, and it is not one finding: ")
    return (head + "; ".join(bits) + f".{tail} Read `confidence` and `unresolved[]` on those rows"
            " before counting them as extraction failures.")


def stopping_rule_line(records: list[dict]) -> str:
    """State the bound on repairs as a number read out of the file.

    "One repair pass per record" is the kind of claim that is written once and
    then quietly stops being true — the rounds here in fact repaired nine rows
    twice. A reader has no way to check a typed claim, and ``repair_round``
    alone (the *last* repair) cannot contradict it. So the bound is recomputed
    from ``repair_rounds`` and the prose is built around whatever comes out.
    """
    # A row with no history field is not a row with no repairs — fall back to
    # what ``repair_round`` still proves, so a missing key understates the depth
    # rather than erasing the repair.
    depth = [len(e["repair_rounds"]) if e.get("repair_rounds") is not None
             else (1 if e["repair_round"] else 0)
             for e in (r["extraction"] for r in records)]
    most = max(depth, default=0)
    if not most:
        return ("**The stopping rule:** nothing was repaired, so nothing was fitted to a"
                " reviewer.")
    n = sum(1 for d in depth if d == most)
    return (
        f"**The stopping rule:** no row was repaired more than {most}"
        f" time{'s' if most != 1 else ''} — {n} reached that bound — and no row was repaired"
        " again once the two-lens pair had read a text produced by that pair's own objection."
        " Whatever it still faults there is left `flagged` and named below, not repaired a"
        " further time. Iterating until the reviewers stop objecting would be fitting the data"
        " to the reviewer, and the number that came out of it would mean nothing."
    )


def review_depth_line(records: list[dict]) -> str:
    """``confidence`` records how hard the row was argued with, not only the run.

    A faulted row went through a repair pass whose whole job was to move claims
    the digest does not support out of ``algorithms``/``datasets`` and into
    ``discarded``/``unresolved``, and to lower ``confidence`` wherever the
    trajectory does not settle which run shipped. A row nobody faulted never had
    that done to it. So the two columns are not one scale: read across the
    repaired/never-faulted boundary and what varies is the review, not the agent.
    The document has to say where that boundary runs, because the field names
    give no hint of it.
    """
    rep = [r for r in records if r["extraction"]["repair_round"]]
    plain = [r for r in records if not r["extraction"]["repair_round"]]
    if not rep or not plain:
        return ""
    hi = lambda rs: _pct(sum(1 for r in rs if r["confidence"] == "high"), len(rs))  # noqa: E731
    un = lambda rs: _median([len(r["unresolved"] or []) for r in rs])  # noqa: E731
    return (
        f"**`confidence` and `unresolved[]` are not comparable across rows.** The {len(rep)}"
        f" repaired rows report `high` on {hi(rep)} of themselves and carry a median"
        f" {un(rep)} unresolved notes; the {len(plain)} never faulted report `high` on"
        f" {hi(plain)} and carry a median {un(plain)}. That gap is the repair pass, not the"
        " runs: repairing a row means demoting whatever the digest does not settle, and a row"
        " no reviewer objected to was never asked to do that. Compare within"
        " `extraction.repair_round`, not across it, and treat the repaired rows' figures as"
        " the honest ones."
    )


def _lens_line(two_lens: int, n: int) -> str:
    """The method paragraph must not claim a coverage the status table contradicts."""
    scope = ("Two adversarial lenses per recipe" if two_lens == n else
             f"Two adversarial lenses per recipe for {two_lens} of the {n}"
             f" (the rest were read by a single verifier)")
    return f"{scope}, each told to refute it."


def _why_flagged(record: dict) -> str:
    """The problem that flagged the row, not the first one recorded.

    ``problems`` is the concatenation of both lenses' findings in the order the
    lenses returned, and a flagged row usually carries a dozen minor notes
    alongside the one major finding that flagged it. Printing ``problems[0]``
    therefore prints a nit next to the word "flagged" most of the time, which
    reads as an over-strict reviewer rather than as the exclusion it is.
    """
    ranked = {"fatal": 0, "major": 1, "minor": 2}
    worst = min(record["extraction"]["problems"],
                key=lambda p: ranked.get(p.get("severity"), 3))
    return f'**{worst.get("severity")}** — {worst["issue"][:200]}'


def _flagged_split(flagged: list[dict]) -> str:
    """"Survived the repair round" is not true of every flagged row.

    Most were repaired and faulted again, which is a statement about how hard the
    row is. A few were never repaired at all — the repair agent died on them
    every time — which is a statement about the apparatus. Both end up `flagged`
    and both should be excluded, but only the first kind is evidence about the
    trajectory, and a sentence that merges them overstates the review.

    Keyed on ``repair_never_returned``, not on ``repair_round``. A row whose
    repair never came back still ships the text some earlier round wrote, so its
    ``repair_round`` is set and truthy — indistinguishable, from this side, from
    a row that was repaired and faulted again. Only the round that asked for the
    repair knows the difference, so it records it.
    """
    never = [r for r in flagged if r["extraction"].get("repair_never_returned")]
    fixed = [r for r in flagged if not r["extraction"].get("repair_never_returned")]
    if not never:
        return "Every one of them was repaired first, and faulted again on the repaired text."
    if not fixed:
        return ("No repair pass ever returned a text for any of them: the objection each one"
                " carries is against text no repair round replaced.")
    return (f"{len(fixed)} were repaired first and faulted again on the repaired text;"
            f" {len(never)} carr{'ies' if len(never) == 1 else 'y'} the objection unrepaired,"
            " because the repair round asked for a replacement text and never got one back. The"
            " second kind says nothing about the trajectory — it is a hole in the apparatus, and"
            " it is marked the same way because the row is equally unusable either way.")


def _top(counter: Counter, n: int, blank: str = "(none)") -> list[list[str]]:
    """Truncate, but never silently: a dropped tail reads as "that's all of them".

    Ties break on the name. ``most_common`` breaks them on insertion order, and
    these counters are fed from per-record ``set`` comprehensions, so insertion
    order follows the hash seed — the same records would render two different
    tables in two processes and the document would never match a fresh build.
    """
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    rows = [[f"`{k}`" if k else blank, v] for k, v in ranked]
    rest = len(counter) - len(rows)
    if rest:
        rows.append([f"*+{rest} more, {sum(counter.values()) - sum(r[1] for r in rows)} runs*", ""])
    return rows


def render(records: list[dict], spec: dict, jsonl_name: str) -> str:
    """Build the companion markdown. Every figure is recomputed from ``records``."""
    n = len(records)
    ds = spec["dataset"]
    ext = [r["extraction"] for r in records]
    anchors, anchor_total = anchor_rows(records)
    two_lens = sum(1 for e in ext if e["review_lenses"] >= 2)
    on_shipped = sum(1 for e in ext if e["reviewed_version_is_the_one_here"])
    sev = Counter(p.get("severity") for e in ext for p in e["problems"])
    flagged = [r for r in records if r["extraction"]["status"] == "flagged"]

    pipelines = Counter("→".join(r["pipeline"]) for r in records)
    families = Counter(f for r in records for f in {a["family"] for a in r["algorithms"]})
    datasets = Counter(d for r in records for d in {y["dataset_id"] for y in r["datasets"]})
    scored = [r["accuracy"] for r in records if r.get("accuracy") is not None]

    return f"""# What these {n} agents actually shipped

One line per **train** run of `{spec['name']}`: the post-training recipe the agent
actually shipped — which data, in what mixture, trained with which algorithm at which
hyper-parameters — read out of that run's own trajectory and anchored to it quote by quote.

The split itself is `{spec['name']}.yaml`; this file adds nothing to its membership and
changes nothing about it. It is a *reading* of the {n} train runs, and it is only as good as
the audit reported below — which is why the audit is here and not in a commit message.

`{jsonl_name}` — one JSON object per line, `run` is the key, same order as the YAML's
`splits.train`.

## Where it comes from

| | |
|---|---|
| dataset | `{ds['repo']}` ({ds['repo_type']}) |
| revision | `{ds['revision']}` |
| catalogue | `{ds['catalog']}`, sha256 `{ds['catalog_sha256']}` |
| benchmark | `{spec['benchmark']}` |
| population | the {n} runs in `splits.train`, unchanged |

Both pins are the split's, copied here so this file can be checked on its own. Nothing was
read from anywhere else: the only admissible evidence for a claim in a row is that run's own
event stream at that revision.

## One row

Catalogue facts, copied — `run`, `experiment`, `benchmark`, `trained_model`, `agent_model`,
`trace_format`, `seed`, `time_budget_h`, `time_taken`.

The recipe, extracted — `pipeline` (the normalised stage order, e.g. `sft→rft→grpo`),
`algorithms[]`, `datasets[]`, `hyperparams`, `total_train_examples`, `inference_tricks[]`,
`discarded[]` (what the agent tried and abandoned), `unresolved[]` (what the trajectory does
not settle), `confidence`.

The outcome, joined on afterwards — `accuracy`, `stderr`, `total_cost_usd`, `num_turns`,
`duration_ms`. **Afterwards** is load-bearing: the digest the extractor read carries no score
at all, so it could not describe a recipe as good because it knew the number was good.

The audit, per row — `extraction.{{status, problems, evidence_anchors, repair_round, ...}}`.

Every `algorithms[]` and `datasets[]` entry carries `evidence_i` (an event index in the full
stream) and `evidence_quote` (text from that event). That pair is what makes a row checkable
rather than merely plausible.

## How it was built

1. **Filter.** Each run's event stream is cut to the events that can carry a recipe —
   training scripts, launch commands, the agent's own statements about mixture and method,
   and the tail of any result that directly follows one. Four scaffolds name the same action
   four ways (`Bash` / `command_execution` / `shell` / `bash`), so the vocabularies are
   normalised first; a filter written against one of them keeps nothing for the other three.
   The budget is spent from the end backwards, because an agent's last hour is the run it
   submits. {_digest_line(ext)}
2. **Extract.** One model per run, reading only that digest, told that absent is null and
   that every claim needs a verbatim quote.
3. **Review.** {_lens_line(two_lens, n)} *Evidence fidelity*
   checks that every quote is really in the block it cites and that every non-null number is
   stated in the digest rather than defaulted. *Shipped, not tried* reads the end of the
   trajectory independently and asks whether the row describes the run that was submitted.
   A recipe is faithful only if both lenses agree.
4. **Repair, then re-review.** Anything faulted major or fatal was repaired against the digest
   and read again by both lenses.

{stopping_rule_line(records)}

## What the audit measured

{_table(["status", "runs", "share", "lenses", "what it licenses"], status_rows(records))}

{two_lens}/{n} ({_pct(two_lens, n)}) were read by both lenses; {on_shipped}/{n}
({_pct(on_shipped, n)}) carry a verdict against the exact text in the row rather than against
a version that was later repaired. {repair_line(records)}

{review_depth_line(records)}

Problems recorded across all rows: {sev.get('minor', 0)} minor, {sev.get('major', 0)} major,
{sev.get('fatal', 0)} fatal. Minor notes are kept rather than cleared — most are "this quote is
short" or "this field is defensible but under-evidenced", and a reader checking a specific
number is better served by the note than by its absence.

### The evidence anchors, checked without a model

An agent *saying* it grepped is not a grep. Every `evidence_quote` in the file was re-checked
in code against the digest block it cites — whitespace collapsed and a short closed set of
look-alike characters folded to ASCII, no fuzzy matching, no edit distance:

{_table(["verdict", "anchors", "share", "meaning"], anchors)}

{anchor_total} anchors total. Reproduce with `awm.analysis.evidence.audit(row, digest_text)`;
`tests/test_evidence.py` pins the checker against paraphrase, out-of-order elision and
cross-block quotes, because a lenient checker would turn this table into a restatement of the
extractor's own confidence.

### By wire format

{_table(["trace_format", "runs", "major/fatal", "median digest events", "anchors ok"], by_format_rows(records))}

## What is in the corpus

{n} runs, {sum(len(r['algorithms']) for r in records)} training stages,
{sum(len(r['datasets']) for r in records)} dataset entries,
{sum(len(r['inference_tricks']) for r in records)} inference-time tricks,
{sum(len(r['discarded']) for r in records)} abandoned attempts.
{len(scored)} carry an accuracy, {f'{min(scored):.3f}–{max(scored):.3f}' if scored else 'n/a'}.

**Pipelines** (normalised stage order, by run)

{_table(["pipeline", "runs"], _top(pipelines, 10, blank="(no training stage)"))}

{no_stage_line(records)}

**Algorithm families** (a run counts once per family, not once per stage)

{_table(["family", "runs"], _top(families, 12))}

**Data sources** (by run)

{_table(["dataset_id", "runs"], _top(datasets, 12))}

## Field coverage

{_table(["field", "runs", "share"], coverage_rows(records))}

**A blank is NA, not zero.** `lr: null` means the trajectory did not state a learning rate,
not that the run trained without one. Every share above is a lower bound on what the run
actually did — it measures what the agent wrote down and a reviewer let stand.

## What this file can and cannot be used for

It can: describe what recipes this population of agents converged on; pair a recipe with the
score it got, for a predictor that reads recipes rather than agent names; find the runs that
tried a given method; supply `discarded[]` as negatives.

It cannot: stand in for the trajectories — every row is a lossy reading of one, and the digest
that produced it dropped most of the stream by design. It also cannot be read as ground truth
about the training that happened: it is what the agent said it did, checked against what the
agent's own log shows, which is not the same as what the GPUs did.

{f'''**{len(flagged)} row{"s are" if len(flagged) != 1 else " is"} `flagged`** — a reviewer found something major or fatal in the exact
text shipped here and it is still there. {_flagged_split(flagged)} They are in the file rather
than dropped, because dropping them would make the audit look cleaner than the data is. Filter on
`extraction.status != "flagged"` if a clean subset is what you need:

''' + "\n".join(f'- `{r["run"]}` — {_why_flagged(r)}' for r in flagged)
   if flagged else '**No row is `flagged`**: every record carries a verdict, against its own text, with no major or fatal problem left standing.'}

## Regenerating

The recipe file and this document are both generated; neither is hand-edited. `render()` in
`awm/analysis/report.py` recomputes every figure above from the records themselves, so a
number here cannot drift from the file it describes.
"""

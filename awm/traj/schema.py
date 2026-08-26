"""The unified event schema (v0) and its on-disk form.

Every trajectory we analyse — public or self-run, whatever harness produced it —
is converted into this shape once, and the analysis layer reads nothing else.

The four event types come from Prime Intellect's release, which already
normalised nine harnesses (claude-code, codex, prime-agent, kimi-code, grok-cli,
qwen-code, pi, and two goal-driver variants) into the same stream. Adopting it
means the 41 PI runs need no conversion at all, and the shape is known to
survive contact with the other harnesses.

On disk, per run:
    events/<source>/<run_id>.jsonl.gz    one Event per line, main and sub agents
    events/<source>/<run_id>.meta.json   one RunMeta

Conventions a converter must honour:

*   ``i`` numbers each ``(run_id, agent_id)`` stream separately, from 0, in
    chronological order. Sub-agent events live in the same file, distinguished
    by ``agent_id``.
*   ``usage`` is recorded once per ``turn``, on the first event of that turn.
    Harnesses that repeat it on every event of a turn must drop the repeats,
    or token totals come out inflated by the number of events per turn.
*   ``origin`` separates what the agent did from what the scaffolding did to it.
    Launcher-injected ``continue`` prompts, goal re-injections, compaction
    handoffs and task notifications are ``harness``; they are not decisions and
    must not be counted as such.
*   ``ts`` is optional. Some releases carry no timestamps at all (the
    PostTrainBench codex runs, PI's grok-4.5 run), so ordering is ``i``, never
    time. Never synthesise a timestamp.
*   Text that the upstream release truncated is flagged with ``truncated``
    rather than silently kept, so a tool result cut to 1.5 kB is never mistaken
    for the whole thing.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "hv-events-v0"

EVENT_TYPES = ("text", "thinking", "tool_use", "tool_result")
ROLES = ("user", "assistant")
ORIGINS = ("agent", "harness", "human")

MAIN_AGENT = "main"

#: Canonical token-usage keys. Converters map each harness's spelling onto these.
#:
#: Whether ``in`` includes ``cache_read`` is the *harness's* convention, not ours,
#: and converters preserve it rather than trying to reconcile: OpenAI counts cached
#: input inside ``input_tokens`` while Anthropic reports it separately, so a codex
#: run sums to roughly twice its billed total and a Claude Code run to roughly its
#: own. Summing ``usage.values()`` across harnesses is therefore meaningless;
#: compare ``out`` (unambiguous everywhere) or stay within one harness.
USAGE_KEYS = ("in", "out", "cache_read", "cache_write", "reasoning_out")


@dataclass
class Event:
    """One step of a trajectory.

    Required: ``run_id``, ``agent_id``, ``i``, ``type``, ``role``. Everything
    else is absent unless the source actually carried it — a converter must not
    invent values to fill the shape.
    """

    run_id: str
    agent_id: str
    i: int
    type: str
    role: str

    ts: str | None = None
    turn: int | None = None

    text: str | None = None
    redacted: bool = False
    truncated: bool = False

    tool: str | None = None
    args: dict[str, Any] | None = None
    summary: str | None = None
    is_error: bool | None = None

    #: Identity of a ``tool_use`` (harness-native id) so results and spawned
    #: sub-agents can be linked back to the call that produced them.
    tool_use_id: str | None = None
    #: On a ``tool_result``: the ``tool_use_id`` it answers. On a sub-agent's
    #: first event: the parent's ``tool_use_id`` that spawned it.
    parent_tool_use: str | None = None

    usage: dict[str, int] | None = None
    origin: str = "agent"
    #: Pointer back into the upstream file, ``{"file": ..., "line": ...}``.
    source_ref: dict[str, Any] | None = None
    #: Harness-specific fields worth keeping but not worth promoting.
    extra: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """Drop absent fields so the on-disk lines stay readable and small."""
        d = asdict(self)
        return {
            k: v
            for k, v in d.items()
            if v is not None and not (k in ("redacted", "truncated") and v is False)
        }


@dataclass
class SubAgent:
    id: str
    label: str | None = None
    parent_tool_use: str | None = None
    t_start: str | None = None
    t_end: str | None = None
    n_events: int = 0


@dataclass
class RunMeta:
    """What the run was, what it scored, and where it came from."""

    run_id: str
    source: str
    benchmark: str
    task_id: str | None = None

    model: str | None = None
    harness: str | None = None

    #: ``{"hours": float, "gpus": int, "gpu_type": str}`` as the run was configured.
    budget: dict[str, Any] = field(default_factory=dict)

    t_start: str | None = None
    t_end: str | None = None
    duration_s: float | None = None

    #: ``{"metric", "value", "direction", "normalized", "baseline", "reference"}``
    final_score: dict[str, Any] | None = None
    tokens: dict[str, int] | None = None
    cost_usd: float | None = None

    n_events: int = 0
    n_by_type: dict[str, int] = field(default_factory=dict)
    n_by_origin: dict[str, int] = field(default_factory=dict)
    tools: dict[str, int] = field(default_factory=dict)
    subagents: list[dict[str, Any]] = field(default_factory=list)

    #: Contamination / validity verdicts as published, never re-judged here.
    #: Verdicts only — provenance goes in ``source_paths`` and descriptive
    #: metadata in ``extra``. The index reads a non-clear value here as "this run
    #: has a problem", so anything else parked in this dict flags the run.
    flags: dict[str, Any] = field(default_factory=dict)
    #: Upstream files this run was built from, for provenance.
    source_paths: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class SchemaError(ValueError):
    """An event or run violates the schema."""


def validate_event(e: Event) -> None:
    if e.type not in EVENT_TYPES:
        raise SchemaError(f"unknown type {e.type!r} (expected one of {EVENT_TYPES})")
    if e.role not in ROLES:
        raise SchemaError(f"unknown role {e.role!r} (expected one of {ROLES})")
    if e.origin not in ORIGINS:
        raise SchemaError(f"unknown origin {e.origin!r} (expected one of {ORIGINS})")
    if e.type == "tool_use" and not e.tool:
        raise SchemaError(f"tool_use without a tool name (i={e.i})")
    if e.type == "tool_result" and e.role != "user":
        raise SchemaError(f"tool_result must have role 'user' (i={e.i}, got {e.role!r})")
    if e.redacted and e.text:
        raise SchemaError(f"redacted event carries text (i={e.i})")
    if e.usage is not None:
        bad = set(e.usage) - set(USAGE_KEYS)
        if bad:
            raise SchemaError(f"unknown usage keys {sorted(bad)} (i={e.i}); map onto {USAGE_KEYS}")


def validate_stream(events: Iterable[Event], run_id: str) -> None:
    """Check the invariants that cross events: ids, per-agent numbering, usage."""
    per_agent: dict[str, int] = {}
    seen_turn_usage: set[tuple[str, int]] = set()
    for e in events:
        validate_event(e)
        if e.run_id != run_id:
            raise SchemaError(f"event run_id {e.run_id!r} != {run_id!r}")
        expected = per_agent.get(e.agent_id, 0)
        if e.i != expected:
            raise SchemaError(
                f"agent {e.agent_id!r}: expected i={expected}, got {e.i} "
                "(number each agent's stream separately, from 0)"
            )
        per_agent[e.agent_id] = expected + 1
        if e.usage is not None and e.turn is not None:
            key = (e.agent_id, e.turn)
            if key in seen_turn_usage:
                raise SchemaError(
                    f"agent {e.agent_id!r} turn {e.turn}: usage recorded twice; "
                    "keep it on the first event of the turn only"
                )
            seen_turn_usage.add(key)


def summarize(events: list[Event]) -> dict[str, Any]:
    """Counts a converter can drop straight into RunMeta."""
    n_by_type: dict[str, int] = {}
    n_by_origin: dict[str, int] = {}
    tools: dict[str, int] = {}
    tokens = {k: 0 for k in USAGE_KEYS}
    for e in events:
        n_by_type[e.type] = n_by_type.get(e.type, 0) + 1
        n_by_origin[e.origin] = n_by_origin.get(e.origin, 0) + 1
        if e.type == "tool_use" and e.tool:
            tools[e.tool] = tools.get(e.tool, 0) + 1
        if e.usage:
            for k, v in e.usage.items():
                tokens[k] = tokens.get(k, 0) + int(v)
    return {
        "n_events": len(events),
        "n_by_type": n_by_type,
        "n_by_origin": n_by_origin,
        "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
        "tokens": {k: v for k, v in tokens.items() if v},
    }


def events_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / f"{run_id}.jsonl.gz"


def meta_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / f"{run_id}.meta.json"


def write_run(events: list[Event], meta: RunMeta, out_dir: Path, validate: bool = True) -> Path:
    """Write one run's event stream and metadata. Returns the events path."""
    if validate:
        validate_stream(events, meta.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = events_path(out_dir, meta.run_id)
    tmp = ep.with_suffix(ep.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(ep)
    mp = meta_path(out_dir, meta.run_id)
    mp.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return ep


def read_events(path: Path) -> Iterator[Event]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Event(**json.loads(line))


def read_meta(path: Path) -> RunMeta:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunMeta(**d)


def iter_runs(source_dir: Path) -> Iterator[tuple[str, Path, Path]]:
    """Yield ``(run_id, events_path, meta_path)`` for every run under a source."""
    for ep in sorted(source_dir.glob("*.jsonl.gz")):
        run_id = ep.name[: -len(".jsonl.gz")]
        yield run_id, ep, meta_path(source_dir, run_id)

"""Reproducible experiment cards and scientist-owned outcomes.

An experiment is a directory, not a database row.  The directory contains an
immutable plan (``card.yaml``), the manifest produced when that plan is frozen,
an append-only event stream, phase logs, and a separate ``result.yaml`` whose
interpretation belongs to the scientist.

The lifecycle is deliberately small::

    draft -> frozen -> queued -> running -> awaiting_review -> closed

``queued`` is used only by detached runs.  Failed and killed executions still
enter ``awaiting_review``: failure is an experimental observation, not a reason
to lose the card.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

CARD_SCHEMA = "awm-experiment-card-v1"
MANIFEST_SCHEMA = "awm-experiment-manifest-v1"
RESULT_SCHEMA = "awm-experiment-result-v1"
STATE_SCHEMA = "awm-experiment-state-v1"

STATES = ("draft", "frozen", "queued", "running", "awaiting_review", "closed")
VERDICTS = ("supported", "contradicted", "inconclusive", "not_tested")
BASES = ("matched_eval", "diagnostic_eval", "agent_claim_only", "none")
DECISIONS = ("adopt", "reject", "continue", "retry", "abandon")
EXECUTION_STATUSES = ("completed", "failed", "killed", "not_run")
ARTIFACT_STATUSES = ("loadable", "invalid", "missing", "unknown")


class ExperimentError(ValueError):
    """The experiment bundle violates the lifecycle or schema."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(_json_bytes(value))
    os.replace(tmp, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentError(f"missing {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ExperimentError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"{path} must contain a mapping")
    return value


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(text)


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentError(f"{where} must be a mapping")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExperimentError(f"{where} must be a list")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{where} must be a non-empty string")
    return value


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ExperimentError(f"{where}.{key} is required")
    return mapping[key]


def _unique_ids(items: Iterable[dict[str, Any]], key: str, where: str) -> set[str]:
    seen: set[str] = set()
    for index, item in enumerate(items):
        ident = _string(_require(item, key, f"{where}[{index}]"), f"{where}[{index}].{key}")
        if ident in seen:
            raise ExperimentError(f"duplicate {where} {key} {ident!r}")
        seen.add(ident)
    return seen


def validate_card(card: dict[str, Any]) -> None:
    """Validate the semantic boundary and cross-references of a card."""

    if card.get("schema_version") != CARD_SCHEMA:
        raise ExperimentError(f"schema_version must be {CARD_SCHEMA!r}")
    for key in ("experiment_id", "title", "created_at", "scientist"):
        _string(_require(card, key, "card"), f"card.{key}")

    problem = _mapping(_require(card, "observed_problem", "card"), "observed_problem")
    _string(_require(problem, "statement", "observed_problem"), "observed_problem.statement")
    evidence = _list(_require(problem, "evidence", "observed_problem"), "observed_problem.evidence")
    if not evidence:
        raise ExperimentError("observed_problem.evidence must ground the problem in at least one rollout")
    for index, raw in enumerate(evidence):
        item = _mapping(raw, f"observed_problem.evidence[{index}]")
        for key in ("ref", "path", "locator", "observation"):
            _string(_require(item, key, f"observed_problem.evidence[{index}]"),
                    f"observed_problem.evidence[{index}].{key}")

    hypothesis = _mapping(_require(card, "hypothesis", "card"), "hypothesis")
    for key in ("performance_claim", "mechanism_claim", "falsification_condition"):
        _string(_require(hypothesis, key, "hypothesis"), f"hypothesis.{key}")
    if any(token in hypothesis["performance_claim"].lower() for token in ("hope ", "hopefully")):
        raise ExperimentError("hypothesis.performance_claim must be testable, not an aspiration")

    artifacts: list[dict[str, Any]] = []
    for group in ("inputs", "outputs"):
        values = _list(_require(card, group, "card"), group)
        for index, raw in enumerate(values):
            item = _mapping(raw, f"{group}[{index}]")
            for key in ("artifact_id", "kind", "path"):
                _string(_require(item, key, f"{group}[{index}]"), f"{group}[{index}].{key}")
            artifacts.append(item)
    artifact_ids = _unique_ids(artifacts, "artifact_id", "artifacts")

    training_data = _list(_require(card, "training_data", "card"), "training_data")
    for index, raw in enumerate(training_data):
        item = _mapping(raw, f"training_data[{index}]")
        ref = _string(_require(item, "artifact_id", f"training_data[{index}]"),
                      f"training_data[{index}].artifact_id")
        if ref not in artifact_ids:
            raise ExperimentError(f"training_data[{index}] references unknown artifact {ref!r}")
        _string(_require(item, "role", f"training_data[{index}]"), f"training_data[{index}].role")
        _string(_require(item, "selection", f"training_data[{index}]"),
                f"training_data[{index}].selection")
        weight = _require(item, "mixture_weight", f"training_data[{index}]")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ExperimentError(f"training_data[{index}].mixture_weight must be positive")

    intervention = _mapping(_require(card, "intervention", "card"), "intervention")
    for key in ("method", "summary", "parent_checkpoint_artifact_id"):
        _string(_require(intervention, key, "intervention"), f"intervention.{key}")
    parent = intervention["parent_checkpoint_artifact_id"]
    if parent not in artifact_ids:
        raise ExperimentError(f"intervention references unknown parent checkpoint {parent!r}")
    hyperparameters = _mapping(_require(intervention, "hyperparameters", "intervention"),
                               "intervention.hyperparameters")
    if not hyperparameters:
        raise ExperimentError("intervention.hyperparameters must record the planned settings")

    execution = _mapping(_require(card, "execution", "card"), "execution")
    phases_raw = _list(_require(execution, "phases", "execution"), "execution.phases")
    if not phases_raw:
        raise ExperimentError("execution.phases must contain at least one phase")
    phases: list[dict[str, Any]] = []
    for index, raw in enumerate(phases_raw):
        phase = _mapping(raw, f"execution.phases[{index}]")
        _string(_require(phase, "phase_id", f"execution.phases[{index}]"),
                f"execution.phases[{index}].phase_id")
        command = _list(_require(phase, "command", f"execution.phases[{index}]"),
                        f"execution.phases[{index}].command")
        if not command or not all(isinstance(arg, str) and arg for arg in command):
            raise ExperimentError(
                f"execution.phases[{index}].command must be a non-empty argv string list"
            )
        _string(_require(phase, "cwd", f"execution.phases[{index}]"),
                f"execution.phases[{index}].cwd")
        if "timeout_s" in phase and (
            not isinstance(phase["timeout_s"], (int, float)) or phase["timeout_s"] <= 0
        ):
            raise ExperimentError(f"execution.phases[{index}].timeout_s must be positive")
        env = phase.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise ExperimentError(f"execution.phases[{index}].env must map strings to strings")
        phases.append(phase)
    phase_ids = _unique_ids(phases, "phase_id", "execution.phases")

    plan = _mapping(_require(card, "evaluation_plan", "card"), "evaluation_plan")
    protocols_raw = _list(_require(plan, "protocols", "evaluation_plan"),
                          "evaluation_plan.protocols")
    if not protocols_raw:
        raise ExperimentError("evaluation_plan.protocols must contain at least one protocol")
    protocols: list[dict[str, Any]] = []
    for index, raw in enumerate(protocols_raw):
        protocol = _mapping(raw, f"evaluation_plan.protocols[{index}]")
        for key in ("protocol_id", "purpose", "metric", "direction", "phase_id",
                    "dataset_artifact_id", "comparator_checkpoint_artifact_id",
                    "measurement_path"):
            _string(_require(protocol, key, f"evaluation_plan.protocols[{index}]"),
                    f"evaluation_plan.protocols[{index}].{key}")
        if protocol["purpose"] not in ("performance", "mechanism", "both"):
            raise ExperimentError(
                f"evaluation_plan.protocols[{index}].purpose must be performance, mechanism, or both"
            )
        if protocol["direction"] not in ("higher", "lower"):
            raise ExperimentError(
                f"evaluation_plan.protocols[{index}].direction must be higher or lower"
            )
        if protocol["phase_id"] not in phase_ids:
            raise ExperimentError(
                f"evaluation_plan.protocols[{index}] references unknown phase {protocol['phase_id']!r}"
            )
        for field in ("dataset_artifact_id", "comparator_checkpoint_artifact_id"):
            if protocol[field] not in artifact_ids:
                raise ExperimentError(
                    f"evaluation_plan.protocols[{index}].{field} references unknown artifact "
                    f"{protocol[field]!r}"
                )
        protocols.append(protocol)
    _unique_ids(protocols, "protocol_id", "evaluation_plan.protocols")
    policy = _mapping(_require(plan, "decision_policy", "evaluation_plan"),
                      "evaluation_plan.decision_policy")
    for key in ("continue_if", "abort_if"):
        _string(_require(policy, key, "evaluation_plan.decision_policy"),
                f"evaluation_plan.decision_policy.{key}")

    budget = _mapping(_require(card, "budget", "card"), "budget")
    for key in ("wall_time_s", "gpus"):
        value = _require(budget, key, "budget")
        if not isinstance(value, (int, float)) or value < 0:
            raise ExperimentError(f"budget.{key} must be a non-negative number")


def validate_result(result: dict[str, Any], experiment_id: str, evidence_refs: set[str]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ExperimentError(f"result.schema_version must be {RESULT_SCHEMA!r}")
    if result.get("experiment_id") != experiment_id:
        raise ExperimentError(
            f"result.experiment_id {result.get('experiment_id')!r} != {experiment_id!r}"
        )
    facts = _mapping(_require(result, "result", "result"), "result.result")
    if facts.get("execution_status") not in EXECUTION_STATUSES:
        raise ExperimentError(f"result.execution_status must be one of {EXECUTION_STATUSES}")
    if facts.get("artifact_status") not in ARTIFACT_STATUSES:
        raise ExperimentError(f"result.artifact_status must be one of {ARTIFACT_STATUSES}")
    measurements = _list(_require(facts, "measurements", "result"), "result.measurements")
    for index, raw in enumerate(measurements):
        measurement = _mapping(raw, f"result.measurements[{index}]")
        for key in ("protocol_id", "metric", "value", "evidence_ref"):
            _require(measurement, key, f"result.measurements[{index}]")
        if not isinstance(measurement["value"], (int, float)):
            raise ExperimentError(f"result.measurements[{index}].value must be numeric")
        if measurement["evidence_ref"] not in evidence_refs:
            raise ExperimentError(
                f"result.measurements[{index}] has unknown evidence_ref "
                f"{measurement['evidence_ref']!r}"
            )

    assessment = _mapping(_require(result, "scientist_assessment", "result"),
                          "scientist_assessment")
    for name in ("outcome", "mechanism"):
        item = _mapping(_require(assessment, name, "scientist_assessment"),
                        f"scientist_assessment.{name}")
        verdict = item.get("verdict")
        basis = item.get("basis")
        if verdict not in VERDICTS:
            raise ExperimentError(f"scientist_assessment.{name}.verdict must be one of {VERDICTS}")
        if basis not in BASES:
            raise ExperimentError(f"scientist_assessment.{name}.basis must be one of {BASES}")
        summary = _string(_require(item, "summary", f"scientist_assessment.{name}"),
                          f"scientist_assessment.{name}.summary")
        if "TODO" in summary.upper():
            raise ExperimentError(f"scientist_assessment.{name}.summary is still a template")
        refs = _list(_require(item, "evidence_refs", f"scientist_assessment.{name}"),
                     f"scientist_assessment.{name}.evidence_refs")
        unknown = [ref for ref in refs if ref not in evidence_refs]
        if unknown:
            raise ExperimentError(
                f"scientist_assessment.{name} references unknown evidence {unknown}"
            )
        if (
            verdict in ("supported", "contradicted")
            and (basis not in ("matched_eval", "diagnostic_eval") or not refs)
        ):
            raise ExperimentError(
                f"a {verdict} {name} verdict requires measured evidence, not {basis}"
            )
        if (
            name == "mechanism"
            and verdict in ("supported", "contradicted")
            and basis != "diagnostic_eval"
        ):
            raise ExperimentError(
                "a supported/contradicted mechanism verdict requires a diagnostic_eval"
            )

    decision = _mapping(_require(result, "scientist_decision", "result"),
                        "scientist_decision")
    if decision.get("action") not in DECISIONS:
        raise ExperimentError(f"scientist_decision.action must be one of {DECISIONS}")
    rationale = _string(_require(decision, "rationale", "scientist_decision"),
                        "scientist_decision.rationale")
    if "TODO" in rationale.upper():
        raise ExperimentError("scientist_decision.rationale is still a template")


def _resolve(bundle: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (bundle / path).resolve()


def _git_context(path: Path) -> dict[str, Any] | None:
    probe = path if path.is_dir() else path.parent
    try:
        top = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", top, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None
    return {"root": top, "commit": commit, "dirty": dirty}


class ExperimentBundle:
    """Filesystem-backed lifecycle for one experiment."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.card_path = self.directory / "card.yaml"
        self.manifest_path = self.directory / "manifest.json"
        self.state_path = self.directory / "state.json"
        self.events_path = self.directory / "events.jsonl"
        self.result_path = self.directory / "result.yaml"
        self.logs_dir = self.directory / "logs"

    @property
    def card(self) -> dict[str, Any]:
        card = _load_yaml(self.card_path)
        validate_card(card)
        return card

    @property
    def state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise ExperimentError(f"missing {self.state_path}; run experiment scaffold first")
        value = json.loads(self.state_path.read_text())
        if value.get("schema_version") != STATE_SCHEMA or value.get("status") not in STATES:
            raise ExperimentError(f"invalid state in {self.state_path}")
        return value

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        _atomic_json(self.state_path, state)

    def append_event(self, event_type: str, **payload: Any) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / ".events.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            count = 0
            if self.events_path.is_file():
                with self.events_path.open() as existing:
                    count = sum(1 for line in existing if line.strip())
            ref = f"event:{count + 1:06d}"
            value = {"ref": ref, "ts": _now(), "type": event_type, **payload}
            with self.events_path.open("a") as events:
                events.write(json.dumps(value, sort_keys=True) + "\n")
                events.flush()
                os.fsync(events.fileno())
            fcntl.flock(lock, fcntl.LOCK_UN)
        return ref

    def evidence_refs(self) -> set[str]:
        refs: set[str] = set()
        if self.events_path.is_file():
            for line in self.events_path.read_text().splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value.get("ref"), str):
                        refs.add(value["ref"])
        return refs

    @classmethod
    def scaffold(cls, directory: str | Path, experiment_id: str | None = None,
                 title: str | None = None) -> ExperimentBundle:
        bundle = cls(directory)
        if bundle.directory.exists() and any(bundle.directory.iterdir()):
            raise ExperimentError(f"refusing to overwrite non-empty {bundle.directory}")
        bundle.directory.mkdir(parents=True, exist_ok=True)
        ident = experiment_id or bundle.directory.name
        card = card_template(ident, title or ident.replace("-", " ").title())
        _write_yaml(bundle.card_path, card)
        _atomic_json(bundle.state_path, {
            "schema_version": STATE_SCHEMA,
            "experiment_id": ident,
            "status": "draft",
            "created_at": _now(),
            "updated_at": _now(),
            "phase_results": [],
        })
        bundle.logs_dir.mkdir()
        bundle.append_event("experiment_scaffolded", experiment_id=ident)
        return bundle

    def freeze(self) -> dict[str, Any]:
        card = self.card
        state = self.state
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if state["status"] not in ("draft", "frozen"):
            raise ExperimentError(f"cannot freeze an experiment in state {state['status']!r}")
        if state["experiment_id"] != card["experiment_id"]:
            raise ExperimentError("card.experiment_id does not match state.json")

        inputs = []
        missing = []
        for artifact in card["inputs"]:
            path = _resolve(self.directory, artifact["path"])
            exists = path.exists()
            required = artifact.get("required", True)
            record: dict[str, Any] = {
                "artifact_id": artifact["artifact_id"],
                "kind": artifact["kind"],
                "declared_path": artifact["path"],
                "resolved_path": str(path),
                "required": required,
                "exists": exists,
                "declared_integrity": artifact.get("integrity"),
            }
            if exists:
                stat = path.stat()
                record.update({
                    "object_type": "directory" if path.is_dir() else "file",
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
                if path.is_file():
                    record["sha256"] = _sha256(path)
            elif required:
                missing.append(str(path))
            inputs.append(record)
        if missing:
            raise ExperimentError("required inputs are missing:\n  " + "\n  ".join(missing))

        phases = []
        for phase in card["execution"]["phases"]:
            cwd = _resolve(self.directory, phase["cwd"])
            if not cwd.is_dir():
                raise ExperimentError(f"phase {phase['phase_id']!r} cwd does not exist: {cwd}")
            executable = phase["command"][0]
            executable_path = (
                str(_resolve(cwd, executable)) if "/" in executable
                else shutil.which(executable)
            )
            phases.append({
                "phase_id": phase["phase_id"],
                "cwd": str(cwd),
                "command": phase["command"],
                "executable": executable_path,
                "git": _git_context(cwd),
            })

        card_sha = _sha256(self.card_path)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "experiment_id": card["experiment_id"],
            "frozen_at": _now(),
            "card": {"path": str(self.card_path), "sha256": card_sha},
            "inputs": inputs,
            "outputs": [
                {
                    "artifact_id": artifact["artifact_id"],
                    "kind": artifact["kind"],
                    "declared_path": artifact["path"],
                    "resolved_path": str(_resolve(self.directory, artifact["path"])),
                    "required": artifact.get("required", True),
                }
                for artifact in card["outputs"]
            ],
            "phases": phases,
        }
        _atomic_json(self.manifest_path, manifest)
        state.update({"status": "frozen", "card_sha256": card_sha, "frozen_at": _now()})
        self._save_state(state)
        self.append_event("experiment_frozen", card_sha256=card_sha)
        return manifest

    def _assert_frozen_card(self) -> tuple[dict[str, Any], dict[str, Any]]:
        card = self.card
        state = self.state
        if not self.manifest_path.is_file():
            raise ExperimentError("manifest.json is missing; freeze the card before running it")
        manifest = json.loads(self.manifest_path.read_text())
        actual = _sha256(self.card_path)
        frozen = manifest.get("card", {}).get("sha256")
        if actual != frozen:
            raise ExperimentError(
                "card.yaml changed after freeze; create a new experiment or restore the frozen card"
            )
        return card, state

    def run(self, *, queued_ok: bool = False) -> dict[str, Any]:
        card, state = self._assert_frozen_card()
        allowed = {"frozen"}
        if queued_ok:
            allowed.add("queued")
        if state["status"] not in allowed:
            raise ExperimentError(f"cannot run an experiment in state {state['status']!r}")

        state.update({
            "status": "running",
            "started_at": _now(),
            "worker_pid": os.getpid(),
            "claude_session_id": os.environ.get("AWM_CLAUDE_SESSION_ID"),
            "phase_results": [],
        })
        self._save_state(state)
        self.append_event(
            "experiment_started", worker_pid=os.getpid(),
            claude_session_id=state.get("claude_session_id"),
        )

        execution_status = "completed"
        for index, phase in enumerate(card["execution"]["phases"], start=1):
            phase_id = phase["phase_id"]
            cwd = _resolve(self.directory, phase["cwd"])
            stdout_path = self.logs_dir / f"{index:02d}-{phase_id}.stdout.log"
            stderr_path = self.logs_dir / f"{index:02d}-{phase_id}.stderr.log"
            env = dict(os.environ)
            env.update(phase.get("env", {}))
            env.update({
                "AWM_EXPERIMENT_DIR": str(self.directory),
                "AWM_EXPERIMENT_ID": card["experiment_id"],
                "AWM_PHASE_ID": phase_id,
                "AWM_OBSERVATIONS_PATH": str(self.events_path),
            })
            started = time.monotonic()
            started_at = _now()
            state = self.state
            state["current_phase"] = phase_id
            self._save_state(state)
            self.append_event(
                "phase_started", phase_id=phase_id, command=phase["command"], cwd=str(cwd)
            )
            timed_out = False
            returncode: int | None = None
            print(f"[{card['experiment_id']}] phase {phase_id} -> {stdout_path}", flush=True)
            try:
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    process = subprocess.Popen(
                        phase["command"], cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                        text=True, start_new_session=True,
                    )
                    timeout_s = phase.get("timeout_s")
                    while returncode is None:
                        returncode = process.poll()
                        if returncode is not None:
                            break
                        if timeout_s is not None and time.monotonic() - started > timeout_s:
                            timed_out = True
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                returncode = process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                returncode = process.wait()
                            break
                        time.sleep(0.2)
            except KeyboardInterrupt:
                execution_status = "killed"
                if "process" in locals() and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                returncode = process.returncode if "process" in locals() else None
            except OSError as exc:
                execution_status = "failed"
                returncode = 127
                stderr_path.write_text(f"{type(exc).__name__}: {exc}\n")

            elapsed = time.monotonic() - started
            if timed_out or (returncode is not None and returncode != 0):
                execution_status = "failed" if execution_status != "killed" else execution_status
            phase_result = {
                "phase_id": phase_id,
                "started_at": started_at,
                "ended_at": _now(),
                "duration_s": elapsed,
                "returncode": returncode,
                "timed_out": timed_out,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            state = self.state
            state.setdefault("phase_results", []).append(phase_result)
            state["current_phase"] = None
            self._save_state(state)
            self.append_event("phase_finished", **phase_result)
            if execution_status != "completed":
                break

        measurements = self._collect_measurements(card)
        artifacts = []
        missing_required = False
        for artifact in card["outputs"]:
            path = _resolve(self.directory, artifact["path"])
            exists = path.exists()
            if artifact.get("required", True) and not exists:
                missing_required = True
            record = {
                "artifact_id": artifact["artifact_id"],
                "kind": artifact["kind"],
                "path": str(path),
                "exists": exists,
            }
            if exists and path.is_file():
                record.update({"size_bytes": path.stat().st_size, "sha256": _sha256(path)})
            artifacts.append(record)
            self.append_event("artifact_checked", **record)

        if execution_status == "completed" and missing_required:
            execution_status = "failed"
        artifact_status = "missing" if missing_required else "unknown"
        summary = {
            "experiment_id": card["experiment_id"],
            "execution_status": execution_status,
            "artifact_status": artifact_status,
            "measurements": measurements,
            "artifacts": artifacts,
            "completed_at": _now(),
        }
        _atomic_json(self.directory / "run_summary.json", summary)
        self._write_result_draft(summary)
        state = self.state
        state.update({
            "status": "awaiting_review",
            "execution_status": execution_status,
            "completed_at": summary["completed_at"],
        })
        self._save_state(state)
        self.append_event("experiment_finished", execution_status=execution_status)
        return summary

    def run_detached(self) -> int:
        _card, state = self._assert_frozen_card()
        if state["status"] != "frozen":
            raise ExperimentError(f"cannot queue an experiment in state {state['status']!r}")
        supervisor = self.logs_dir / "worker.log"
        state.update({"status": "queued", "queued_at": _now()})
        self._save_state(state)
        repo_root = Path(__file__).resolve().parent.parent
        with supervisor.open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "awm.cli", "experiment", "run", str(self.directory),
                 "--worker"],
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state = self.state
        state["worker_pid"] = process.pid
        self._save_state(state)
        self.append_event("experiment_queued", worker_pid=process.pid, log=str(supervisor))
        return process.pid

    def _collect_measurements(self, card: dict[str, Any]) -> list[dict[str, Any]]:
        measurements = []
        for protocol in card["evaluation_plan"]["protocols"]:
            path = _resolve(self.directory, protocol["measurement_path"])
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                self.append_event(
                    "measurement_invalid", protocol_id=protocol["protocol_id"],
                    path=str(path), error=str(exc),
                )
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("value"), (int, float)):
                self.append_event(
                    "measurement_invalid", protocol_id=protocol["protocol_id"],
                    path=str(path), error="measurement JSON requires numeric value",
                )
                continue
            measurement: dict[str, Any] = {
                "protocol_id": protocol["protocol_id"],
                "purpose": protocol["purpose"],
                "metric": raw.get("metric", protocol["metric"]),
                "value": raw["value"],
                "n": raw.get("n"),
                "matched_parent_measurement": raw.get("parent_value"),
                "path": str(path),
            }
            parent = measurement["matched_parent_measurement"]
            if isinstance(parent, (int, float)):
                measurement["matched_delta"] = measurement["value"] - parent
            ref = self.append_event("measurement_observed", **measurement)
            measurement["evidence_ref"] = ref
            measurements.append(measurement)
        return measurements

    def _write_result_draft(self, summary: dict[str, Any]) -> None:
        if self.result_path.exists():
            return
        draft = {
            "schema_version": RESULT_SCHEMA,
            "experiment_id": summary["experiment_id"],
            "result": {
                "execution_status": summary["execution_status"],
                "artifact_status": summary["artifact_status"],
                "measurements": summary["measurements"],
                "artifacts": summary["artifacts"],
                "selected_as_next_incumbent": None,
                "selected_for_final": False,
            },
            "scientist_assessment": {
                "outcome": {
                    "verdict": "inconclusive",
                    "basis": "none",
                    "summary": "TODO: interpret the performance claim from matched measurements.",
                    "evidence_refs": [],
                },
                "mechanism": {
                    "verdict": "not_tested",
                    "basis": "none",
                    "summary": "TODO: interpret only a dedicated diagnostic evaluation.",
                    "evidence_refs": [],
                },
            },
            "scientist_decision": {
                "action": "reject",
                "rationale": "TODO: record the scientist's continue/adopt/reject decision.",
            },
        }
        _write_yaml(self.result_path, draft)

    def observe(self, kind: str, summary: str, *, phase_id: str | None = None,
                data: dict[str, Any] | None = None, artifact: str | None = None) -> str:
        _string(kind, "kind")
        _string(summary, "summary")
        return self.append_event(
            "observation", kind=kind, summary=summary, phase_id=phase_id,
            data=data or {}, artifact=artifact,
        )

    def finalize(self, result_path: str | Path | None = None) -> dict[str, Any]:
        card = self.card
        state = self.state
        if state["status"] != "awaiting_review":
            raise ExperimentError(
                f"scientist review requires awaiting_review, got {state['status']!r}"
            )
        path = Path(result_path).expanduser().resolve() if result_path else self.result_path
        result = _load_yaml(path)
        validate_result(result, card["experiment_id"], self.evidence_refs())
        if path != self.result_path:
            _write_yaml(self.result_path, result)
        ref = self.append_event(
            "scientist_reviewed",
            outcome_verdict=result["scientist_assessment"]["outcome"]["verdict"],
            mechanism_verdict=result["scientist_assessment"]["mechanism"]["verdict"],
            decision=result["scientist_decision"]["action"],
        )
        state.update({"status": "closed", "closed_at": _now(), "result_event_ref": ref})
        self._save_state(state)
        return result


def card_template(experiment_id: str, title: str) -> dict[str, Any]:
    """Return a deliberately explicit card for the scientist to fill in."""

    return {
        "schema_version": CARD_SCHEMA,
        "experiment_id": experiment_id,
        "title": title,
        "created_at": _now(),
        "scientist": "scientist",
        "observed_problem": {
            "statement": "REPLACE: concrete failure observed in prior rollouts",
            "evidence": [{
                "ref": "rollout:REPLACE",
                "path": "/absolute/path/to/rollout.jsonl",
                "locator": "event or line range",
                "observation": "REPLACE: what the rollout demonstrates",
            }],
        },
        "hypothesis": {
            "performance_claim": "REPLACE: testable prediction relative to a named comparator",
            "mechanism_claim": "REPLACE: proposed reason the intervention should work",
            "falsification_condition": "REPLACE: observation that would contradict the prediction",
        },
        "inputs": [
            {
                "artifact_id": "parent_checkpoint",
                "kind": "checkpoint",
                "path": "/absolute/path/to/parent-checkpoint",
                "required": True,
                "integrity": {"type": "revision", "value": "REPLACE"},
            },
            {
                "artifact_id": "train_dataset",
                "kind": "dataset",
                "path": "/absolute/path/to/train.jsonl",
                "required": True,
                "integrity": {"type": "sha256", "value": "REPLACE"},
            },
            {
                "artifact_id": "diagnostic_dataset",
                "kind": "dataset",
                "path": "/absolute/path/to/diagnostic.jsonl",
                "required": True,
                "integrity": {"type": "sha256", "value": "REPLACE"},
            },
        ],
        "outputs": [{
            "artifact_id": "candidate_checkpoint",
            "kind": "checkpoint",
            "path": "artifacts/candidate-checkpoint",
            "required": True,
        }],
        "training_data": [{
            "artifact_id": "train_dataset",
            "role": "supervised_training",
            "selection": "REPLACE: exact filter/deduplication rule",
            "mixture_weight": 1.0,
        }],
        "intervention": {
            "method": "sft",
            "summary": "REPLACE: one candidate-producing intervention",
            "parent_checkpoint_artifact_id": "parent_checkpoint",
            "hyperparameters": {"learning_rate": "REPLACE"},
        },
        "execution": {
            "phases": [{
                "phase_id": "train",
                "command": ["python3", "train.py", "--config", "config.yaml"],
                "cwd": ".",
                "timeout_s": 3600,
                "env": {},
            }],
        },
        "evaluation_plan": {
            "protocols": [{
                "protocol_id": "diagnostic-v1",
                "purpose": "both",
                "metric": "accuracy",
                "direction": "higher",
                "phase_id": "train",
                "dataset_artifact_id": "diagnostic_dataset",
                "comparator_checkpoint_artifact_id": "parent_checkpoint",
                "measurement_path": "measurements/diagnostic-v1.json",
            }],
            "decision_policy": {
                "continue_if": "REPLACE: causal intermediate observation that justifies more compute",
                "abort_if": "REPLACE: observation that justifies stopping or changing intervention",
            },
        },
        "budget": {"wall_time_s": 3600, "gpus": 1},
    }


def open_experiments(root: str | Path) -> list[dict[str, str]]:
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return []
    open_states = {"queued", "running", "awaiting_review"}
    found = []
    for state_path in sorted(root_path.glob("*/state.json")):
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") in open_states:
            found.append({
                "experiment_id": str(state.get("experiment_id", state_path.parent.name)),
                "status": str(state["status"]),
                "path": str(state_path.parent.resolve()),
            })
    return found

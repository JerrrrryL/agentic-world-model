"""WMA memory: sidecar-owned, outside the session directory, grows across sessions.

Layout under ``AWM_WM_MEMORY`` (default ``<data>/wm-memory``)::

    raw/<session>/<card_id>/          copies of card, contract, observations, pings, replies, result
    structured/cards.jsonl            one row per closed card
    structured/observations.jsonl     one row per observation
    structured/interactions.jsonl     ping -> reply pairs
    structured/outcomes.jsonl         official outcomes, imported later (train side only)
    notes/<session>-<card_id>.md      grounded lessons (llm arm)

Every row carries ``provenance: {session, arm, split_side}``. A memory opened
``readonly`` (held-out sessions) answers queries and discards writes.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .schema import dump_json, now, read_jsonl

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]{2,}")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


class Memory:
    def __init__(self, root: Path, *, session: str, arm: str, split_side: str = "train",
                 readonly: bool = False, visible_sides: tuple[str, ...] = ("train",)):
        self.root = Path(root)
        self.session = session
        self.arm = arm
        self.split_side = split_side
        self.readonly = readonly
        self.visible_sides = tuple(visible_sides)
        self.structured = self.root / "structured"
        if not readonly:
            (self.root / "raw").mkdir(parents=True, exist_ok=True)
            self.structured.mkdir(parents=True, exist_ok=True)
            (self.root / "notes").mkdir(parents=True, exist_ok=True)

    # ---- provenance

    def _prov(self) -> dict[str, str]:
        return {"session": self.session, "arm": self.arm, "split_side": self.split_side, "at": now()}

    def _append(self, table: str, row: dict[str, Any]) -> None:
        if self.readonly:
            return
        row = {**row, "provenance": self._prov()}
        with (self.structured / f"{table}.jsonl").open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _rows(self, table: str) -> list[dict[str, Any]]:
        path = self.structured / f"{table}.jsonl"
        return read_jsonl(path) if path.is_file() else []

    # ---- writes

    def record_observation(self, card: dict[str, Any], observation: dict[str, Any]) -> None:
        self._append("observations", {
            "card_id": card["card_id"],
            "base_model": _base_model(card),
            "method_family": card["setup"]["method"].get("family"),
            "obs_id": observation["obs_id"],
            "step": observation["checkpoint"].get("step"),
            "fraction": observation.get("fraction"),
            "evaluators": {k: {kk: v.get(kk) for kk in ("value", "n", "stderr", "delta_vs_parent", "delta_vs_prev")}
                           for k, v in observation["evaluators"].items()},
            "cause": observation.get("cause"),
        })

    def record_interaction(self, card_id: str, ping: dict[str, Any], reply: dict[str, Any] | None) -> None:
        self._append("interactions", {
            "card_id": card_id, "ping_id": ping["ping_id"], "kind": ping["kind"],
            "raised_by": ping.get("raised_by"), "prediction": ping.get("prediction"),
            "options": [o["id"] for o in ping.get("options", [])],
            "choice": reply["choice"] if reply else None, "by": reply.get("by") if reply else None,
        })

    def record_card(self, card_dir: Path, card: dict[str, Any], contract: dict[str, Any] | None,
                    result: dict[str, Any] | None, state: dict[str, Any]) -> None:
        if self.readonly:
            return
        raw = self.root / "raw" / self.session / card["card_id"]
        raw.mkdir(parents=True, exist_ok=True)
        for name in ("card.yaml", "contract.yaml", "seal.json", "state.json", "manifest.json"):
            src = card_dir / name
            if src.is_file():
                shutil.copy2(src, raw / name)
        for sub in ("observations", "pings", "replies"):
            if (card_dir / sub).is_dir():
                shutil.copytree(card_dir / sub, raw / sub, dirs_exist_ok=True)
        best = None
        for obs in state.get("observations", []):
            sel = (contract or {}).get("selection", {}).get("evaluator")
            v = obs.get("evaluators", {}).get(sel, {}).get("value") if sel else None
            if v is not None and (best is None or v > best):
                best = v
        self._append("cards", {
            "card_id": card["card_id"],
            "base_model": _base_model(card),
            "parent_origin": card["setup"]["parent_checkpoint"].get("origin"),
            "method_family": card["setup"]["method"].get("family"),
            "data_sources": [d.get("source") for d in card["setup"].get("data", [])],
            "problem": card["problem"].get("statement"),
            "claim": card["hypothesis"].get("claim"),
            "hyperparams": card["setup"]["method"].get("hyperparams"),
            "n_observations": len(state.get("observations", [])),
            "best_selection_value": best,
            "parent_value": (state.get("parent") or {}).get(
                (contract or {}).get("selection", {}).get("evaluator", ""), {}).get("value"),
            "final_status": state.get("status"),
            "execution": (result or {}).get("result", {}).get("execution"),
            "verdict": (result or {}).get("conclusion", {}).get("verdict"),
            "decision": (result or {}).get("conclusion", {}).get("decision"),
            "sealed_obs": (state.get("seal") or {}).get("obs_id"),
            "raw_dir": str(raw),
        })

    def record_outcome(self, card_id: str, session: str, official: dict[str, Any]) -> None:
        self._append("outcomes", {"card_id": card_id, "for_session": session, **official})

    def note(self, card_id: str, text: str) -> Path | None:
        if self.readonly:
            return None
        path = self.root / "notes" / f"{self.session}-{card_id}.md"
        path.write_text(text)
        return path

    # ---- reads

    def precedents(self, card: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
        """Nearest closed cards by token overlap on base model, method, data, problem, claim."""
        query = tokens(" ".join([
            _base_model(card) or "", card["setup"]["method"].get("family", ""),
            " ".join(str(d.get("source", "")) for d in card["setup"].get("data", [])),
            str(card["problem"].get("statement", "")), str(card["hypothesis"].get("claim", "")),
        ]))
        scored = []
        for row in self._rows("cards"):
            if row.get("provenance", {}).get("split_side") not in self.visible_sides:
                continue
            doc = tokens(" ".join([
                str(row.get("base_model", "")), str(row.get("method_family", "")),
                " ".join(map(str, row.get("data_sources") or [])),
                str(row.get("problem", "")), str(row.get("claim", "")),
            ]))
            if not doc:
                continue
            overlap = len(query & doc) / max(len(query | doc), 1)
            same_model = 0.25 if row.get("base_model") == _base_model(card) else 0.0
            same_family = 0.15 if row.get("method_family") == card["setup"]["method"].get("family") else 0.0
            scored.append((overlap + same_model + same_family, row))
        scored.sort(key=lambda t: -t[0])
        out = []
        for score, row in scored[:k]:
            delta = None
            if row.get("best_selection_value") is not None and row.get("parent_value") is not None:
                delta = round(row["best_selection_value"] - row["parent_value"], 4)
            out.append({"similarity": round(score, 3), "card_id": row["card_id"],
                        "session": row["provenance"]["session"], "base_model": row.get("base_model"),
                        "method_family": row.get("method_family"), "data_sources": row.get("data_sources"),
                        "delta_best_vs_parent": delta, "decision": row.get("decision"),
                        "verdict": row.get("verdict"), "raw_dir": row.get("raw_dir")})
        return out

    def curves(self, card_ids: list[tuple[str, str]]) -> dict[str, list[dict[str, Any]]]:
        """Observation rows for (session, card_id) pairs, in step order."""
        want = set(card_ids)
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self._rows("observations"):
            key = (row["provenance"]["session"], row["card_id"])
            if key in want:
                out.setdefault(f"{key[0]}/{key[1]}", []).append(row)
        for rows in out.values():
            rows.sort(key=lambda r: (r.get("step") or 0))
        return out

    def stats(self) -> dict[str, int]:
        return {t: len(self._rows(t)) for t in ("cards", "observations", "interactions", "outcomes")}

    # ---- seeding from reconstructed cards

    def seed_from_exp_cards(self, results_dir: Path, *, side: str = "train") -> int:
        """Load ``results/exp-cards/<split>/<side>/<run_ref>/exp-*.yaml`` as precedent rows.

        These are reconstructions, so they are tagged ``reconstructed: true``
        and ``split_side`` from the directory, never mixed into live evidence.
        """
        import yaml

        n = 0
        for path in sorted((results_dir / side).glob("r-*/exp-*.yaml")):
            try:
                card = yaml.safe_load(path.read_text())
            except yaml.YAMLError:
                continue
            if not isinstance(card, dict) or "setup" not in card:
                continue
            setup = card.get("setup") or {}
            res = card.get("result") or {}
            con = card.get("conclusion") or {}
            meas = [m.get("value") for m in (res.get("measurements") or []) if isinstance(m.get("value"), (int, float))]
            comp = ((card.get("evaluation") or {}).get("comparator") or {}).get("value")
            row = {
                "card_id": f"{path.parent.name}/{path.stem}",
                "base_model": (setup.get("parent_checkpoint") or {}).get("path"),
                "parent_origin": (setup.get("parent_checkpoint") or {}).get("origin"),
                "method_family": (setup.get("method") or {}).get("family"),
                "data_sources": [d.get("source") for d in (setup.get("data") or []) if isinstance(d, dict)],
                "problem": (card.get("problem") or {}).get("statement"),
                "claim": (card.get("hypothesis") or {}).get("claim"),
                "hyperparams": (setup.get("method") or {}).get("hyperparams"),
                "n_observations": len(meas),
                "best_selection_value": max(meas) if meas else None,
                "parent_value": comp if isinstance(comp, (int, float)) else None,
                "final_status": "reconstructed",
                "execution": res.get("execution"),
                "verdict": con.get("verdict"),
                "decision": con.get("decision"),
                "reconstructed": True,
                "raw_dir": str(path.parent),
            }
            saved_side = self.split_side
            self.split_side = side
            self._append("cards", row)
            self.split_side = saved_side
            n += 1
        return n


def _base_model(card: dict[str, Any]) -> str | None:
    setup = card.get("setup") or {}
    return setup.get("base_model") or (setup.get("parent_checkpoint") or {}).get("origin_model") \
        or (setup.get("parent_checkpoint") or {}).get("path")


def dump_debug(path: Path, value: Any) -> None:
    dump_json(path, value)

"""Filesystem layout.

Everything large lives outside the repo: trajectories, datasets and model
weights are data, not source, and the root partition is nearly full.

The data volume is reached through ``<repo>/data``, a gitignored symlink to
wherever it physically lives on this machine (``/data2/gangda/hv`` here). Code
and docs can then name one stable path while each machine points it somewhere
different. A fresh clone creates it once:

    ln -s /data2/gangda/hv data

``AWM_DATA_ROOT`` overrides the symlink, which is what the tests use to run
against an empty volume.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_ROOT = REPO_ROOT / "data"

#: Where the symlink points on this machine, used only to make the error message
#: actionable when someone clones the repo and the symlink is missing.
MACHINE_DATA_ROOT = Path("/data2/gangda/hv")


def data_root(require: bool = False) -> Path:
    root = Path(os.environ.get("AWM_DATA_ROOT", DEFAULT_DATA_ROOT))
    if require and not root.exists():
        raise FileNotFoundError(
            f"data volume not found at {root}. Create the symlink with\n"
            f"    ln -s {MACHINE_DATA_ROOT} {DEFAULT_DATA_ROOT}\n"
            "or point AWM_DATA_ROOT elsewhere."
        )
    return root


def raw_dir(source: str) -> Path:
    """Upstream trajectories, byte-for-byte as published."""
    return data_root() / "traj" / "raw" / source


def events_root() -> Path:
    """Parent of the per-source event directories."""
    return data_root() / "traj" / "events"


def events_dir(source: str) -> Path:
    """Unified event streams. Analysis code reads only this layer."""
    return events_root() / source


def runs_dir() -> Path:
    """Our own experiment output, rsynced back from the GPU machines."""
    return data_root() / "traj" / "runs"


def index_path() -> Path:
    return data_root() / "traj" / "index.parquet"


def splits_dir() -> Path:
    return REPO_ROOT / "splits"


def tasks_dir() -> Path:
    return REPO_ROOT / "tasks"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

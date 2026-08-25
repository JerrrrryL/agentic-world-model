"""Shared fixtures.

Tests run against two kinds of input. Committed samples under ``tests/data/``
are truncated real trajectories — a few dozen events each — so the suite runs
anywhere with no data volume mounted. The full releases under ``HV_DATA_ROOT``
are only used by tests marked ``needs_data``, which skip when absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hv.paths import raw_dir

DATA = Path(__file__).parent / "data"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_data: requires the full upstream releases under HV_DATA_ROOT"
    )


@pytest.fixture
def sample_dir() -> Path:
    """Committed, truncated samples."""
    return DATA


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not fetched: {path} (run `hv traj fetch`)")
    return path


@pytest.fixture
def pi_raw() -> Path:
    return _require(raw_dir("pi_speedrun") / "traces", "PI speedrun traces")


@pytest.fixture
def ptb_raw() -> Path:
    return _require(raw_dir("posttrainbench"), "PostTrainBench trajectories")

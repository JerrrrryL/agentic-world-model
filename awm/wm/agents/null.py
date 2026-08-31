"""The null arm: deterministic grounding, a default contract, bundles only.

The control. It never retrieves, never predicts, never objects, never asks.
Its notices are the observation bundle and nothing else; contract rules are
evaluated by the runtime, not here.
"""

from __future__ import annotations

from .base import WorldModelAgent


class NullAgent(WorldModelAgent):
    arm = "null"

"""The predictor arm: llm + a learned prior at the brief and a posterior after each observation.

Not implemented in this drop. ``predict`` should return
``{metric, horizon: next_obs|final, delta_mean, delta_sd, basis}`` fitted on
``memory`` (structured cards + observation curves, train side only) and be
called from ``on_proposal`` (prior) and ``on_observation`` (posterior). The
runtime carries the prediction into the ping unchanged and memory records it
against the eventual outcome.
"""

from __future__ import annotations

from .retrieval import RetrievalAgent


class PredictorAgent(RetrievalAgent):
    arm = "predictor"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "the predictor arm is not implemented yet; run --arm null or --arm retrieval. "
            "Implement predict() on memory.structured and call it from on_proposal/on_observation."
        )

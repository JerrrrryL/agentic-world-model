"""The llm arm: retrieval + written objections, grounded reports, requests, agent-raised decisions.

Not implemented in this drop. The runtime, protocol, and memory are arm-agnostic,
so this class only has to fill the four hooks; every sentence it emits must cite
``{path, locator}`` evidence or the runtime's grounding lint will replace it with
the bundle. Any external model call belongs here, made with the sidecar's own
credentials — never the scientist's.
"""

from __future__ import annotations

from .retrieval import RetrievalAgent


class LLMAgent(RetrievalAgent):
    arm = "llm"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "the llm arm is not implemented yet; run --arm null or --arm retrieval. "
            "Implement on_proposal (objections), on_observation (grounded report, "
            "request_evaluators, agent-raised decision) and on_close (lint-checked note)."
        )

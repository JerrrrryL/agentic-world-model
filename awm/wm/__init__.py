"""World-model agent runtime: the protocol between a scientist and an advising agent.

See doc/spec/2026-08-30-world-model-agent.md. The scientist issues cards,
yields the GPU, and answers pings; the runtime here validates, freezes,
evaluates, seals, and keeps the ledger; the agent (``awm.wm.agents``) decides
what to say.
"""

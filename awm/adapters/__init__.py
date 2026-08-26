"""Adapters: give a benchmark a runner it does not ship.

Design spec section 2 (D2): where upstream publishes only task definitions, we borrow
Harbor's task-directory format and generate one directory per task. The scoring logic is
never reimplemented -- the generated ``tests/test.sh`` calls the upstream ``evaluate.py``.
"""

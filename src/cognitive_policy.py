"""Shared utilities for Part 3 cognitive prompt packets.

The active causal policy lives in :mod:`src.part3_closed_loop`. This module is
kept deliberately small so packet builders share one context-size estimate
without retaining the superseded mock-policy scaffold.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def estimate_context_tokens(context: Mapping[str, Any]) -> int:
    """Return the conservative character-based estimate used by Part 3 jobs."""

    return max(64, int(len(json.dumps(context, default=str)) / 4))

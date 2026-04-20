"""
Collator base for LeRobot-backed runs.

Samples are still :class:`PreferenceSample` / :class:`ProgressSample` after dataset materialization;
this module exists so training code can swap imports explicitly (``BaseCollatorLRB3`` vs ``BaseCollator``).
"""

from __future__ import annotations

from robometer.data.collators.base import BaseCollator


class BaseCollatorLRB3(BaseCollator):
    """Drop-in alias of :class:`BaseCollator` for LeRobot v3 data pipelines."""

    pass

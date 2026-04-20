"""RBM head collator for LeRobot-backed runs (same behavior as :class:`RBMBatchCollator`)."""

from __future__ import annotations

from robometer.data.collators.rbm_heads import RBMBatchCollator


class RBMBatchCollatorLRB3(RBMBatchCollator):
    """Drop-in alias of :class:`RBMBatchCollator`; use with ``RBMDatasetLRB3`` / ``BaseDatasetLRB3``."""

    pass

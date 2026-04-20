"""RBMDataset + metadata-first LeRobot v3 loading (lazy frames in samplers)."""

from __future__ import annotations

from robometer.data.datasets.base_dataset_lrb3 import BaseDatasetLRB3
from robometer.data.datasets.rbm_data import RBMDataset
from robometer.data.samplers.pref_lrb3 import PrefSamplerLRB3
from robometer.data.samplers.progress_lrb3 import ProgressSamplerLRB3


class RBMDatasetLRB3(BaseDatasetLRB3, RBMDataset):
    """Uses :class:`PrefSamplerLRB3` / :class:`ProgressSamplerLRB3` when ``ROBOMETER_LEROBOT_DATASET_ROOT`` is set."""

    _pref_sampler_cls = PrefSamplerLRB3
    _progress_sampler_cls = ProgressSamplerLRB3

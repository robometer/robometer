"""Preference sampler with lazy LeRobot frame loading."""

from __future__ import annotations

from robometer.data.samplers.lerobot_lazy_frame_mixin_lrb3 import LeRobotLazyFramesMixin
from robometer.data.samplers.pref import PrefSampler


class PrefSamplerLRB3(LeRobotLazyFramesMixin, PrefSampler):
    """Same as :class:`PrefSampler`; decodes video only inside ``_get_traj_from_data``."""

    pass

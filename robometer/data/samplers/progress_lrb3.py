"""Progress sampler with lazy LeRobot frame loading."""

from __future__ import annotations

from robometer.data.samplers.lerobot_lazy_frame_mixin_lrb3 import LeRobotLazyFramesMixin
from robometer.data.samplers.progress import ProgressSampler


class ProgressSamplerLRB3(LeRobotLazyFramesMixin, ProgressSampler):
    """Same as :class:`ProgressSampler`; decodes video only inside ``_get_traj_from_data``."""

    pass

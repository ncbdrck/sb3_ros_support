#!/bin/python3

"""
Backwards-compat shim for ``sb3_ros_support.ddpg_goal.DDPG_GOAL``.

Goal-conditioned DDPG support was consolidated into the single
:class:`sb3_ros_support.ddpg.DDPG` class. ``DDPG`` now auto-detects
the policy type from the env's observation space
(:class:`gymnasium.spaces.Dict` → ``"MultiInputPolicy"``, otherwise
``"MlpPolicy"``) and enables Hindsight Experience Replay (HER) via
the ``use_her`` constructor flag or the YAML config's ``use_HER``
key.

``DDPG_GOAL`` remains a working alias for backwards compatibility
but emits a :class:`DeprecationWarning` on instantiation. New code
should import :class:`sb3_ros_support.ddpg.DDPG` directly.
"""

import warnings

from sb3_ros_support.ddpg import DDPG


class DDPG_GOAL(DDPG):
    """Deprecated. Use :class:`sb3_ros_support.ddpg.DDPG` with ``use_her=True``."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DDPG_GOAL is deprecated. Use sb3_ros_support.ddpg.DDPG instead: "
            "the policy type is auto-detected from the env's observation "
            "space, and HER is enabled via use_her=True (or YAML "
            "use_HER: true).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["DDPG_GOAL"]

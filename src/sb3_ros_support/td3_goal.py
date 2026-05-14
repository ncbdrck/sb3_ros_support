#!/bin/python3

"""
Backwards-compat shim for ``sb3_ros_support.td3_goal.TD3_GOAL``.

Goal-conditioned TD3 support was consolidated into the single
:class:`sb3_ros_support.td3.TD3` class. ``TD3`` now auto-detects the
policy type from the env's observation space
(:class:`gymnasium.spaces.Dict` → ``"MultiInputPolicy"``, otherwise
``"MlpPolicy"``) and enables Hindsight Experience Replay (HER) via
the ``use_her`` constructor flag or the YAML config's ``use_HER``
key.

``TD3_GOAL`` remains a working alias for backwards compatibility but
emits a :class:`DeprecationWarning` on instantiation. New code should
import :class:`sb3_ros_support.td3.TD3` directly.
"""

import warnings

from sb3_ros_support.td3 import TD3


class TD3_GOAL(TD3):
    """Deprecated. Use :class:`sb3_ros_support.td3.TD3` with ``use_her=True``."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TD3_GOAL is deprecated. Use sb3_ros_support.td3.TD3 instead: "
            "the policy type is auto-detected from the env's observation "
            "space, and HER is enabled via use_her=True (or YAML "
            "use_HER: true).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["TD3_GOAL"]

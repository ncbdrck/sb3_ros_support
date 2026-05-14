#!/bin/python3

"""
Backwards-compat shim for ``sb3_ros_support.dqn_goal.DQN_GOAL``.

Goal-conditioned DQN support was consolidated into the single
:class:`sb3_ros_support.dqn.DQN` class. ``DQN`` now auto-detects the
policy type from the env's observation space
(:class:`gymnasium.spaces.Dict` → ``"MultiInputPolicy"``, otherwise
``"MlpPolicy"``) and enables Hindsight Experience Replay (HER) via
the ``use_her`` constructor flag or the YAML config's ``use_HER``
key.

``DQN_GOAL`` remains a working alias for backwards compatibility but
emits a :class:`DeprecationWarning` on instantiation. New code should
import :class:`sb3_ros_support.dqn.DQN` directly.
"""

import warnings

from sb3_ros_support.dqn import DQN


class DQN_GOAL(DQN):
    """Deprecated. Use :class:`sb3_ros_support.dqn.DQN` with ``use_her=True``."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DQN_GOAL is deprecated. Use sb3_ros_support.dqn.DQN instead: "
            "the policy type is auto-detected from the env's observation "
            "space, and HER is enabled via use_her=True (or YAML "
            "use_HER: true).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["DQN_GOAL"]

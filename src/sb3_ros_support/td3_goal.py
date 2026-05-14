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
from typing import Any, Optional

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

    @staticmethod
    def load_trained_model(model_path: str, model_pkg: Optional[str] = None,
                           env: Optional[Any] = None,
                           config_file_pkg: Optional[str] = None,
                           config_filename: Optional[str] = None,
                           abs_config_path: Optional[str] = None,
                           use_her: bool = True) -> "TD3_GOAL":
        """Backwards-compat loader preserving the pre-cleanup defaults.

        Defaults ``config_filename`` to ``td3_goal.yaml`` and ``use_her``
        to ``True`` so legacy callers using
        ``TD3_GOAL.load_trained_model(path, env=goal_env)`` keep the
        goal-conditioned behaviour instead of silently picking up
        ``td3.yaml`` with HER disabled.
        """
        if config_file_pkg is None and config_filename is None and abs_config_path is None:
            config_file_pkg = "sb3_ros_support"
            config_filename = "td3_goal.yaml"
        elif model_pkg is not None and config_filename is not None and config_file_pkg is None:
            config_file_pkg = model_pkg

        return TD3_GOAL(
            env=env, save_model_path=model_path, log_path=model_path,
            model_pkg_path=model_pkg, load_trained=True,
            load_model_path=model_path, config_file_pkg=config_file_pkg,
            config_filename=config_filename, abs_config_path=abs_config_path,
            use_her=use_her,
        )


__all__ = ["TD3_GOAL"]

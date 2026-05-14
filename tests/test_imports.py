"""
Smoke tests for the sb3_ros_support public API.

These tests don't yet cover behavioural correctness — we haven't
landed any sb3_ros_support-specific bug fixes in the cleanup
rounds. They guard against the obvious failure mode: imports break
under refactoring.

When future cleanup work targets sb3_ros_support (e.g. the proposed
goal/non-goal algorithm-pair consolidation), add behavioural tests
alongside these smoke tests.
"""
import inspect


ALGORITHMS_STANDALONE = ["a2c", "ppo", "dqn", "ddpg", "sac", "td3"]
ALGORITHMS_GOAL = ["dqn_goal", "ddpg_goal", "sac_goal", "td3_goal"]


class TestBasicModelExists:
    def test_core_basic_model_importable(self):
        from sb3_ros_support.core import BasicModel
        assert inspect.isclass(BasicModel)

    def test_basic_model_init_accepts_env_and_paths(self):
        from sb3_ros_support.core import BasicModel
        sig = inspect.signature(BasicModel.__init__)
        params = list(sig.parameters)
        # Round-1 documentation: every algo construction goes through
        # these args. If they ever change, all algorithm subclasses
        # break in user training scripts.
        assert "env" in params
        assert "save_model_path" in params
        assert "log_path" in params


class TestAllAlgorithmModulesImport:
    """Each algorithm module must import cleanly under stubs."""

    def test_a2c_imports(self):
        from sb3_ros_support import a2c
        assert hasattr(a2c, "A2C")

    def test_ppo_imports(self):
        from sb3_ros_support import ppo
        assert hasattr(ppo, "PPO")

    def test_dqn_imports(self):
        from sb3_ros_support import dqn
        assert hasattr(dqn, "DQN")

    def test_ddpg_imports(self):
        from sb3_ros_support import ddpg
        assert hasattr(ddpg, "DDPG")

    def test_sac_imports(self):
        from sb3_ros_support import sac
        assert hasattr(sac, "SAC")

    def test_td3_imports(self):
        from sb3_ros_support import td3
        assert hasattr(td3, "TD3")


class TestGoalAlgorithmModulesImport:
    """The goal-conditioned variants (HER-paired)."""

    def test_dqn_goal_imports(self):
        from sb3_ros_support import dqn_goal
        assert hasattr(dqn_goal, "DQN_GOAL")

    def test_ddpg_goal_imports(self):
        from sb3_ros_support import ddpg_goal
        assert hasattr(ddpg_goal, "DDPG_GOAL")

    def test_sac_goal_imports(self):
        from sb3_ros_support import sac_goal
        assert hasattr(sac_goal, "SAC_GOAL")

    def test_td3_goal_imports(self):
        from sb3_ros_support import td3_goal
        assert hasattr(td3_goal, "TD3_GOAL")


class TestAlgorithmsSubclassBasicModel:
    """Every algorithm class must extend core.BasicModel — that's the
    contract user training scripts depend on."""

    def test_td3_subclasses_basic_model(self):
        from sb3_ros_support.core import BasicModel
        from sb3_ros_support.td3 import TD3
        assert issubclass(TD3, BasicModel)

    def test_sac_subclasses_basic_model(self):
        from sb3_ros_support.core import BasicModel
        from sb3_ros_support.sac import SAC
        assert issubclass(SAC, BasicModel)

    def test_ddpg_subclasses_basic_model(self):
        from sb3_ros_support.core import BasicModel
        from sb3_ros_support.ddpg import DDPG
        assert issubclass(DDPG, BasicModel)

    def test_td3_goal_subclasses_basic_model(self):
        from sb3_ros_support.core import BasicModel
        from sb3_ros_support.td3_goal import TD3_GOAL
        assert issubclass(TD3_GOAL, BasicModel)

    def test_ddpg_goal_subclasses_basic_model(self):
        from sb3_ros_support.core import BasicModel
        from sb3_ros_support.ddpg_goal import DDPG_GOAL
        assert issubclass(DDPG_GOAL, BasicModel)


class TestUtilsImports:
    def test_sb3_common_importable(self):
        from sb3_ros_support.utils import sb3_common
        assert hasattr(sb3_common, "get_policy_kwargs")
        assert hasattr(sb3_common, "get_action_noise")

    def test_yaml_utils_importable(self):
        from sb3_ros_support.utils import yaml_utils
        # Public function used by every algorithm's __init__
        assert hasattr(yaml_utils, "load_yaml")

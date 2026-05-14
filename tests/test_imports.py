"""
Smoke tests for the sb3_ros_support public API.

These tests cover the import surface and basic class identity. They
guard against the obvious failure mode: imports break under
refactoring. Algorithm-level behavioural tests would require real
training environments and are out of scope here.
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
        # Every algo construction goes through these args. If they
        # ever change, all algorithm subclasses break in user
        # training scripts.
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


class TestAlgorithmConsolidation:
    """Each *_GOAL alias is a deprecation shim that delegates to its
    base class. The base class accepts a ``use_her`` kwarg and
    auto-detects MlpPolicy vs MultiInputPolicy from the env."""

    def test_sac_accepts_use_her_kwarg(self):
        from sb3_ros_support.sac import SAC
        sig = inspect.signature(SAC.__init__)
        assert "use_her" in sig.parameters

    def test_td3_accepts_use_her_kwarg(self):
        from sb3_ros_support.td3 import TD3
        sig = inspect.signature(TD3.__init__)
        assert "use_her" in sig.parameters

    def test_ddpg_accepts_use_her_kwarg(self):
        from sb3_ros_support.ddpg import DDPG
        sig = inspect.signature(DDPG.__init__)
        assert "use_her" in sig.parameters

    def test_dqn_accepts_use_her_kwarg(self):
        from sb3_ros_support.dqn import DQN
        sig = inspect.signature(DQN.__init__)
        assert "use_her" in sig.parameters

    def test_sac_goal_is_sac_subclass(self):
        from sb3_ros_support.sac import SAC
        from sb3_ros_support.sac_goal import SAC_GOAL
        assert issubclass(SAC_GOAL, SAC)

    def test_td3_goal_is_td3_subclass(self):
        from sb3_ros_support.td3 import TD3
        from sb3_ros_support.td3_goal import TD3_GOAL
        assert issubclass(TD3_GOAL, TD3)

    def test_ddpg_goal_is_ddpg_subclass(self):
        from sb3_ros_support.ddpg import DDPG
        from sb3_ros_support.ddpg_goal import DDPG_GOAL
        assert issubclass(DDPG_GOAL, DDPG)

    def test_dqn_goal_is_dqn_subclass(self):
        from sb3_ros_support.dqn import DQN
        from sb3_ros_support.dqn_goal import DQN_GOAL
        assert issubclass(DQN_GOAL, DQN)

    def test_is_dict_obs_space_helper(self):
        """The policy auto-detection helper resolves Dict → MultiInputPolicy."""
        from unittest.mock import MagicMock
        from sb3_ros_support.utils.sb3_common import is_dict_obs_space
        try:
            import gymnasium
        except ImportError:
            import gym as gymnasium
        env = MagicMock()
        env.observation_space = gymnasium.spaces.Dict({
            "observation": gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            "achieved_goal": gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            "desired_goal": gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
        })
        assert is_dict_obs_space(env) is True

        env2 = MagicMock()
        env2.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        assert is_dict_obs_space(env2) is False


class TestUtilsImports:
    def test_sb3_common_importable(self):
        from sb3_ros_support.utils import sb3_common
        assert hasattr(sb3_common, "get_policy_kwargs")
        assert hasattr(sb3_common, "get_action_noise")

    def test_yaml_utils_importable(self):
        from sb3_ros_support.utils import yaml_utils
        # Public function used by every algorithm's __init__
        assert hasattr(yaml_utils, "load_yaml")

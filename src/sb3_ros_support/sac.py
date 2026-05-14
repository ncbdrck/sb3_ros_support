#!/bin/python3

import os
import stable_baselines3
from sb3_ros_support import core
from sb3_ros_support.utils import sb3_common, yaml_utils

# ROS packages required
import rospy
import rospkg


class SAC(core.BasicModel):
    """
    Soft Actor-Critic (SAC) with optional HER for goal-conditioned envs.

    Paper: https://arxiv.org/abs/1801.01290

    Policy selection is automatic:
      * ``gymnasium.spaces.Dict`` observation space → ``"MultiInputPolicy"``
      * anything else → ``"MlpPolicy"``

    Hindsight Experience Replay (HER) is enabled when ``use_her=True``
    or when the YAML config has ``use_HER: true``. HER is only valid
    for goal-conditioned envs.
    """

    def __init__(self, env, save_model_path, log_path, model_pkg_path=None, load_trained=False,
                 load_model_path=None, config_file_pkg=None, config_filename=None, abs_config_path=None,
                 use_her=False):
        """
        Args:
            env (gym.Env): The environment to be used.
            save_model_path (str): The path to save the model. Can be absolute or relative.
            log_path (str): The abs path to save the log. Can be absolute or relative.
            model_pkg_path (str): The package name to save or load the model.
            load_trained (bool): Whether to load a trained model or not.
            load_model_path (str): The path to load the model. Should include the model name. Can be absolute or relative.
            config_file_pkg (str): The package name of the config file. Required if abs_config_path is not provided.
            config_filename (str): The name of the config file. Required if abs_config_path is not provided.
            abs_config_path (str): The absolute path to the config file. Required if config_file_pkg and config_filename are not provided.
            use_her (bool): Whether to use Hindsight Experience Replay. Only valid for goal-conditioned envs (Dict obs space).
        """
        policy = "MultiInputPolicy" if sb3_common.is_dict_obs_space(env) else "MlpPolicy"

        rospy.loginfo("Init SAC " + policy)
        print("Init SAC " + policy)

        # --- Set the environment
        self.env = env

        # --- Set the save and log path
        if model_pkg_path is not None:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path(model_pkg_path)

            # check if the path starts with "/"
            if save_model_path[0] != "/":
                save_model_path = "/" + save_model_path
            if log_path[0] != "/":
                log_path = "/" + log_path

            # check if the path ends with "/"
            if save_model_path[-1] != "/":
                save_model_path = save_model_path + "/"
            if log_path[-1] != "/":
                log_path = log_path + "/"

            save_model_path = pkg_path + save_model_path
            log_path = pkg_path + log_path

            if load_trained:
                # check if the path starts with "/"
                if load_model_path[0] != "/":
                    load_model_path = "/" + load_model_path

                load_model_path = pkg_path + load_model_path

        # Load YAML Config File
        parm_dict = yaml_utils.load_yaml(pkg_name=config_file_pkg, file_name=config_filename,
                                         file_abs_path=abs_config_path)

        # --- Init superclass
        super().__init__(env, save_model_path, log_path, parm_dict, load_trained=load_trained)

        if load_trained:
            rospy.logwarn("Loading trained model")
            self.model = stable_baselines3.SAC.load(load_model_path, env=env)
            return

        # --- SDE for SAC
        if parm_dict["use_sde"]:
            model_sde = True
            model_sde_sample_freq = parm_dict["sde_params"]["sde_sample_freq"]
            model_use_sde_at_warmup = parm_dict["sde_params"]["use_sde_at_warmup"]
            self.action_noise = None
        else:
            model_sde = False
            model_sde_sample_freq = -1
            model_use_sde_at_warmup = False

        # --- Build kwargs shared between load and fresh-construction paths.
        p = parm_dict["sac_params"]
        common_kwargs = dict(
            verbose=1,
            action_noise=self.action_noise,
            use_sde=model_sde,
            sde_sample_freq=model_sde_sample_freq,
            use_sde_at_warmup=model_use_sde_at_warmup,
            learning_rate=p["learning_rate"],
            buffer_size=p["buffer_size"],
            learning_starts=p["learning_starts"],
            batch_size=p["batch_size"],
            tau=p["tau"],
            gamma=p["gamma"],
            gradient_steps=p["gradient_steps"],
            ent_coef=p["ent_coef"],
            target_update_interval=p["target_update_interval"],
            target_entropy=p["target_entropy"],
            train_freq=(p["train_freq"]["freq"], p["train_freq"]["unit"]),
            seed=p["seed"],
        )

        # HER replay buffer (only for goal-conditioned envs).
        her_enabled = use_her or parm_dict.get("use_HER", False)
        if her_enabled:
            common_kwargs["replay_buffer_class"] = stable_baselines3.HerReplayBuffer
            common_kwargs["replay_buffer_kwargs"] = sb3_common.her_replay_buffer_kwargs(parm_dict)

        # --- Create or load model
        if parm_dict["load_model"]:  # Load model
            model_name = parm_dict["model_name"]

            assert os.path.exists(save_model_path + model_name + ".zip"), \
                "Model {} doesn't exist".format(model_name)
            rospy.logwarn("Loading model: " + model_name)

            self.model = stable_baselines3.SAC.load(
                save_model_path + model_name, env=env, **common_kwargs,
            )

            if os.path.exists(save_model_path + model_name + "_replay_buffer.pkl"):
                rospy.logwarn("Loading replay buffer")
                self.model.load_replay_buffer(save_model_path + model_name + "_replay_buffer")
            else:
                rospy.logwarn("No replay buffer found")

        else:  # Create a new model
            rospy.logwarn("Creating new model")

            self.model = stable_baselines3.SAC(
                policy, env,
                policy_kwargs=self.policy_kwargs,
                **common_kwargs,
            )

        # --- Logger
        self.set_model_logger()

    @staticmethod
    def load_trained_model(model_path, model_pkg=None, env=None, config_file_pkg=None, config_filename=None,
                           abs_config_path=None, use_her=False):
        """
        Load a trained model. Use only with predict function, as the logs will not be saved.

        Args:
            model_path (str): The path to the trained model. Can be absolute or relative.
            model_pkg (str): The package name to load the model. Required if abs_model_path is relative.
            env (gym.Env): The environment to be used.
            config_file_pkg (str): The package name of the config file. Use the same package as model_pkg if not provided.
            config_filename (str): The name of the config file.
            abs_config_path (str): The absolute path to the config file.
            use_her (bool): Whether to use Hindsight Experience Replay. Only valid for goal-conditioned envs.
        Returns:
            model: The loaded model.
        """

        if config_file_pkg is None and config_filename is None and abs_config_path is None:
            config_file_pkg = "sb3_ros_support"
            config_filename = "sac.yaml"

            rospy.logwarn("Using default config file: " + config_filename + " from package: " + config_file_pkg)

        elif model_pkg is not None and config_filename is not None and config_file_pkg is None:
            config_file_pkg = model_pkg

        model = SAC(env=env, save_model_path=model_path, log_path=model_path, model_pkg_path=model_pkg,
                    load_trained=True, load_model_path=model_path, config_file_pkg=config_file_pkg,
                    config_filename=config_filename, abs_config_path=abs_config_path,
                    use_her=use_her)

        return model

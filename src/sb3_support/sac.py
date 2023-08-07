#!/bin/python3

import os
import stable_baselines3
from sb3_support import core
from sb3_support.utils import yaml_utils

# ROS packages required
import rospy
import rospkg


class SAC(core.BasicModel):
    """
    Soft Actor-Critic (SAC) algorithm.

    Paper: https://arxiv.org/abs/1801.01290
    """

    def __init__(self, env, save_model_path, log_path, model_pkg_path=None, load_trained=False,
                 load_model_path=None, config_file_pkg=None, config_filename=None, abs_config_path=None):
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
        """

        rospy.loginfo("Init SAC Policy")
        print("Init SAC Policy")

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

        # --- Init super class
        super().__init__(env, save_model_path, log_path, parm_dict, load_trained=load_trained)

        if load_trained:
            rospy.logwarn("Loading trained model")
            self.model = stable_baselines3.SAC.load(load_model_path, env=env)
        else:
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

            # --- SAC model parameters
            model_learning_rate = parm_dict["sac_params"]["learning_rate"]
            model_buffer_size = parm_dict["sac_params"]["buffer_size"]
            model_learning_starts = parm_dict["sac_params"]["learning_starts"]
            model_batch_size = parm_dict["sac_params"]["batch_size"]
            model_tau = parm_dict["sac_params"]["tau"]
            model_gamma = parm_dict["sac_params"]["gamma"]
            model_gradient_steps = parm_dict["sac_params"]["gradient_steps"]
            model_ent_coef = parm_dict["sac_params"]["ent_coef"]
            model_target_update_interval = parm_dict["sac_params"]["target_update_interval"]
            model_target_entropy = parm_dict["sac_params"]["target_entropy"]
            model_train_freq_freq = parm_dict["sac_params"]["train_freq"]["freq"]
            model_train_freq_unit = parm_dict["sac_params"]["train_freq"]["unit"]

            # --- Create or load model
            if parm_dict["load_model"]:  # Load model
                model_name = parm_dict["model_name"]

                assert os.path.exists(save_model_path + model_name + ".zip"), "Model {} doesn't exist".format(
                    model_name)
                rospy.logwarn("Loading model: " + model_name)

                self.model = stable_baselines3.SAC.load(save_model_path + model_name, env=env, verbose=1,
                                                        action_noise=self.action_noise,
                                                        use_sde=model_sde, sde_sample_freq=model_sde_sample_freq,
                                                        use_sde_at_warmup=model_use_sde_at_warmup,
                                                        learning_rate=model_learning_rate,
                                                        buffer_size=model_buffer_size,
                                                        learning_starts=model_learning_starts,
                                                        batch_size=model_batch_size, tau=model_tau, gamma=model_gamma,
                                                        gradient_steps=model_gradient_steps,
                                                        ent_coef=model_ent_coef,
                                                        target_update_interval=model_target_update_interval,
                                                        target_entropy=model_target_entropy,
                                                        train_freq=(model_train_freq_freq, model_train_freq_unit))

                if os.path.exists(save_model_path + model_name + "_replay_buffer.pkl"):
                    rospy.logwarn("Loading replay buffer")
                    self.model.load_replay_buffer(save_model_path + model_name + "_replay_buffer")
                else:
                    rospy.logwarn("No replay buffer found")

            else:  # Create new model
                rospy.logwarn("Creating new model")

                self.model = stable_baselines3.SAC("MlpPolicy", env, verbose=1, action_noise=self.action_noise,
                                                   use_sde=model_sde, sde_sample_freq=model_sde_sample_freq,
                                                   use_sde_at_warmup=model_use_sde_at_warmup,
                                                   learning_rate=model_learning_rate, buffer_size=model_buffer_size,
                                                   learning_starts=model_learning_starts,
                                                   batch_size=model_batch_size, tau=model_tau, gamma=model_gamma,
                                                   gradient_steps=model_gradient_steps,
                                                   policy_kwargs=self.policy_kwargs, ent_coef=model_ent_coef,
                                                   target_update_interval=model_target_update_interval,
                                                   target_entropy=model_target_entropy,
                                                   train_freq=(model_train_freq_freq, model_train_freq_unit))

            # --- Logger
            self.set_model_logger()


def load_trained_model(model_path, model_pkg_path=None, env=None):
    """
    Load a trained model. Use only with predict function, as the logs will not be saved.

    Args:
        model_path (str): The path to the trained model.
        model_pkg_path (str): The package name to load the model.
        env (gym.Env): The environment to be used.
    Returns:
        model: The loaded model.
    """

    model = SAC(env=env, save_model_path=model_path, log_path=model_path, load_model_path=model_path,
                model_pkg_path=model_pkg_path, load_trained=True)

    return model

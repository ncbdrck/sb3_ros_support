#!/bin/python3

import os
import stable_baselines3
from sb3_support import core
from sb3_support.utils import yaml_utils

# ROS packages required
import rospy
import rospkg


class A2C(core.BasicModel):
    """
    Advantage Actor-Critic (A2C) algorithm.

    Paper: https://arxiv.org/abs/1602.01783
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

        rospy.loginfo("Init A2C Policy")
        print("Init A2C Policy")

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
            self.model = stable_baselines3.A2C.load(load_model_path, env=env)
        else:
            # --- SDE for A2C
            if parm_dict["use_sde"]:
                model_sde = True
                model_sde_sample_freq = parm_dict["sde_params"]["sde_sample_freq"]
                self.action_noise = None
            else:
                model_sde = False
                model_sde_sample_freq = -1

            # --- A2C model parameters
            model_learning_rate = parm_dict["a2c_params"]["learning_rate"]
            model_n_steps = parm_dict["a2c_params"]["n_steps"]
            model_gamma = parm_dict["a2c_params"]["gamma"]
            model_gae_lambda = parm_dict["a2c_params"]["gae_lambda"]
            model_ent_coef = parm_dict["a2c_params"]["ent_coef"]
            model_vf_coef = parm_dict["a2c_params"]["vf_coef"]
            model_max_grad_norm = parm_dict["a2c_params"]["max_grad_norm"]
            model_use_rms_prop = parm_dict["a2c_params"]["use_rms_prop"]
            model_rms_prop_eps = parm_dict["a2c_params"]["rms_prop_eps"]
            model_norm_advant = parm_dict["a2c_params"]["normalize_advantage"]

            # --- Create or load model
            if parm_dict["load_model"]:  # Load model
                model_name = parm_dict["model_name"]

                assert os.path.exists(save_model_path + model_name + ".zip"), "Model {} doesn't exist".format(
                    model_name)
                rospy.logwarn("Loading model: " + model_name)

                self.model = stable_baselines3.A2C.load(save_model_path + model_name, env=env, verbose=1,
                                                        learning_rate=model_learning_rate,
                                                        n_steps=model_n_steps, gamma=model_gamma,
                                                        gae_lambda=model_gae_lambda, ent_coef=model_ent_coef,
                                                        vf_coef=model_vf_coef, max_grad_norm=model_max_grad_norm,
                                                        use_sde=model_sde, sde_sample_freq=model_sde_sample_freq,
                                                        use_rms_prop=model_use_rms_prop,
                                                        rms_prop_eps=model_rms_prop_eps,
                                                        normalize_advantage=model_norm_advant)

                if os.path.exists(save_model_path + model_name + "_replay_buffer.pkl"):
                    rospy.logwarn("Loading replay buffer")
                    self.model.load_replay_buffer(save_model_path + model_name + "_replay_buffer")
                else:
                    rospy.logwarn("No replay buffer found")

            else:  # Create new model
                rospy.logwarn("Creating new model")
                self.model = stable_baselines3.A2C("MlpPolicy", env, verbose=1, policy_kwargs=self.policy_kwargs,
                                                   learning_rate=model_learning_rate, n_steps=model_n_steps,
                                                   gamma=model_gamma,
                                                   gae_lambda=model_gae_lambda, ent_coef=model_ent_coef,
                                                   vf_coef=model_vf_coef, max_grad_norm=model_max_grad_norm,
                                                   use_sde=model_sde, sde_sample_freq=model_sde_sample_freq,
                                                   use_rms_prop=model_use_rms_prop, rms_prop_eps=model_rms_prop_eps,
                                                   normalize_advantage=model_norm_advant)

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

    model = A2C(env=env, save_model_path=model_path, log_path=model_path, load_model_path=model_path,
                model_pkg_path=model_pkg_path, load_trained=True)

    return model

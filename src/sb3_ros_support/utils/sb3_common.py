#!/bin/python3

import numpy as np
import torch as th
import stable_baselines3

# ROS packages required
import rospy

# Noise
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env

# Callbacks
from stable_baselines3.common.callbacks import BaseCallback
import time


def get_policy_kwargs(parm_dict: dict) -> dict:
    """
    Function to get the policy kwargs from the parm_dict.

    Args:
        parm_dict (dict): The dictionary containing the parameters.

    Returns:
        dict: Dictionary containing the policy kwargs.
    """

    if parm_dict["use_custom_policy"]:
        # Activation function for the policy
        activation_function = parm_dict["policy_params"]["activation_fn"].lower()
        if activation_function == "relu":
            activation_fn = th.nn.ReLU
        elif activation_function == "tanh":
            activation_fn = th.nn.Tanh
        elif activation_function == "elu":
            activation_fn = th.nn.ELU
        elif activation_function == "selu":
            activation_fn = th.nn.SELU
        else:
            rospy.logwarn("Activation function not found, using ReLU")
            activation_fn = th.nn.ReLU

        # Feature extractor for the policy
        feature_extractor = parm_dict["policy_params"]["features_extractor_class"]
        if feature_extractor == "FlattenExtractor":
            features_extractor_class = stable_baselines3.common.torch_layers.FlattenExtractor
        elif feature_extractor == "BaseFeaturesExtractor":
            features_extractor_class = stable_baselines3.common.torch_layers.BaseFeaturesExtractor
        elif feature_extractor == "CombinedExtractor":
            features_extractor_class = stable_baselines3.common.torch_layers.CombinedExtractor
        else:
            rospy.logwarn("Feature extractor not found, using FlattenExtractor")
            features_extractor_class = stable_baselines3.common.torch_layers.FlattenExtractor

        # Optimizer for the policy
        optimizer_class = parm_dict["policy_params"]["optimizer_class"]
        if optimizer_class == "Adam":
            optimizer_class = th.optim.Adam
        elif optimizer_class == "SGD":
            optimizer_class = th.optim.SGD
        elif optimizer_class == "RMSprop":
            optimizer_class = th.optim.RMSprop
        elif optimizer_class == "Adagrad":
            optimizer_class = th.optim.Adagrad
        elif optimizer_class == "Adadelta":
            optimizer_class = th.optim.Adadelta
        else:
            rospy.logwarn("Optimizer not found, using Adam")
            optimizer_class = th.optim.Adam

        # Net Archiecture for the policy
        net_arch = parm_dict["policy_params"]["net_arch"]

        policy_kwargs = dict(activation_fn=activation_fn, features_extractor_class=features_extractor_class,
                             optimizer_class=optimizer_class, net_arch=net_arch)

        # log
        rospy.logwarn(policy_kwargs)
        print(policy_kwargs)
    else:
        policy_kwargs = None

    return policy_kwargs


def get_action_noise(action_space_shape, parm_dict: dict, action_noise_type="normal"):
    """
    Function to get the action noise from the parm_dict.

    Args:
        action_space_shape (int): The shape of the action space.
        parm_dict (dict): The dictionary containing the parameters.
        action_noise_type (str): The type of action noise to use. Can be "normal" or "ornstein".

    Returns:
        action_noise: The action noise.
    """

    action_noise = None
    if parm_dict["use_action_noise"] is None:
        rospy.loginfo("Parameter use_action_noise was not found")
        return action_noise

    if parm_dict["use_action_noise"] is True:
        action_mean = parm_dict["action_noise"]["mean"]
        action_sigma = parm_dict["action_noise"]["sigma"]

        # normal
        if action_noise_type == "normal":

            # create noise
            action_noise = NormalActionNoise(mean=action_mean * np.ones(action_space_shape),
                                             sigma=action_sigma * np.ones(action_space_shape))

        # ornstein
        elif action_noise_type == "ornstein":
            action_theta = parm_dict["action_noise"]["theta"]
            action_dt = parm_dict["action_noise"]["dt"]

            # initial noise
            if parm_dict["action_noise"]["initial_noise"] is not None:
                action_initial_noise = parm_dict["action_noise"]["initial_noise"]
            else:
                action_initial_noise = None

            # create noise
            action_noise = OrnsteinUhlenbeckActionNoise(mean=action_mean * np.ones(action_space_shape),
                                                        sigma=action_sigma * np.ones(action_space_shape),
                                                        theta=action_theta, dt=action_dt,
                                                        initial_noise=action_initial_noise)

    return action_noise


def test_env(env):
    """
    Use SB3 env checker.
    """
    check_env(env)
    return True


class TimeLimitCallback(BaseCallback):
    """
    Callback for setting an action cycle for training.
    """

    def __init__(self, action_cycle_time, verbose=0):
        """
        Args:
            action_cycle_time (float): The time in seconds for the action cycle.
            verbose (int): The verbosity level: 0 none, 1 training information, 2 debug.
        """
        super(TimeLimitCallback, self).__init__(verbose)
        self.action_cycle_time = action_cycle_time
        self.next_action_cycle = time.time() + action_cycle_time

    def _on_step(self) -> bool:
        # Wait until it's time for the next action cycle
        wait_time = max(0, self.next_action_cycle - time.time())
        time.sleep(wait_time)
        self.next_action_cycle = time.time() + self.action_cycle_time
        return True

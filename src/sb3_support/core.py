#!/bin/python3
"""
Extending the SB3 models of the frobs_rl library.

Recreated to overcome the following errors:
    - Cannot use with multiple environments
    - Support for goal conditioned environments
    - pyyaml for loading parameters
"""
import os
from datetime import datetime
from sb3_support.utils.sb3_common import get_policy_kwargs, get_action_noise

# ROS packages required
import rospy

# SB3 Callbacks
from stable_baselines3.common.callbacks import CheckpointCallback

# Logger
from stable_baselines3.common.logger import configure


class BasicModel:
    """

    Base class for all the algorithms of Stable Baselines3.
    """

    def __init__(self, env, save_model_path, log_path, load_trained=False) -> None:
        """
        Args:
            env (gym.Env): The environment to be used.
            save_model_path (str): The path to save the model.
            log_path (str): The path to save the log.
            load_trained (bool): Whether to load a trained model or not.
        """

        self.env = env
        self.save_model_path = save_model_path
        self.log_path = log_path
        self.save_trained_model_path = None
        self.model = None

        if load_trained is False:
            # --- Policy kwargs
            self.policy_kwargs = get_policy_kwargs(ns=ns)

            # --- Noise kwargs
            self.action_noise = get_action_noise(self.env.action_space.shape[-1], ns=ns)

            # --- Callback
            save_freq = rospy.get_param(ns + "/model_params/save_freq")
            save_prefix = rospy.get_param(ns + "/model_params/save_prefix")
            self.checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_model_path,
                                                          name_prefix=save_prefix)

    def train(self) -> bool:
        """
        Function to train the model the number of steps specified in the ROS parameter server.
        The function will automatically save the model after training.

        :return: True if the training was successful, False otherwise.
        :rtype: bool
        """

        training_steps = rospy.get_param(self.ns + "/model_params/training_steps")
        learn_log_int = rospy.get_param(self.ns + "/model_params/log_interval")
        learn_reset_num_tm = rospy.get_param(self.ns + "/model_params/reset_num_timesteps")

        if learn_reset_num_tm is False:
            self.env = self.model.get_env()
            self.env.reset()

        self.model.learn(total_timesteps=int(training_steps), callback=self.checkpoint_callback,
                         log_interval=learn_log_int, reset_num_timesteps=learn_reset_num_tm)

        self.save_model()

        return True

    def save_model(self) -> bool:
        """
        Function to save the model.

        :return: True if the model was saved, False otherwise.
        :rtype: bool
        """

        # --- Model name
        trained_model_name = rospy.get_param(self.ns + "/model_params/trained_model_name")

        # If file exists, name the new model with a suffix
        self.save_trained_model_path = self.save_model_path + trained_model_name
        if os.path.isfile(self.save_model_path + trained_model_name + ".zip"):
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            self.save_trained_model_path = self.save_trained_model_path + "_" + dt_string
            rospy.logwarn("Trained model name already exists, saving as: " + trained_model_name + "_" + dt_string)

        self.model.save(self.save_trained_model_path)
        self.save_replay_buffer()

        return True

    def save_replay_buffer(self) -> bool:
        """
        Funtion to save the replay buffer, to be used the training must be finished or an error will be raised.

        :return: True if the replay buffer was saved, False otherwise.
        :rtype: bool
        """

        if self.save_trained_model_path is None:
            raise ValueError("Model not trained yet, cannot save replay buffer")

        if rospy.get_param(self.ns + "/model_params/save_replay_buffer"):
            rospy.logwarn("Saving replay buffer")
            self.model.save_replay_buffer(self.save_trained_model_path + '_replay_buffer')

    def set_model_logger(self) -> bool:
        """
        Function to set the logger of the model.

        :return: True if the logger was set, False otherwise.
        :rtype: bool
        """

        log_folder = rospy.get_param(self.ns + "/model_params/log_folder")
        log_path = self.log_path + log_folder
        assert not os.path.exists(log_path), "Log folder already exists, to log into that folder first delete it."
        new_logger = configure(log_path + '/', ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

        return True

    def close_env(self) -> bool:
        """
        Use the env close method to close the environment.

        :return: True if the environment was closed, False otherwise.
        :rtype: bool
        """

        self.env.close()
        return True

    def check_env(self) -> bool:
        """
        Use the stable-baselines check_env method to check the environment.

        :return: True if the environment is correct, False otherwise.
        :rtype: bool
        """

        self.env.check_env()
        return True

    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the current action based on the observation, state or mask

        :param observation: The enviroment observation
        :type observation: ndarray

        :param state: The previous states of the enviroment, used in recurrent policies.
        :type state: ndarray

        :param mask: The mask of the last states, used in recurrent policies.
        :type mask: ndarray

        :param deterministic: Whether or not to return deterministic actions.
        :type deterministic: bool

        :return: The action to be taken and the next state(for recurrent policies)
        :rtype: ndarray, ndarray
        """

        return self.model.predict(observation, state=state, deterministic=deterministic)


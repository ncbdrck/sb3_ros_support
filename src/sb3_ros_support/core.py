#!/bin/python3
"""
Extending the SB3 models of the frobs_rl library.

Recreated to overcome the following errors:
    - Cannot use with multiple environments
    - Support for goal-conditioned environments
    - pyyaml for loading parameters
"""
import os
from datetime import datetime
from sb3_ros_support.utils.sb3_common import get_policy_kwargs, get_action_noise, test_env, TimeLimitCallback

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

    def __init__(self, env, save_model_path, log_path, parm_dict, load_trained=False,
                 action_noise_type="normal") -> None:
        """
        Args:
            env (gym.Env): The environment to be used.
            save_model_path (str): The path to save the model.
            parm_dict (dict): The dictionary containing the parameters.
            log_path (str): The path to save the log.
            load_trained (bool): Whether to load a trained model or not.
            action_noise_type (str): The type of action noise to use. Can be "normal" or "ornstein". (Optional)
        """

        self.env = env
        self.save_model_path = save_model_path
        self.log_path = log_path
        self.save_trained_model_path = None
        self.model = None
        self.parm_dict = parm_dict

        if load_trained is False:
            # --- Policy kwargs
            self.policy_kwargs = get_policy_kwargs(parm_dict)

            # --- Noise kwargs
            self.action_noise = get_action_noise(self.env.action_space.shape[-1], parm_dict, action_noise_type)

            # --- Callback
            save_freq = parm_dict["save_freq"]
            save_prefix = parm_dict["save_prefix"]
            self.checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_model_path,
                                                          name_prefix=save_prefix)

    def train(self, action_cycle_time=None) -> bool:
        """
        Function to train the model the number of steps specified in the yaml config file.
        The function will automatically save the model after training.

        Args:
            action_cycle_time (float): The time to wait between actions. (Optional)

        Returns:
            bool: True if the model was trained, False otherwise.
        """
        training_steps = self.parm_dict["training_steps"]
        learn_log_int = self.parm_dict["log_interval"]
        learn_reset_num_tm = self.parm_dict["reset_num_timesteps"]

        if learn_reset_num_tm is False:
            self.env = self.model.get_env()
            self.env.reset()

        # Create the list of callbacks
        callbacks = [self.checkpoint_callback]

        if action_cycle_time is not None:
            # Create the callback
            time_limit_callback = TimeLimitCallback(action_cycle_time=action_cycle_time)
            callbacks.append(time_limit_callback)

        self.model.learn(total_timesteps=int(training_steps), callback=callbacks,
                         log_interval=learn_log_int, reset_num_timesteps=learn_reset_num_tm)

        self.save_model()

        return True

    def save_model(self) -> bool:
        """
        Function to save the model.

        Returns:
            bool: True if the model was saved, False otherwise.
        """

        # --- Model name
        trained_model_name = self.parm_dict["trained_model_name"]

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

    def save_replay_buffer(self):
        """
        Funtion to save the replay buffer, to be used the training must be finished or an error will be raised.

        Returns:
            bool: True if the replay buffer was saved, False otherwise.
        """

        if self.save_trained_model_path is None:
            raise ValueError("Model not trained yet, cannot save replay buffer")

        if self.parm_dict["save_replay_buffer"]:
            rospy.logwarn("Saving replay buffer")
            self.model.save_replay_buffer(self.save_trained_model_path + '_replay_buffer')

    def set_model_logger(self) -> bool:
        """
        Function to set a logger of the model.

        Returns:
            bool: True if the logger was set, False otherwise.
        """
        log_folder = self.parm_dict["log_folder"]
        log_path = self.log_path + log_folder
        assert not os.path.exists(log_path), "Log folder already exists, to log into that folder first delete it."
        new_logger = configure(log_path + '/', ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

        return True

    def close_env(self) -> bool:
        """
        Use the env close method to close the environment.

        Returns:
            bool: True if the environment was closed, False otherwise.
        """

        self.env.close()
        return True

    def check_env(self) -> bool:
        """
        Use the stable-baselines check_env method to check the environment.

        Returns:
            bool: True if the environment was checked, False otherwise.
        """
        test_env(self.env)
        return True

    def predict(self, observation, state=None, deterministic=False):
        """
        Get the current action based on the observation, state or mask

        Args:
            observation (ndarray): The environment observation.
            state (ndarray): The previous states of the environment, used in recurrent policies. (Optional)
            deterministic (bool): Whether to return deterministic actions or not. (Optional)

        Returns:
            ndarray: The action to be taken.
        """

        return self.model.predict(observation, state=state, deterministic=deterministic)


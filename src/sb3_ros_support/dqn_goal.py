#!/bin/python3

import os
import stable_baselines3
from sb3_ros_support import core
from sb3_ros_support.utils import yaml_utils

# ROS packages required
import rospy
import rospkg


class DQN_GOAL(core.BasicModel):
    """
    Deep Q Network (DQN) algorithm.

    Paper: https://arxiv.org/abs/1312.5602
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
            use_her (bool): Whether to use Hindsight Experience Replay or not.
        """

        rospy.loginfo("Init DQN MultiInputPolicy")
        print("Init DQN MultiInputPolicy")

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
            self.model = stable_baselines3.DQN.load(load_model_path, env=env)
        else:
            # --- DQN model parameters
            model_learning_rate = parm_dict["dqn_params"]["learning_rate"]
            model_buffer_size = parm_dict["dqn_params"]["buffer_size"]
            model_learning_starts = parm_dict["dqn_params"]["learning_starts"]
            model_batch_size = parm_dict["dqn_params"]["batch_size"]
            model_tau = parm_dict["dqn_params"]["tau"]
            model_gamma = parm_dict["dqn_params"]["gamma"]
            model_gradient_steps = parm_dict["dqn_params"]["gradient_steps"]
            model_train_freq_freq = parm_dict["dqn_params"]["train_freq"]["freq"]
            model_train_freq_unit = parm_dict["dqn_params"]["train_freq"]["unit"]
            model_target_update_interval = parm_dict["dqn_params"]["target_update_interval"]
            model_exploration_fraction = parm_dict["dqn_params"]["exploration_fraction"]
            model_exploration_initial_eps = parm_dict["dqn_params"]["exploration_initial_eps"]
            model_exploration_final_eps = parm_dict["dqn_params"]["exploration_final_eps"]
            model_max_grad_norm = parm_dict["dqn_params"]["max_grad_norm"]

            # --- Create or load model
            if parm_dict["load_model"]:  # Load model
                model_name = parm_dict["model_name"]

                assert os.path.exists(save_model_path + model_name + ".zip"), "Model {} doesn't exist".format(
                    model_name)
                rospy.logwarn("Loading model: " + model_name)

                if use_her or parm_dict["use_HER"]:
                    # HER parameters
                    if "n_sampled_goal" in parm_dict["her_params"]:
                        n_sampled_goal = parm_dict["her_params"]["n_sampled_goal"]
                    else:
                        n_sampled_goal = 4

                    if "goal_selection_strategy" in parm_dict["her_params"]:
                        goal_selection_strategy = parm_dict["her_params"]["goal_selection_strategy"]
                    else:
                        goal_selection_strategy = "future"

                    if "max_episode_length" in parm_dict["her_params"]:
                        max_episode_length = parm_dict["her_params"]["max_episode_length"]
                    else:
                        max_episode_length = None

                    if "online_sampling" in parm_dict["her_params"]:
                        online_sampling = parm_dict["her_params"]["online_sampling"]
                    else:
                        online_sampling = True

                    self.model = stable_baselines3.DQN.load(save_model_path + model_name, env=env, verbose=1,
                                                            learning_rate=model_learning_rate,
                                                            buffer_size=model_buffer_size,
                                                            learning_starts=model_learning_starts,
                                                            batch_size=model_batch_size,
                                                            tau=model_tau, gamma=model_gamma,
                                                            gradient_steps=model_gradient_steps,
                                                            target_update_interval=model_target_update_interval,
                                                            exploration_fraction=model_exploration_fraction,
                                                            exploration_initial_eps=model_exploration_initial_eps,
                                                            exploration_final_eps=model_exploration_final_eps,
                                                            max_grad_norm=model_max_grad_norm,
                                                            train_freq=(model_train_freq_freq, model_train_freq_unit),

                                                            replay_buffer_class=stable_baselines3.HerReplayBuffer,
                                                            replay_buffer_kwargs=dict(
                                                                n_sampled_goal=n_sampled_goal,
                                                                goal_selection_strategy=goal_selection_strategy,
                                                                max_episode_length=max_episode_length,
                                                                online_sampling=online_sampling, )
                                                            )

                else:

                    self.model = stable_baselines3.DQN.load(save_model_path + model_name, env=env, verbose=1,
                                                            learning_rate=model_learning_rate,
                                                            buffer_size=model_buffer_size,
                                                            learning_starts=model_learning_starts,
                                                            batch_size=model_batch_size,
                                                            tau=model_tau, gamma=model_gamma,
                                                            gradient_steps=model_gradient_steps,
                                                            target_update_interval=model_target_update_interval,
                                                            exploration_fraction=model_exploration_fraction,
                                                            exploration_initial_eps=model_exploration_initial_eps,
                                                            exploration_final_eps=model_exploration_final_eps,
                                                            max_grad_norm=model_max_grad_norm,
                                                            train_freq=(model_train_freq_freq, model_train_freq_unit))

                if os.path.exists(save_model_path + model_name + "_replay_buffer.pkl"):
                    rospy.logwarn("Loading replay buffer")
                    self.model.load_replay_buffer(save_model_path + model_name + "_replay_buffer")
                else:
                    rospy.logwarn("No replay buffer found")

            else:  # Create a new model
                rospy.logwarn("Creating new model")

                if use_her or parm_dict["use_HER"]:
                    # HER parameters
                    if "n_sampled_goal" in parm_dict["her_params"]:
                        n_sampled_goal = parm_dict["her_params"]["n_sampled_goal"]
                    else:
                        n_sampled_goal = 4

                    if "goal_selection_strategy" in parm_dict["her_params"]:
                        goal_selection_strategy = parm_dict["her_params"]["goal_selection_strategy"]
                    else:
                        goal_selection_strategy = "future"

                    if "max_episode_length" in parm_dict["her_params"]:
                        max_episode_length = parm_dict["her_params"]["max_episode_length"]
                    else:
                        max_episode_length = None

                    if "online_sampling" in parm_dict["her_params"]:
                        online_sampling = parm_dict["her_params"]["online_sampling"]
                    else:
                        online_sampling = True

                    self.model = stable_baselines3.DQN("MultiInputPolicy", env, verbose=1,
                                                       learning_rate=model_learning_rate,
                                                       buffer_size=model_buffer_size,
                                                       learning_starts=model_learning_starts,
                                                       batch_size=model_batch_size,
                                                       tau=model_tau, gamma=model_gamma,
                                                       gradient_steps=model_gradient_steps,
                                                       target_update_interval=model_target_update_interval,
                                                       exploration_fraction=model_exploration_fraction,
                                                       exploration_initial_eps=model_exploration_initial_eps,
                                                       exploration_final_eps=model_exploration_final_eps,
                                                       max_grad_norm=model_max_grad_norm,
                                                       policy_kwargs=self.policy_kwargs,
                                                       train_freq=(model_train_freq_freq, model_train_freq_unit),

                                                       replay_buffer_class=stable_baselines3.HerReplayBuffer,
                                                       replay_buffer_kwargs=dict(
                                                           n_sampled_goal=n_sampled_goal,
                                                           goal_selection_strategy=goal_selection_strategy,
                                                           max_episode_length=max_episode_length,
                                                           online_sampling=online_sampling, )
                                                       )

                else:

                    self.model = stable_baselines3.DQN("MultiInputPolicy", env, verbose=1,
                                                       learning_rate=model_learning_rate,
                                                       buffer_size=model_buffer_size,
                                                       learning_starts=model_learning_starts,
                                                       batch_size=model_batch_size,
                                                       tau=model_tau, gamma=model_gamma,
                                                       gradient_steps=model_gradient_steps,
                                                       target_update_interval=model_target_update_interval,
                                                       exploration_fraction=model_exploration_fraction,
                                                       exploration_initial_eps=model_exploration_initial_eps,
                                                       exploration_final_eps=model_exploration_final_eps,
                                                       max_grad_norm=model_max_grad_norm,
                                                       policy_kwargs=self.policy_kwargs,
                                                       train_freq=(model_train_freq_freq, model_train_freq_unit))

            # --- Logger
            self.set_model_logger()

    @staticmethod
    def load_trained_model(model_path, model_pkg=None, env=None, config_file_pkg=None, config_filename=None,
                           abs_config_path=None):
        """
        Load a trained model. Use only with predict function, as the logs will not be saved.

        Args:
            model_path (str): The path to the trained model. Can be absolute or relative.
            model_pkg (str): The package name to load the model. Required if abs_model_path is relative.
            env (gym.Env): The environment to be used.
            config_file_pkg (str): The package name of the config file. Use the same package as model_pkg if not provided.
            config_filename (str): The name of the config file.
            abs_config_path (str): The absolute path to the config file.
        Returns:
            model: The loaded model.
        """

        if config_file_pkg is None and config_filename is None and abs_config_path is None:
            config_file_pkg = "sb3_ros_support"
            config_filename = "td3.yaml"

            rospy.logwarn("Using default config file: " + config_filename + " from package: " + config_file_pkg)

        elif model_pkg is not None and config_filename is not None and config_file_pkg is None:
            config_file_pkg = model_pkg

        model = DQN_GOAL(env=env, save_model_path=model_path, log_path=model_path, model_pkg_path=model_pkg,
                         load_trained=True, load_model_path=model_path, config_file_pkg=config_file_pkg,
                         config_filename=config_filename, abs_config_path=abs_config_path)

        return model

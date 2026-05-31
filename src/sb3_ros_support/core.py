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
from typing import Any, Dict, Optional, Tuple

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

    def __init__(self, env: Any, save_model_path: str, log_path: str,
                 parm_dict: Dict[str, Any], load_trained: bool = False,
                 action_noise_type: str = "normal", action_noise: bool = True,
                 seed: Optional[int] = None) -> None:
        """
        Args:
            env (gym.Env): The environment to be used.
            save_model_path (str): The path to save the model.
            parm_dict (dict): The dictionary containing the parameters.
            log_path (str): The path to save the log.
            load_trained (bool): Whether to load a trained model or not.
            action_noise_type (str): The type of action noise to use. Can be "normal" or "ornstein". (Optional)
            action_noise (bool): Whether to use action noise or not. (Optional)
            seed (int): If provided, appended to ``save_prefix`` /
                ``trained_model_name`` / ``log_folder`` as
                ``_s<seed>_<YYYYmmdd_HHMMSS>`` so every run lands in
                its own checkpoint + log directory (no clobber across
                seeds or repeat runs).
        """

        self.env = env
        self.save_model_path = save_model_path
        self.log_path = log_path
        self.save_trained_model_path = None
        self.model = None
        self.parm_dict = parm_dict
        # Remember whether this instance is a fresh training run or a reload (used so optional
        # Weights & Biases monitoring only starts a run during training, not during validation).
        self.load_trained = load_trained
        self._wandb_run = None

        # Per-run suffix used by save_prefix, trained_model_name and
        # log_folder so repeated runs (same seed or otherwise) never
        # clobber a previous run's artifacts. Frozen once at
        # construction so all three sites resolve to the same suffix.
        if seed is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_tag = f"_s{seed}_{ts}"
        else:
            self._run_tag = ""

        if load_trained is False:
            # --- Policy kwargs
            self.policy_kwargs = get_policy_kwargs(parm_dict)

            # --- Noise kwargs
            if action_noise:
                self.action_noise = get_action_noise(self.env.action_space.shape[-1], parm_dict, action_noise_type)
            else:
                self.action_noise = None

            # --- Callback
            save_freq = parm_dict["save_freq"]
            save_prefix = parm_dict["save_prefix"] + self._run_tag
            self.checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_model_path,
                                                          name_prefix=save_prefix)

    def train(self, action_cycle_time: Optional[float] = None) -> bool:
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
        trained_model_name = self.parm_dict["trained_model_name"] + self._run_tag

        # If file exists, name the new model with a further timestamp
        # suffix. (When seed was passed, ``_run_tag`` already carries a
        # construction-time stamp so collisions are extremely rare.)
        self.save_trained_model_path = self.save_model_path + trained_model_name
        if os.path.isfile(self.save_model_path + trained_model_name + ".zip"):
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            self.save_trained_model_path = self.save_trained_model_path + "_" + dt_string
            rospy.logwarn("Trained model name already exists, saving as: " + trained_model_name + "_" + dt_string)

        self.model.save(self.save_trained_model_path)
        self.save_replay_buffer()

        return True

    def save_replay_buffer(self) -> None:
        """
        Function to save the replay buffer, to be used the training must be finished or an error will be raised.

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

        The log directory is composed as
        ``<log_path>/<log_folder><run_tag>`` where ``run_tag`` is the
        per-run ``_s<seed>_<timestamp>`` suffix (empty when no seed was
        supplied). If the resulting directory already exists (e.g. two
        runs collide within a one-second timestamp window), an extra
        timestamp segment is appended so the new logger never writes
        over an existing run's TensorBoard data.

        Returns:
            bool: True if the logger was set, False otherwise.
        """
        log_folder = self.parm_dict["log_folder"] + self._run_tag
        log_path = self.log_path + log_folder
        if os.path.exists(log_path):
            log_path = log_path + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Start the optional W&B run BEFORE the TensorBoard writer is created so that, with
        # sync_tensorboard=True, every metric SB3 writes to TensorBoard is mirrored to W&B
        # without any per-algorithm changes.
        self._maybe_init_wandb(log_path)
        new_logger = configure(log_path + '/', ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

        return True

    def _maybe_init_wandb(self, log_dir: str) -> None:
        """
        Start a Weights & Biases run that mirrors the TensorBoard metrics, if enabled.

        Opt-in via the config: set ``use_wandb: True`` (and optionally a ``wandb_params`` block
        with ``project`` / ``entity``). Off by default, so TensorBoard remains the standalone
        local option. Skipped for reloaded models (validation) and degrades to a warning if the
        ``wandb`` package is not installed, so training never breaks on account of monitoring.
        """
        if self.load_trained:
            return
        if not self.parm_dict.get("use_wandb", False):
            return

        try:
            import wandb
        except ImportError:
            rospy.logwarn("use_wandb is set but the 'wandb' package is not installed; "
                          "skipping W&B (install with: pip install wandb).")
            return

        # configure() creates this dir later; make it now so wandb.init(dir=...) is happy.
        os.makedirs(log_dir, exist_ok=True)
        wandb_params = self.parm_dict.get("wandb_params") or {}
        self._wandb_run = wandb.init(
            project=wandb_params.get("project", "uniros"),
            entity=wandb_params.get("entity"),
            name=self.parm_dict["log_folder"] + self._run_tag,
            config=self.parm_dict,
            sync_tensorboard=True,
            dir=log_dir,
            reinit=True,
        )
        rospy.logwarn("Weights & Biases monitoring enabled: " + str(self._wandb_run.url))

    def close_env(self) -> bool:
        """
        Use the env close method to close the environment.

        Returns:
            bool: True if the environment was closed, False otherwise.
        """

        self.env.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        return True

    def check_env(self) -> bool:
        """
        Use the stable-baselines check_env method to check the environment.

        Returns:
            bool: True if the environment was checked, False otherwise.
        """
        test_env(self.env)
        return True

    def predict(self, observation: Any, state: Optional[Any] = None,
                deterministic: bool = False) -> Tuple[Any, Any]:
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


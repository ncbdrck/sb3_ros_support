# SB3 ROS Support: The ROS Support Package for Stable Baselines3

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/uniros/badge/?version=latest)](https://uniros.readthedocs.io/en/latest/?badge=latest)

📚 **Full documentation**: [uniros.readthedocs.io](https://uniros.readthedocs.io/) (ecosystem-wide docs hosted via UniROS)

This package is a **convenience layer** for [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) users who want config-driven, ROS-aware training scripts on top of [UniROS](https://github.com/ncbdrck/UniROS) environments. It adds YAML-loaded hyperparameters, ROS package path resolution, a uniform `train` / `save_model` / `load_trained_model` / `predict` surface, and HER support for goal-conditioned envs.

> **Note**: The environments produced by UniROS / MultiROS / RealROS are **standard gymnasium environments**, so you are not locked in to this package. Plain Stable Baselines3 is the tested path here; CleanRL, Tianshou, RLlib, and custom loops should also work but are not exercised by this repo's test suite — see the [training guide](https://uniros.readthedocs.io/en/latest/guides/training.html) for examples. This package is the easiest path if you're using SB3 and want one less thing to wire up.

This package extends the functionality of SB3 models in [FRobs_RL](https://github.com/jmfajardod/frobs_rl) package to provide the following features:
 1. Support for goal-conditioned RL tasks
 2. HER (Hindsight Experience Replay) for goal-conditioned RL tasks
 3. Support for training custom environments with [RealROS](https://github.com/ncbdrck/realros) or [MultiROS](https://github.com/ncbdrck/multiros) frameworks
 4. Updated for the new version of SB3 (Stable Baselines3) which uses **gymnasium** instead of **gym**.

## Prerequisites

Before installing this package, make sure you have the following prerequisites:

### ROS Installation

This package requires a working installation of ROS. If you haven't installed ROS yet, please follow the official [ROS installation guide](http://wiki.ros.org/ROS/Installation) for your specific operating system. This package has been tested with [ROS Noetic](http://wiki.ros.org/noetic) version.

###  ROS Workspace
Before using this package, you need a ROS workspace to build and run your ROS packages. If you are using a different operating system or ROS version, make sure to adapt the commands accordingly. Follow the steps in the [official guide](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) to create a workspace if you haven't done already.

Please note that the instructions assume you are using Ubuntu 20.04 and ROS Noetic. 

## Installation

To get started, follow these steps:

1. Clone the repository:
    ```shell
    cd ~/catkin_ws/src
    git clone https://github.com/ncbdrck/sb3_ros_support.git
    cd  sb3_ros_support
    git checkout gymnasium
    ```

2. This package relies on several Python packages. You can install them by running the following command:

    ```shell
    # Install pip if you haven't already by running this command
    sudo apt-get install python3-pip

    # install the required Python packages by running
    cd ~/catkin_ws/src/sb3_ros_support/
    pip3 install -r requirements.txt
    ```
3. Build the ROS packages and source the environment:
    ```shell
   cd ~/catkin_ws/
   rosdep install --from-paths src --ignore-src -r -y
   catkin build
   source devel/setup.bash
    ```
   
## Usage

you can refer to the [examples](https://github.com/ncbdrck/rl_training_validation) to see how to use this package to train robots using ROS and Stable Baselines3.

It also showcases:
- How to use [RealROS](https://github.com/ncbdrck/realros) to create a real-world environment for RL applications.
- Train the Rx200 robot directly in the real world to perform a simple reach task.
- Use [MultiROS](https://github.com/ncbdrck/multiros) framework to create a simulation environment for the same robot and train it in the simulation environment. Then transfer the learned policy to the real-world environment.
- Train both environments (sim and real) in real-time to obtain a generalised policy that performs well in both environments.

The installation instructions for the examples are provided in the respective repositories.

or you can follow the following example steps to train a robot using this package:
```python
#!/bin/python3

# ROS packages required
import rospy

# simulation or real-world environment framework
import uniros as gym
# or 
# import gymnasium as gym

# the custom ROS-based environments (real or sim)
import rl_environments

# Models. New code only needs the single SAC class — it auto-selects
# MultiInputPolicy for Dict observation spaces and enables HER via
# use_her=True (or the YAML config's use_HER key). SAC_GOAL is kept
# as a deprecated alias and emits a DeprecationWarning.
from sb3_ros_support.sac import SAC


if __name__ == '__main__':
   
    # normal environments
    env_base = gym.make('RX200ReacherSim-v0', gazebo_gui=False)

    # or you can use

    # goal-conditioned environments
    env_goal = gym.make('RX200ReacherGoalSim-v0', gazebo_gui=True, ee_action_type=False, 
                        delta_action=False, reward_type="sparse")
   
    # reset the environments
    env_base.reset()
    env_goal.reset()
   
    # create the models. Training configs live in the
    # rl_training_validation package; rl_environments only holds
    # environment-side config (controllers, task definitions).
    pkg_path = "rl_training_validation"
    config_file_name_base = "rx200_reacher_sac.yaml"
    config_file_name_goal = "rx200_reacher_sac_goal.yaml"
    save_path = "/models/sac/"
    log_path = "/logs/sac/"

    # --------------------------------------------------------------------------------------------
    # Creating a model - normal environments (Box observation space → MlpPolicy auto-selected)
    model_base = SAC(env_base, save_path, log_path, model_pkg_path=pkg_path, 
                     config_file_pkg=pkg_path, config_filename=config_file_name_base)
    
    # train the models
    model_base.train()
    model_base.save_model()

    # --------------------------------------------------------------------------------------------
    # Creating a model - goal-conditioned environments
    # (Dict observation space → MultiInputPolicy auto-selected; use_her=True enables HER)
    model_goal = SAC(env_goal, save_path, log_path, model_pkg_path=pkg_path, 
                     config_file_pkg=pkg_path, config_filename=config_file_name_goal,
                     use_her=True)
    
    # train the models
    model_goal.train()
    model_goal.save_model()

    # --------------------------------------------------------------------------------------------
    # validate the models
    obs, _ = env_base.reset()
    episodes = 1000
    epi_count = 0
    while epi_count < episodes:
        action, _states = model_base.predict(observation=obs, deterministic=True)
        obs, _, terminated,truncated, info = env_base.step(action)
        if terminated or truncated:
            epi_count += 1
            rospy.logwarn("Episode: " + str(epi_count))
            obs, _ = env_base.reset()

    env_base.close()
    
    # We can also use the goal-conditioned model to validate
    # Just follow the same procedure as above. Not shown here.
    env_goal.close()
    
    # If you want to load saved models and validate results, you can use the following code.
    # For goal-conditioned envs, pass use_her=True (or rely on the YAML use_HER: true key).
    model = SAC.load_trained_model(save_path + "trained_model_name_without_.zip",
                                   model_pkg=pkg_path,
                                   env=env_goal,
                                   config_file_pkg=pkg_path,
                                   config_filename=config_file_name_goal,
                                   use_her=True)
    # Then you can follow the same validation procedure as above
```
**Note**: Please note that the examples are provided for reference only. You may need to modify the code to suit your specific needs.

## License

This package is released under the [MIT Licence](https://opensource.org/licenses/MIT). Please see the LICENCE file for more details.

## Acknowledgements

We would like to thank the following projects and communities for their valuable contributions, as well as the authors of relevant libraries and tools used in this package.
- [ROS (Robot Operating System)](https://www.ros.org/)
- [FRobs_RL](https://frobs-rl.readthedocs.io/en/latest/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- [OpenAI Gym](https://gym.openai.com/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)


## Contact

For questions, suggestions, or collaborations, contact the project maintainer at [j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie).

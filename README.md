# SB3 ROS Support: The ROS Support Package for Stable Baselines3

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This package is an extension of the [SB3](https://stable-baselines3.readthedocs.io/en/master/) package that provides ROS support for Stable Baselines3. It allows you to train robotics RL agents in the real world and simulations using ROS.

This package extends the functionality of SB3 models in [FRobs_RL](https://github.com/jmfajardod/frobs_rl) package to provide the following features:
 1. Support for goal-conditioned RL tasks
 2. HER (Hindsight Experience Replay) for goal-conditioned RL tasks
 3. Support for training custom environments with [ROS_RL](https://github.com/ncbdrck/ros_rl) or [MultiROS](https://github.com/ncbdrck/multiros) frameworks

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

you can refer to the [examples](https://github.com/ncbdrck/reactorx200_ros_reacher) to see how to use this package to train robots using ROS and Stable Baselines3.

It also showcases:
- How to use [ROS_RL](https://github.com/ncbdrck/ros_rl) to create a real-world environment for RL applications.
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
from multiros.core import multiros_gym as gym
# or 
# import gym

# the custom ROS based environments (real or sim)
import reactorx200_ros_reacher

# Models
from sb3_ros_support.sac import SAC
from sb3_ros_support.sac_goal import SAC_GOAL


if __name__ == '__main__':
   
    # normal environments
    env_base = gym.make('RX200ReacherEnvSim-v0', gazebo_gui=False)
   
    # goal-conditioned environments
    env_goal = gym.make('RX200ReacherGoalEnvSim-v0', gazebo_gui=True, ee_action_type=False, 
                        delta_action=False, reward_type="sparse")
   
    # reset the environments
    env_base.reset()
    env_goal.reset()
   
    # create the models
    pkg_path = "reactorx200_ros_reacher"
    config_file_name_base = "sac.yaml"
    config_file_name_goal = "sac_goal.yaml"
    save_path = "/models/sac/"
    log_path = "/logs/sac/"
    
    # normal environments
    model_base = SAC(env_base, save_path, log_path, model_pkg_path=pkg_path, 
                     config_file_pkg=pkg_path, config_filename=config_file_name_base)
    
    # train the models
    model_base.train()
    model_base.save_model()
    
    # goal-conditioned environments
    model_goal = SAC_GOAL(env_goal, save_path, log_path, model_pkg_path=pkg_path, 
                          config_file_pkg=pkg_path, config_filename=config_file_name_goal)
    
    # train the models
    model_goal.train()
    model_goal.save_model()
    
    # validate the models
    obs = env_base.reset()
    episodes = 1000
    epi_count = 0
    while epi_count < episodes:
        action, _states = model_base.predict(observation=obs, deterministic=True)
        obs, _, dones, info = env_base.step(action)
        if dones:
            epi_count += 1
            rospy.logwarn("Episode: " + str(epi_count))
            obs = env_base.reset()

    env_base.close()
    
    # we can also use the goal-conditioned model to validate the normal environment
    # Just follow the same procedure as above. Not shown here.
    env_goal.close()
    
    # if you want to load saved models and validate results, you can use the following code
    model = SAC.load_trained_model(save_path + "trained_model_name_without_.zip", 
                                   model_pkg= pkg_path,
                                   env=env_goal,
                                   config_filename=config_file_name_goal)
    # then you can follow the same validation procedure as above
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


## Contact

For questions, suggestions, or collaborations, contact the project maintainer at [j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie).

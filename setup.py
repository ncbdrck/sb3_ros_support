#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    name="sb3_ros_support",
    packages=['sb3_ros_support'],
    package_dir={'': 'src'},

    description="The ROS Support Package for Stable Baselines3",
    url="https://github.com/ncbdrck/sb3_ros_support",
    keywords=['ROS', 'reinforcement learning', 'machine-learning', 'gym', 'robotics', 'openai', 'stable-baselines3', 'multiros', 'sb3'],

    author='Jayasekara Kapukotuwa',
    author_email='j.kapukotuwa@research.ait.ie',

    license="MIT",
)

setup(**setup_args)

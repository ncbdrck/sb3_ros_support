#!/bin/python3

import rospkg
import rospy
import os
import yaml
from yaml.loader import SafeLoader


# load the yaml file and return dict that contain all the parameters
def load_yaml(pkg_name=None, file_name=None, file_abs_path=None) -> dict:
    """
    Fetch a YAML file from a package or an abs path, parse and converts to a Python dictionary.

    Args:
        pkg_name (str): name of package. Required if file_abs_path is None.
        file_name (str): name of file. Required if file_abs_path is None.
        file_abs_path (str): Absolute path of the YAML file. Required if pkg_name and file_name are None.

    Returns:
        dict: Dictionary containing the YAML file.
    """

    # If pkg_name and file_name are not None, try to locate the package and check if the YAML file exists within it
    if pkg_name and file_name is not None:
        rospack = rospkg.RosPack()
        try:
            pkg_path = rospack.get_path(pkg_name)
            rospy.logdebug(f"Package {pkg_name} located!.")
        except rospkg.common.ResourceNotFound:
            rospy.logerr(f"Package {pkg_name} not found!.")
            raise rospkg.common.ResourceNotFound(f"Package {pkg_name} not found!.")

        file_abs_path = pkg_path + "/config/" + file_name
        if os.path.exists(pkg_path + "/config/" + file_name) is False:
            print(f"Config file {file_name} in {file_abs_path} does not exist")
            raise FileNotFoundError(f"Config file {file_name} in {file_abs_path} does not exist")

    # If pkg_name and file_name are both None but file_abs_path is not None,
    # check if the YAML file exists at the given absolute path
    elif file_abs_path is not None:
        if os.path.exists(file_abs_path) is False:
            print(f"Config file {file_abs_path} does not exist!")
            raise FileNotFoundError(f"Config file {file_abs_path} does not exist!")

    # If none of these conditions are met, return False
    else:
        print("Load Failed! Requires either the absolute path or the pkg_name and the file_name as input!")
        raise FileNotFoundError("Load Failed! Requires either the absolute path or the pkg_name and the file_name as input!")

    # If the YAML file exists, load it and return the dictionary
    with open(file_abs_path, 'r') as stream:
        try:
            yaml_dict = yaml.load(stream, Loader=SafeLoader)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return dict([])



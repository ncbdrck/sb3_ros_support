"""
Shared stubs for the sb3_ros_support test suite. Stubs are installed
at module load (before any test file imports sb3_ros_support code)
because every algorithm module does ``import rospy`` at module load.
"""
import sys
import types


def _force_stub(name, attrs=None):
    sys.modules[name] = types.SimpleNamespace(**(attrs or {}))


_force_stub("rospy", {
    "loginfo":   lambda *a, **k: None,
    "logwarn":   lambda *a, **k: None,
    "logdebug":  lambda *a, **k: None,
    "logerr":    lambda *a, **k: None,
    "logfatal":  lambda *a, **k: None,
    "ROSException": Exception,
    "on_shutdown": lambda cb: None,
    "is_shutdown": lambda: False,
    "init_node":  lambda *a, **k: None,
    "wait_for_service": lambda *a, **k: None,
    "get_param":  lambda *a, **k: None,
    "has_param":  lambda *a, **k: False,
    "set_param":  lambda *a, **k: None,
})
_force_stub("rosparam", {"upload_params": lambda *a, **k: None})
_force_stub("rospkg", {
    "RosPack": lambda: types.SimpleNamespace(get_path=lambda *a: "/tmp"),
    "common":  types.SimpleNamespace(ResourceNotFound=Exception),
})

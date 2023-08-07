from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    name="sb3_support",
    packages=['sb3_support'],
    package_dir={'': 'src'},

    description="Stable Baselines3 Support for ROS",
    url="https://github.com/ncbdrck/sb3_support",
    keywords=['ROS', 'reinforcement learning', 'machine-learning', 'gym', 'robotics', 'openai', 'stable-baselines3',
              'multiros', 'sb3'],

    author='Jayasekara Kapukotuwa',
    author_email='j.kapukotuwa@research.ait.ie',

    license="MIT",
)

setup(**setup_args)

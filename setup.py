import glob
import os
from setuptools import find_packages
from setuptools import setup

package_name = 'path_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/path_planning/launch/sim', glob.glob(os.path.join('launch', 'sim', '*launch.*'))),
        ('share/path_planning/launch/real', glob.glob(os.path.join('launch', 'real', '*launch.*'))),
        ('share/path_planning/launch/debug', glob.glob(os.path.join('launch', 'debug', '*launch.*'))),
        (os.path.join('share', package_name, 'config', 'sim'), glob.glob('config/sim/*.yaml')),
        (os.path.join('share', package_name, 'config', 'real'), glob.glob('config/real/*.yaml')),
        (os.path.join('share', package_name, 'config', 'debug'), glob.glob('config/debug/*.yaml')),
        ('share/path_planning/example_trajectories', glob.glob(os.path.join('example_trajectories', '*.traj')))],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='sebastianag2002@gmail.com',
    description='Path Planning ROS2 Package',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_builder = path_planning.trajectory_builder:main',
            'trajectory_loader = path_planning.trajectory_loader:main',
            'trajectory_planner = path_planning.trajectory_planner:main',
            'trajectory_follower = path_planning.trajectory_follower:main',
            'trajectory_planner_astar = path_planning.trajectory_planner_astar:main',
            'trajectory_planner_rrt_star = path_planning.trajectory_planner_rrt_star:main',
            'trajectory_follower_mpc = path_planning.trajectory_follower_mpc:main',
            'track_publisher = path_planning.track_publisher:main',
            'lane_follower_pp = path_planning.lane_follower_pp:main',
            'state_machine_b = path_planning.state_machine_b:main',
            'cone_detector_final_challenge_b = visual_servoing.cone_detector_final_challenge_b:main',
            'yolo_annotator_final_challenge_b = visual_servoing.yolo_annotator_final_challenge_b:main',
        ],
    },
)

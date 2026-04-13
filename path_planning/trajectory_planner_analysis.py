import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node

import numpy as np
import heapq
from scipy.ndimage import binary_dilation
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import matplotlib.pyplot as plt

if __name__ == "__main__":
    '''
    Analysis data format: 
    
    RRT*: [elapsed_time, straight_line_distance, actual_path_length, path_efficiency, max_iterations]
    A*: [elapsed_time, straight_line_distance, actual_path_length, path_efficiency]
    '''
    
    rrt_star_data_path1 = [
        [],        
    ]
    
    a_star_data_path1 = [
        [],
    ]    
    
    rrt_star_data_path2 = [
        [],        
    ]
    
    a_star_data_path2 = [
        [],
    ]
    
    rrt_star_data_path3 = [
        [],        
    ]
    
    a_star_data_path3 = [
        [],
    ]
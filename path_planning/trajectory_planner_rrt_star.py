import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node

import numpy as np
import heapq
from scipy.ndimage import binary_dilation
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

class TreeNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            map_qos)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.grid_map = None
        self.map_info = None
        self.current_pose = None
        self.map_dilation_factor = 0.5 #by default, this used to be self.robot_radius which is set to 0.5, so 0.5 is likely our "default setting" assuming robot_radius is still set at 0.5

        self.robot_radius = 0.5

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    # coordinate transforms
    def world_to_grid(self, x, y):
        origin = self.map_info.origin
        res = self.map_info.resolution
        # rotation angle from map origin quaternion
        theta = 2 * np.arctan2(origin.orientation.z, origin.orientation.w)
        dx = x - origin.position.x
        dy = y - origin.position.y
        map_x =  dx * np.cos(-theta) - dy * np.sin(-theta)
        map_y =  dx * np.sin(-theta) + dy * np.cos(-theta)
        col = int(map_x / res)
        row = int(map_y / res)
        return (row, col)

    def grid_to_world(self, row, col):
        origin = self.map_info.origin
        res = self.map_info.resolution
        theta = 2 * np.arctan2(origin.orientation.z, origin.orientation.w)
        map_x = col * res
        map_y = row * res
        x = map_x * np.cos(theta) - map_y * np.sin(theta) + origin.position.x
        y = map_x * np.sin(theta) + map_y * np.cos(theta) + origin.position.y
        return (x, y)

    # call back functions
    def map_cb(self, msg):
        self.get_logger().info("Map received")
        self.map_info = msg.info
        self.grid_map = np.array(msg.data).reshape(msg.info.height, msg.info.width)

    def pose_cb(self, pose):
        self.current_pose = pose.pose.pose

    def goal_cb(self, msg):
        self.get_logger().info("Goal received")
        if self.current_pose is None or self.grid_map is None:
            self.get_logger().warn(f"No pose or map received yet: pose={self.current_pose is not None}, map={self.grid_map is not None}")
            return
        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y
        end_x = msg.pose.position.x
        end_y = msg.pose.position.y

        start_map = self.world_to_grid(start_x, start_y)
        end_map = self.world_to_grid(end_x, end_y)
        self.plan_path(start_map, end_map, self.grid_map)

    # Utility functions
    def bresenham(self, x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2*dy - dx
        y = 0

        for x in range(dx + 1):
            yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy

    def line_of_sight(self, p1, p2, map):
        for r, c in self.bresenham(p1[0], p1[1], p2[0], p2[1]):
            if r < 0 or r >= map.shape[0] or c < 0 or c >= map.shape[1]:
                return False
            if map[r, c] != 0:
                return False
        return True

    def parse_path(self, path, map):

        if len(path) < 3:
            return path

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.line_of_sight(path[i], path[j], map):
                    break
                j -= 1
            simplified.append(path[j])
            i = j

        return simplified

    # RRT* functions
    def rrt_star(self, start_point, end_point, map, max_iterations=1000, step_size=5.0, search_radius=10.0):
        '''
        Adding in functionality to time how long astar takes as well as calculating the difference between the absolute distance between the start and goal points as well as the actual path length.
        
        Time: 
        1. Start timer at beginning of rrt_star function
        2. End timer right before return statement
        3. Calculate elapsed time and log it
        
        Path length:
        1. Calculate straight line distance between start and end points using Euclidean distance
        2. Calculate actual path length by summing distances between consecutive nodes in the path
        3. Log both the straight line distance and the actual path length, as well as the ratio of actual path length to straight line distance (path efficiency)
        ''' 
        time_start = self.get_clock().now()  # Start timer
        
        start_node = TreeNode(start_point[0], start_point[1])
        start_node.cost = 0.0

        tree = [start_node]
        best_goal_node = None
        best_cost = np.inf

        for iteration in range(max_iterations):
            # Goal biasing: 10% chance to sample goal directly
            if np.random.rand() < 0.1:
                rand_point = end_point
            else:
                rand_point = (np.random.randint(0, map.shape[0]), np.random.randint(0, map.shape[1]))

            # Find nearest node in tree
            nearest_node = min(tree, key=lambda node: (node.x - rand_point[0])**2 + (node.y - rand_point[1])**2)

            # Steer from nearest toward random point
            new_node_pos = self.steer(nearest_node, rand_point, step_size)

            # Check collision
            if not self.line_of_sight(tuple([nearest_node.x, nearest_node.y]), new_node_pos, map):
                continue

            # Create new node
            new_node = TreeNode(new_node_pos[0], new_node_pos[1])
            dist_to_nearest = np.sqrt((new_node.x - nearest_node.x)**2 + (new_node.y - nearest_node.y)**2)
            new_node.cost = nearest_node.cost + dist_to_nearest
            new_node.parent = nearest_node

            # Find nearby nodes for rewiring (RRT* optimization step)
            nearby_nodes = []
            for node in tree:
                dist = np.sqrt((node.x - new_node.x)**2 + (node.y - new_node.y)**2)
                if dist < search_radius:
                    nearby_nodes.append(node)

            # Find best parent among nearby nodes
            best_parent = nearest_node
            best_cost_through_parent = new_node.cost

            for nearby_node in nearby_nodes:
                if self.line_of_sight(tuple([nearby_node.x, nearby_node.y]), new_node_pos, map):
                    dist_through_nearby = np.sqrt((nearby_node.x - new_node.x)**2 + (nearby_node.y - new_node.y)**2)
                    cost_through_nearby = nearby_node.cost + dist_through_nearby

                    if cost_through_nearby < best_cost_through_parent:
                        best_parent = nearby_node
                        best_cost_through_parent = cost_through_nearby

            new_node.parent = best_parent
            new_node.cost = best_cost_through_parent
            tree.append(new_node)

            # Rewire nearby nodes through new node
            for nearby_node in nearby_nodes:
                if self.line_of_sight(tuple([new_node.x, new_node.y]), tuple([nearby_node.x, nearby_node.y]), map):
                    dist_through_new = np.sqrt((new_node.x - nearby_node.x)**2 + (new_node.y - nearby_node.y)**2)
                    cost_through_new = new_node.cost + dist_through_new

                    if cost_through_new < nearby_node.cost:
                        nearby_node.parent = new_node
                        nearby_node.cost = cost_through_new

            # Check if new node can reach goal
            if self.line_of_sight(new_node_pos, end_point, map):
                dist_to_goal = np.sqrt((new_node.x - end_point[0])**2 + (new_node.y - end_point[1])**2)
                cost_to_goal = new_node.cost + dist_to_goal

                if cost_to_goal < best_cost:
                    best_cost = cost_to_goal
                    best_goal_node = new_node

        if best_goal_node is None:
            self.get_logger().warn("RRT* failed to find a path")
            return None

        path = self.reconstruct_path(best_goal_node)
        path.append(end_point)
        
        ##########################################################################################
        # For analysis: calculate straight line distance, actual path length, and path efficiency
        ##########################################################################################
        straight_line_distance = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
        actual_path_length = sum(np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2) for i in range(1, len(path)))
        path_efficiency = actual_path_length / straight_line_distance if straight_line_distance > 0 else float('inf')   
        time_end = self.get_clock().now()  # End timer
        elapsed_time = (time_end - time_start).nanoseconds / 1e9  # Calculate elapsed time in seconds
        self.get_logger().info(f"RRT* elapsed time: {elapsed_time:.4f} seconds")
        self.get_logger().info(f"RRT* straight line distance: {straight_line_distance:.4f}")
        self.get_logger().info(f"RRT* actual path length: {actual_path_length:.4f}")
        self.get_logger().info(f"RRT* path efficiency: {path_efficiency:.4f}")
        self.get_logger().info(f"RRT* number of iterations: {max_iterations}")
        
        return path

    def steer(self, from_node, to_point, step_size=1.0):
        """Steer from a node toward a point, limited by step_size."""
        direction = np.array(to_point) - np.array([from_node.x, from_node.y])
        length = np.linalg.norm(direction)

        if length < step_size:
            return tuple(to_point)
        else:
            direction = direction / length
            new_point = np.array([from_node.x, from_node.y]) + direction * step_size
            return (int(new_point[0]), int(new_point[1]))

    def reconstruct_path(self, end_node):
        """Reconstruct path by tracing back through parent pointers."""
        path = []
        current = end_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

    # Plan Path function
    def plan_path(self, start_point, end_point, map):
        self.trajectory.clear()

        #inflate map
        # r = int(self.robot_radius/self.map_info.resolution)
        r = int(self.map_dilation_factor / self.map_info.resolution)
        y,x = np.ogrid[-r:r+1, -r:r+1]
        kernel = x**2 + y**2 <= r**2

        #mask: treat occupied (>0) and unknown (-1) as obstacles
        obstacles = map != 0

        inflated = binary_dilation(obstacles, structure=kernel)

        inflated_map = inflated.astype(np.uint8)
        inflated_map[start_point[0], start_point[1]] = 0
        inflated_map[end_point[0], end_point[1]] = 0

        result = self.rrt_star(start_point, end_point, inflated_map)
        # self.get_logger().info(f"RRT* result: {result}")

        if result is None:
            return

        for row, col in self.parse_path(result, inflated_map):
            self.trajectory.addPoint(self.grid_to_world(row, col))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

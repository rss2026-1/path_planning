import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node

import numpy as np
import heapq
from scipy.ndimage import binary_dilation
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

class PathPlanAStar(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner_astar")
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
        # self.res = 


        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def world_to_grid(self, x, y):
        origin = self.map_info.origin
        res = self.map_info.resolution
        self.get_logger().info(f'Resolution:{res}')
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


    def a_star(self, start_point, end_point, map):

        def _is_free(row, col):
            if row < 0 or row >= map.shape[0] or col < 0 or col >= map.shape[1]:
                return False
            return map[row, col] == 0  # only free cells (inflated map has 0=free, 1=obstacle)

        def calc_h(row, col):
            return np.sqrt((row - end_point[0])**2 + (col - end_point[1])**2)

        def trace_path(parent_row, parent_col):
            path = []
            row, col = end_point
            while (row, col) != start_point:
                path.append((row, col))
                row, col = parent_row[row, col], parent_col[row, col]
            path.append(start_point)
            path.reverse()

            return path

        
        
        sr, sc = start_point
        er, ec = end_point

        if not _is_free(sr, sc) or not _is_free(er, ec):
            self.get_logger().warn(f"Occupied goal/start: start={map[sr,sc]} end={map[er,ec]}")
            return None

        if start_point == end_point:
            self.get_logger().warn("Already at goal")
            return None

        rows, cols = map.shape
        f          = np.full((rows, cols), np.inf)
        g          = np.full((rows, cols), np.inf)
        parent_row = np.full((rows, cols), -1, dtype=int)
        parent_col = np.full((rows, cols), -1, dtype=int)

        g[sr, sc] = 0.0
        f[sr, sc] = calc_h(sr, sc)
        parent_row[sr, sc] = sr
        parent_col[sr, sc] = sc

        open_list = [(f[sr, sc], sr, sc)]
        closed_set = set()

        directions = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]

        while open_list:
            _, i, j = heapq.heappop(open_list)

            if (i, j) in closed_set:
                continue
            closed_set.add((i, j))

            for dr, dc in directions:
                row, col = i + dr, j + dc

                if (row, col) in closed_set or not _is_free(row, col):
                    continue

                if (row, col) == end_point:
                    parent_row[row, col] = i
                    parent_col[row, col] = j
                    return trace_path(parent_row, parent_col)

                new_g = g[i, j] + np.sqrt(dr**2 + dc**2)
                # new_f = new_g + calc_h(row, col)
                epsilon = 1.0 # heuristic weight, can be tuned- mainly for analyis
                new_f = new_g + epsilon * calc_h(row, col)


                if new_f < f[row, col]:
                    f[row, col] = new_f
                    g[row, col] = new_g
                    parent_row[row, col] = i
                    parent_col[row, col] = j
                    heapq.heappush(open_list, (new_f, row, col))

        self.get_logger().warn("No path found")
        return None

    def _is_free(self, row, col):
            if self.grid_map[row, col] == 0:
                return True
            return False

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

        result = self.a_star(start_point, end_point, inflated_map)
        # self.get_logger().info(f"A* result: {result}")


        if result is None:
            return

        for row, col in self.parse_path(result, inflated_map):
            self.trajectory.addPoint(self.grid_to_world(row, col))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()



def main(args=None):
    rclpy.init(args=args)
    planner = PathPlanAStar()
    rclpy.spin(planner)
    rclpy.shutdown()

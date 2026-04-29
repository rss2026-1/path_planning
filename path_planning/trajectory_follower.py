"""
Testing pure pursuit

- ros2 launch racecar_simulator simulate.launch.xml
- ros2 launch path_planning sim_plan_follow.launch.xml
- ros2 launch path_planning load_trajectory.launch.xml
"""

import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64
from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.5    # meters
        self.speed = 1.0    # m/s
        self.wheelbase_length = 0.34   # meters
        self.goal_threshold = 0.5

        # Take the car to the start line
        self.start_threshold = 0.5
        self.reached_start = False

        self.error_pub = self.create_publisher(Float64, "/cross_track_error", 1)

        self.initialized_traj = False
        self.stopped = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.lookahead_pub = self.create_publisher(Marker,
                                                   "/lookahead_point",
                                                   1)

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj or self.stopped:
            return

        points = np.array(self.trajectory.points)
        if len(points) < 2:
            return

        # Extract car pose
        pos = odometry_msg.pose.pose.position
        car_x, car_y = pos.x, pos.y
        car_pos = np.array([car_x, car_y])

        q = odometry_msg.pose.pose.orientation
        # yaw from quaterion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        car_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Check if we reached goal first
        goal = points[-1]
        if np.hypot(car_x - goal[0], car_y - goal[1]) <= self.goal_threshold:
            self.publish_drive(0.0, 0.0)
            self.stopped = True
            self.get_logger().info("Stopping since we reached goal")
            return

        # Drive to the start
        start = points[0]
        dist_to_start = np.hypot(car_x - start[0], car_y - start[1])
        if not self.reached_start and dist_to_start > self.start_threshold:
            dx = start[0] - car_x
            dy = start[1] - car_y
            local_y = -np.sin(car_yaw) * dx + np.cos(car_yaw) * dy
            local_x = np.cos(car_yaw) * dx + np.sin(car_yaw) * dy

            lookahead_dist = max(dist_to_start, self.lookahead)
            curvature = 2.0 * local_y / (lookahead_dist ** 2)
            steering_angle = np.arctan(curvature * self.wheelbase_length)

            if local_x < 0:
                self.publish_drive(-self.speed, -steering_angle)
            else:
                self.publish_drive(self.speed, steering_angle)

            self.get_logger().info("Driving to the start")
            return

        if not self.reached_start:
            self.reached_start = True
            self.get_logger().info("Reached the start")


        if self.reached_start:
            error = self.compute_across_track_error(car_pos, points)
            error_msg = Float64()
            error_msg.data = error
            self.error_pub.publish(error_msg)

        # Car's forward unit vector
        forward = np.array([np.cos(car_yaw), np.sin(car_yaw)])

        # Find nearest segment - prefer segments whose closest point is in front
        starts = points[:-1]
        ends = points[1:]
        segs = ends - starts

        seg_len_sq = np.sum(segs ** 2, axis=1)
        seg_len_sq = np.maximum(seg_len_sq, 1e-10) # avoiding division by zero

        t = np.sum((car_pos - starts) * segs, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        closest_points = starts + t[:, None] * segs
        dists = np.linalg.norm(closest_points - car_pos, axis=1)

        # Penalize segments whose closest point is behind the car so we never
        # lock onto a segment we have already passed.
        forward_proj = ((closest_points - car_pos) * forward).sum(axis=1)
        penalty = np.where(forward_proj < 0, 1e6, 0.0)
        nearest_seg_idx = int(np.argmin(dists + penalty))

        # Find lookahead point - intersection must lie in the front semicircle
        lookahead_point = None
        for i in range(nearest_seg_idx, len(segs)):
            p1 = points[i]
            p2 = points[i+1]

            d = p2 - p1
            f = p1 - car_pos

            a = np.dot(d, d)
            b = 2.0 * np.dot(f, d)
            c = np.dot(f, f) - self.lookahead ** 2

            discriminant = b ** 2 - 4 * a * c

            if discriminant < 0:
                continue # no intersection with this segment

            sqrt_disc = np.sqrt(discriminant)
            t2 = (-b + sqrt_disc) / (2 * a)
            t1 = (-b - sqrt_disc) / (2 * a)

            # Prefer the farther intersection (t2) but require it to be in front
            for t_val in [t2, t1]:
                if 0.0 <= t_val <= 1.0:
                    candidate = p1 + t_val * d
                    if np.dot(candidate - car_pos, forward) > 0:
                        lookahead_point = candidate
                        break

            if lookahead_point is not None:
                break

        # Pick the nearest path point that is still in the front semicircle
        if lookahead_point is None:
            forward_dots = ((points - car_pos) * forward).sum(axis=1)
            front_mask = forward_dots > 0
            if front_mask.any():
                front_pts = points[front_mask]
                lookahead_point = front_pts[np.argmin(np.linalg.norm(front_pts - car_pos, axis=1))]
            else:
                self.publish_drive(0.0, 0.0)
                self.stopped = True
                self.get_logger().info("Path end reached")
                return

        # visualize lookahead point
        self.publish_lookahead_marker(lookahead_point)

        # Compute steering angle
        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y

        # local_x = np.cos(car_yaw) * dx + np.sin(car_yaw) * dy
        local_y = -np.sin(car_yaw) * dx + np.cos(car_yaw) * dy

        curvature = 2.0 * local_y / (self.lookahead ** 2)
        steering_angle = np.arctan(curvature * self.wheelbase_length)

        # Publish drive command
        self.publish_drive(self.speed, steering_angle)

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def publish_lookahead_marker(self, point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.2
        marker.color.a= 1.0
        self.lookahead_pub.publish(marker)

    def compute_across_track_error(self, car_pos, points):
        starts = points[:-1]
        ends = points[1:]
        segs = ends - starts

        seg_len_sq = np.sum(segs ** 2, axis = 1)
        seg_len_sq = np.maximum(seg_len_sq, 1e-10)

        t = np.sum((car_pos - starts) * segs, axis = 1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        closest_points = starts + t[:, None] * segs
        dists = np. linalg.norm(closest_points - car_pos, axis = 1)

        return float(np.min(dists))

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True
        self.stopped = False

        self.reached_start = False


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

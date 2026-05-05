"""
Lane following pure pursuit — subscribes to /lane_center_path (nav_msgs/Path)
published by lane_detector in base_link frame.
"""

import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from visualization_msgs.msg import Marker


class LaneFollowerPP(Node):
    """Pure pursuit that follows the lane center path from lane_detector."""

    def __init__(self):
        super().__init__("lane_follower_pp")
        self.declare_parameter('drive_topic', "default")
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.speed = 4.0 # m/s
        self.lookahead = self.speed/2 # metes
        self.lookahead = 2.5


        self.wheelbase_length = 0.34 # meters

        self.local_y_avg = 0.0
        self.local_y_alpha = 0.5    # exponential moving avg
        self.lane_switch_threshold = 0.5    # meters , half lane width
        self.recovering = False
        self.recovery_frames = 0
        self.recovery_max_frames = 30   # give up recovery after ~1 s at 30 Hz

        self.last_mean_y = None
        self.mean_y_jump_threshold = 0.5    # meters. sudden lateral jump means detector switched lanes

        self.last_drive = None  # (speed, steering_angle) of last published command

        self.traj_sub = self.create_subscription(Path,
                                                 "/lane_center_path",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        # self.lookahead_pub = self.create_publisher(Marker, "/lookahead_point", 1)

        self.get_logger().info("Lane Follower Initialized")


    def trajectory_callback(self, path_msg):
        if len(path_msg.poses) < 2:
            return

        # Path is in base_link. car is at origin, forward is +x
        points = np.array([[p.pose.position.x, p.pose.position.y]
                           for p in path_msg.poses])
        car_pos = np.array([0.0, 0.0])
        forward = np.array([1.0, 0.0])


        mean_y = np.mean(points[:, 1])
        if self.last_mean_y is not None and abs(mean_y - self.last_mean_y) > self.mean_y_jump_threshold:
            self.get_logger().warn(
                f"Detected Lane snap rejected: mean_y jumped {mean_y - self.last_mean_y:.3f} m")
            if self.last_drive is not None:
                self.publish_drive(*self.last_drive)
            else:
                self.publish_drive(self.speed, 0.0)
            return
        self.last_mean_y = mean_y

        segs = points[1:] - points[:-1]
        lookahead_point = None

        for i in range(len(segs)):
            p1, p2 = points[i], points[i + 1]
            d = p2 - p1
            f = p1 - car_pos

            a = np.dot(d, d)
            b = 2.0 * np.dot(f, d)
            c = np.dot(f, f) - self.lookahead ** 2
            disc = b ** 2 - 4 * a * c

            if disc < 0:
                continue

            sqrt_disc = np.sqrt(disc)

            # solution to circle-line intersection
            for t_val in [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]:
                if 0.0 <= t_val <= 1.0:
                    candidate = p1 + t_val * d
                    if np.dot(candidate - car_pos, forward) > 0:
                        lookahead_point = candidate
                        break

            if lookahead_point is not None:
                break

        # Fall back to furthest forward point if no circle intersection found
        if lookahead_point is None:
            front_mask = points[:, 0] > 0
            if front_mask.any():
                front_pts = points[front_mask]
                lookahead_point = front_pts[np.argmax(front_pts[:, 0])]
            else:
                self.publish_drive(0.0, 0.0)
                return

        # self.publish_lookahead_marker(lookahead_point)

        # Path is in base_link so local_y is just the y coordinate of the lookahead
        local_y = lookahead_point[1]

        self.local_y_avg = (self.local_y_alpha * local_y
                            + (1.0 - self.local_y_alpha) * self.local_y_avg)

        if abs(self.local_y_avg) > self.lane_switch_threshold:
            self.recovering = True
            self.recovery_frames = 0

        if self.recovering:
            self.recovery_frames += 1
            if abs(self.local_y_avg) < 0.1 or self.recovery_frames > self.recovery_max_frames:
                self.recovering = False
                self.recovery_frames = 0
            else:
                self.get_logger().warn(
                    f"Recovering from lane switch: local_y_avg={self.local_y_avg:.3f} m "
                    f"(frame {self.recovery_frames}/{self.recovery_max_frames})")
                recovery_angle = np.clip(-np.sign(self.local_y_avg) * 0.5, -1.0, 1.0)
                self.publish_drive(self.speed, recovery_angle)
                return

        curvature = 2.0 * local_y / (self.lookahead ** 2)
        steering_angle = np.arctan(curvature * self.wheelbase_length)

        steering_angle = np.clip(steering_angle, -1.0, 1.0)

        self.publish_drive(self.speed, steering_angle)

    def publish_drive(self, speed, steering_angle, save=True):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)
        if save:
            self.last_drive = (float(speed), float(steering_angle))

    def publish_lookahead_marker(self, point):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "base_link"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        self.lookahead_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    follower = LaneFollowerPP()
    rclpy.spin(follower)
    rclpy.shutdown()

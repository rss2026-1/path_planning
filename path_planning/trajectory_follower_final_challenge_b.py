"""
Topic wiring:
/person_detected -> yolo_annotator_final_challenge_b
/red_light_detected -> cone_detector_final_challenge_b
/parking_object_px -> yolo_annotator_final_challenge_b
/relative_cone -> homography_transformer
/relative_cone_px -> homography_transformer  (we publish during parking)
/goal_pose -> trajectory_planner_astar
"""

import rclpy
import numpy as np
from enum import Enum

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation, ConeLocationPixel

from .utils import LineTrajectory


class State(Enum):
    FOLLOW_TO_PARK1 = 0 # pure pursuit toward parking-lot-1 A* goal
    STOPPED = 1 # person or red light detected; resume on clear
    PARKING_1 = 2   # drive toward detected parking object (chair) and park
    WAIT_AT_PARK1 = 3   # stay parked for wait_duration seconds
    BACK_UP_FROM_PARK1 = 4  # reverse to clear the parked object
    FOLLOW_TO_PARK2 = 5 # pure pursuit toward parking-lot-2 A* goal
    PARKING_2 = 6   # same parking logic for lot 2
    WAIT_AT_PARK2 = 7   # stay parked for wait_duration seconds
    BACK_UP_FROM_PARK2 = 8  # reverse to clear
    FOLLOW_TO_END = 9   # pure pursuit to the final goal
    DONE = 10   # stop forever


class PurePursuit(Node):
    """Pure pursuit path follower with an integrated state machine for Final Challenge B."""

    def __init__(self):
        super().__init__("trajectory_follower")

        # ROS parameters
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")
        # Map-frame goals for each A* segment (should be set in launch file)
        self.declare_parameter('park1_goal_x', 0.0)
        self.declare_parameter('park1_goal_y', 0.0)
        self.declare_parameter('park2_goal_x', 0.0)
        self.declare_parameter('park2_goal_y', 0.0)
        self.declare_parameter('end_goal_x', 0.0)
        self.declare_parameter('end_goal_y', 0.0)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.park1_goal = (
            self.get_parameter('park1_goal_x').value,
            self.get_parameter('park1_goal_y').value,
        )
        self.park2_goal = (
            self.get_parameter('park2_goal_x').value,
            self.get_parameter('park2_goal_y').value,
        )
        self.end_goal = (
            self.get_parameter('end_goal_x').value,
            self.get_parameter('end_goal_y').value,
        )

        # Pure pursuit parameters
        self.lookahead = 1.0    # m
        self.speed = 1.0    # m/s
        self.wheelbase_length = 0.34    # m
        self.goal_threshold = 1.0   # m — how close counts as "reached goal"
        self.start_threshold = 1.0  # m — how close counts as "at trajectory start"

        self.reached_start = False
        self.initialized_traj = False
        self.stopped = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        # Parking parameters
        self.parking_distance = 0.4 # m — desired stop distance from object
        self.parking_tolerance = 0.05   # m — within this of parking_distance -> parked
        self.max_steering = 0.34    # rad
        self.R_min = self.wheelbase_length / np.tan(self.max_steering)
        self.buffer = 0.3
        self.kturn_active = False
        # Switch from path following to parking when object is within this range
        self.parking_trigger_dist = 2.0 # m

        # Timing
        self.wait_duration = 5.0    # seconds to sit parked
        self.backup_duration = 2.0  # seconds to reverse after parking
        self.backup_speed = -0.5    # m/s (negative = reverse)
        self.wait_start_time = None
        self.backup_start_time = None

        # State machine
        self.state = State.FOLLOW_TO_PARK1
        self.prev_state = None  # state to return to after STOPPED

        # Detection state
        self.person_detected = False
        self.red_light_detected = False
        self.parking_object_px = None   # latest ConeLocationPixel from YOLO
        self.parking_object_time = None # ROS time of last parking_object_px
        self.relative_cone_x = 0.0  # car-frame cone position (from homography)
        self.relative_cone_y = 0.0

        # Subscribers
        self.pose_sub = self.create_subscription(
            Odometry, self.odom_topic, self.pose_callback, 1)
        self.traj_sub = self.create_subscription(
            PoseArray, "/trajectory/current", self.trajectory_callback, 1)
        self.person_sub = self.create_subscription(
            Bool, "/person_detected", self.person_callback, 10)
        self.red_light_sub = self.create_subscription(
            Bool, "/red_light_detected", self.red_light_callback, 10)
        self.parking_px_sub = self.create_subscription(
            ConeLocationPixel, "/parking_object_px", self.parking_px_callback, 10)
        self.cone_sub = self.create_subscription(
            ConeLocation, "/relative_cone", self.cone_callback, 10)

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.lookahead_pub = self.create_publisher(Marker, "/lookahead_point", 1)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        # Relay parking object pixels to homography transformer
        self.cone_px_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)

        # State machine runs at 10 Hz independently of odometry callbacks
        self.create_timer(0.1, self.state_machine_tick)

        self.get_logger().info("Final Challenge B: State Machine + Pure Pursuit initialized")
        self.get_logger().info(f"Park1 goal: {self.park1_goal}, Park2 goal: {self.park2_goal}, End: {self.end_goal}")

    # Callbacks

    def person_callback(self, msg: Bool):
        self.person_detected = msg.data

    def red_light_callback(self, msg: Bool):
        self.red_light_detected = msg.data

    def parking_px_callback(self, msg: ConeLocationPixel):
        self.parking_object_px = msg
        self.parking_object_time = self.get_clock().now()

    def cone_callback(self, msg: ConeLocation):
        self.relative_cone_x = msg.x_pos
        self.relative_cone_y = msg.y_pos

    def trajectory_callback(self, msg: PoseArray):
        self.get_logger().info(f"New trajectory received: {len(msg.poses)} points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.initialized_traj = True
        self.stopped = False
        self.reached_start = False

    # State machine

    def state_machine_tick(self):
        now = self.get_clock().now()

        if self.state == State.DONE:
            return

        # Is the parking object still freshly detected? (stale after 0.5 s)
        parking_visible = (
            self.parking_object_px is not None
            and self.parking_object_time is not None
            and (now - self.parking_object_time).nanoseconds / 1e9 < 0.5
        )

        follow_states = (State.FOLLOW_TO_PARK1, State.FOLLOW_TO_PARK2, State.FOLLOW_TO_END)

        # STOP for pedestrian / red light
        if self.state in follow_states and (self.person_detected or self.red_light_detected):
            self.prev_state = self.state
            self.state = State.STOPPED
            self.publish_drive(0.0, 0.0)
            reason = "person" if self.person_detected else "red light"
            self.get_logger().info(f"STOPPED: {reason} detected")
            return

        if self.state == State.STOPPED:
            if not self.person_detected and not self.red_light_detected:
                self.get_logger().info(f"Resuming {self.prev_state.name}")
                self.state = self.prev_state
                self.prev_state = None
            else:
                self.publish_drive(0.0, 0.0)
            return

        # Trigger parking when object is close enough
        if self.state in (State.FOLLOW_TO_PARK1, State.FOLLOW_TO_PARK2) and parking_visible:
            dist = np.hypot(self.relative_cone_x, self.relative_cone_y)
            if 0.01 < dist < self.parking_trigger_dist:
                next_state = State.PARKING_1 if self.state == State.FOLLOW_TO_PARK1 else State.PARKING_2
                self.state = next_state
                self.kturn_active = False
                self.get_logger().info(f"Parking object at {dist:.2f} m → {next_state.name}")
                return

        # PARKING
        if self.state in (State.PARKING_1, State.PARKING_2):
            # Feed the parking object pixel into the homography pipeline
            if parking_visible:
                self.cone_px_pub.publish(self.parking_object_px)

            speed, steer = self.compute_parking_drive(self.relative_cone_x, self.relative_cone_y)
            self.publish_drive(speed, steer)

            if speed == 0.0 and steer == 0.0:
                wait_state = State.WAIT_AT_PARK1 if self.state == State.PARKING_1 else State.WAIT_AT_PARK2
                self.state = wait_state
                self.wait_start_time = now
                self.get_logger().info(f"Parked → {wait_state.name} ({self.wait_duration:.0f} s)")
            return

        # WAIT at parking spot
        if self.state in (State.WAIT_AT_PARK1, State.WAIT_AT_PARK2):
            self.publish_drive(0.0, 0.0)
            elapsed = (now - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= self.wait_duration:
                backup_state = (
                    State.BACK_UP_FROM_PARK1 if self.state == State.WAIT_AT_PARK1
                    else State.BACK_UP_FROM_PARK2
                )
                self.state = backup_state
                self.backup_start_time = now
                self.get_logger().info(f"Wait done → {backup_state.name}")
            return

        # BACK UP from parking spot
        if self.state in (State.BACK_UP_FROM_PARK1, State.BACK_UP_FROM_PARK2):
            self.publish_drive(self.backup_speed, 0.0)
            elapsed = (now - self.backup_start_time).nanoseconds / 1e9
            if elapsed >= self.backup_duration:
                if self.state == State.BACK_UP_FROM_PARK1:
                    self.state = State.FOLLOW_TO_PARK2
                    self.publish_goal(*self.park2_goal)
                else:
                    self.state = State.FOLLOW_TO_END
                    self.publish_goal(*self.end_goal)
                # Reset pure pursuit for the new trajectory
                self.stopped = False
                self.initialized_traj = False
                self.reached_start = False
                self.parking_object_px = None
                self.parking_object_time = None
                self.get_logger().info(f"Back-up done → {self.state.name}")
            return

        # FOLLOW_TO_PARK1 / FOLLOW_TO_PARK2 / FOLLOW_TO_END:
        # pure pursuit runs in pose_callback; DONE transition happens there too

    # Pure pursuit

    def pose_callback(self, odometry_msg: Odometry):
        follow_states = (State.FOLLOW_TO_PARK1, State.FOLLOW_TO_PARK2, State.FOLLOW_TO_END)
        if self.state not in follow_states:
            return
        if not self.initialized_traj or self.stopped:
            return

        points = np.array(self.trajectory.points)
        if len(points) < 2:
            return

        pos = odometry_msg.pose.pose.position
        car_x, car_y = pos.x, pos.y
        car_pos = np.array([car_x, car_y])

        q = odometry_msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        car_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Check if we reached the A* goal
        goal = points[-1]
        if np.hypot(car_x - goal[0], car_y - goal[1]) <= self.goal_threshold:
            self.publish_drive(0.0, 0.0)
            self.stopped = True
            if self.state == State.FOLLOW_TO_END:
                self.state = State.DONE
                self.get_logger().info("DONE: reached end goal")
            else:
                # Reached parking area — wait for YOLO to detect the object
                self.get_logger().info(
                    f"Reached {self.state.name} goal, waiting for parking object detection")
            return

        # Drive to trajectory start if we are far from it
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
            return

        if not self.reached_start:
            self.reached_start = True

        # Find nearest trajectory segment
        starts = points[:-1]
        ends = points[1:]
        segs = ends - starts
        seg_len_sq = np.maximum(np.sum(segs ** 2, axis=1), 1e-10)
        t = np.sum((car_pos - starts) * segs, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest_points = starts + t[:, None] * segs
        dists = np.linalg.norm(closest_points - car_pos, axis=1)
        nearest_seg_idx = int(np.argmin(dists))

        # Find lookahead point via circle-segment intersection
        lookahead_point = None
        for i in range(nearest_seg_idx, len(segs)):
            p1 = points[i]
            p2 = points[i + 1]
            d = p2 - p1
            f = p1 - car_pos
            a = np.dot(d, d)
            b = 2.0 * np.dot(f, d)
            c = np.dot(f, f) - self.lookahead ** 2
            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                continue
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            if 0.0 <= t2 <= 1.0:
                lookahead_point = p1 + t2 * d
                break
            if 0.0 <= t1 <= 1.0:
                lookahead_point = p1 + t1 * d
                break

        if lookahead_point is None:
            self.publish_drive(0.0, 0.0)
            self.stopped = True
            if self.state == State.FOLLOW_TO_END:
                self.state = State.DONE
            return

        self.publish_lookahead_marker(lookahead_point)

        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y
        local_y = -np.sin(car_yaw) * dx + np.cos(car_yaw) * dy
        curvature = 2.0 * local_y / (self.lookahead ** 2)
        steering_angle = np.arctan(curvature * self.wheelbase_length)
        self.publish_drive(self.speed, steering_angle)

    # Parking math (ported from parking_controller.py)

    def compute_parking_drive(self, cone_x: float, cone_y: float):
        dist = np.sqrt(cone_x ** 2 + cone_y ** 2)
        if dist < 1e-3:
            return 0.0, 0.0

        distance_error = dist - self.parking_distance
        alpha = np.arctan2(cone_y, cone_x)
        steering = float(np.arctan(2.0 * self.wheelbase_length * np.sin(alpha) / dist))
        steering = float(np.clip(steering, -self.max_steering, self.max_steering))

        if dist < self.R_min + self.buffer and abs(alpha) > np.radians(50):
            self.kturn_active = True
        if self.kturn_active and abs(alpha) < np.radians(20):
            self.kturn_active = False

        if self.kturn_active:
            return -0.8, float(-np.sign(alpha) * self.max_steering)
        if cone_x < 0:  # cone is behind us
            return -0.8, float(-np.sign(alpha) * self.max_steering)
        if abs(distance_error) < self.parking_tolerance:
            return 0.0, 0.0  # parked
        speed = float(np.sign(distance_error) * 0.8)
        if speed < 0:
            steering = -steering
        return speed, steering

    # Helpers

    def publish_drive(self, speed: float, steering_angle: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def publish_goal(self, x: float, y: float):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.get_logger().info(f"Published A* goal ({x:.2f}, {y:.2f})")

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
        marker.color.a = 1.0
        self.lookahead_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

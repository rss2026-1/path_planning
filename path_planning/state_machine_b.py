"""
INSTRUCTIONS FOR LAUNCHING FINAL CHALLENGE PART B

THESE HAVE TO BE LAUNCHED IN ORDER (i.e. wait until launching the next one)

Rviz2
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed
ros2 launch path_planning combined.launch.xml
ros2 launch racecar_simulator localization_simulate.launch.xml
    - make sure to source this in ~/sim_ws with source install/setup.bash

Topics to add
- /clicked_point PointStamped
- /cone_debug_img Camera
- vesc/odom/ Odometry
- scan/ LaserScan
- /map	Map
- /pf/pose/odom	Odometry
- /trajectory/current	PoseArray
- /followed_trajectory	--NOT AVAILABLE-PoseArray MAYBE PATH?
- /lookahead_point	Marker
- /cone_marker	Marker
- /yolo/annotated_image	Image

STATE MACHINE

Coordinates between:
  - trajectory_follower.py (pure-pursuit, enabled/disabled via /trajectory_follower/enabled)
  - trajectory_planner_astar.py (A* planner, receives goals via /goal_pose)
  - yolo_annotator_final_challenge_b.py (publishes /parking_object_px, /person_detected)
  - cone_detector_final_challenge_b.py (publishes /red_light_detected)
  - homography_transformer (receives /relative_cone_px, publishes /relative_cone)
  - parking_controller (receives /relative_cone, publishes drive during PARKING states)

Topic wiring:
  Subscribes:
    odom_topic (Odometry): car pose (from particle filter)
    /person_detected (Bool)
    /red_light_detected (Bool)
    /parking_object_px (ConeLocationPixel) from YOLO
    /relative_cone (ConeLocation) from homography transformer
    /clicked_point (PointStamped) RViz: first click = park1, second = park2
    /initialpose (PoseStamped) RViz initial pose doubles as end goal

  Publishes:
    /goal_pose (PoseStamped) to A* planner
    drive_topic (AckermannDriveStamped) ONLY during WAIT/BACKUP/STOPPED states
    /relative_cone_px (ConeLocationPixel) relay to homography whenever object visible
    /trajectory_follower/enabled (Bool) gate pure-pursuit drive output

  Delegates:
    FOLLOW states  -> trajectory_follower (pure pursuit)
    PARKING states -> parking_controller (receives /relative_cone from homography)
    WAIT/BACKUP/STOPPED -> state machine publishes drive directly

States:
  FOLLOW_TO_PARK1: trajectory_follower drives; state machine monitors for parking trigger
  STOPPED: blocked by pedestrian or red light; resumes on clear
  PARKING_1: parking_controller drives; state machine monitors distance for "parked"
  WAIT_AT_PARK1: parked; sit still for wait_duration seconds
  BACK_UP_FROM_PARK1: reverse to clear the object
  FOLLOW_TO_PARK2: trajectory_follower drives to second parking area
  PARKING_2: same parking logic
  WAIT_AT_PARK2
  BACK_UP_FROM_PARK2
  FOLLOW_TO_END: trajectory_follower drives back to start
  DONE: done... stop forever
"""

import rclpy
import numpy as np
from enum import Enum

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from vs_msgs.msg import ConeLocation, ConeLocationPixel


class State(Enum):
    FOLLOW_TO_PARK1 = 0
    STOPPED = 1
    PARKING_1 = 2
    WAIT_AT_PARK1 = 3
    BACK_UP_FROM_PARK1 = 4
    FOLLOW_TO_PARK2 = 5
    PARKING_2 = 6
    WAIT_AT_PARK2 = 7
    BACK_UP_FROM_PARK2 = 8
    FOLLOW_TO_END = 9
    DONE = 10

class StateMachineB(Node):
    """State machine orchestrator for Final Challenge B."""

    def __init__(self):
        super().__init__("state_machine_b")

        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")
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

        # Switch from trajectory following to parking when object is within this range
        self.parking_trigger_dist = 2.0 # m

        # These mirror parking_controller's values and are used only to detect "parked"
        self.parking_distance = 0.4 # m (must match parking_controller)
        self.parking_tolerance = 0.05   # m (must match parking_controller)

        # How many consecutive 10 Hz ticks within tolerance before we call it parked
        self.parked_ticks_needed = 3
        self.parked_ticks = 0

        # Timing for WAIT and BACK_UP states
        self.wait_duration = 5.0    # s
        self.backup_duration = 2.0  # s
        self.backup_speed = -0.5    # m/s (negative = reverse)
        self.wait_start_time = None
        self.backup_start_time = None

        # Goal proximity threshold (same as trajectory_follower)
        self.goal_threshold = 1.0   # m

        # State machine
        self.state = State.FOLLOW_TO_PARK1
        self.prev_state = None  # restored after STOPPED

        # Car pose (from odometry)
        self.car_x = 0.0
        self.car_y = 0.0

        # Detection state
        self.person_detected = False
        self.red_light_detected = False
        self.parking_object_px = None   # latest ConeLocationPixel from YOLO
        self.parking_object_time = None
        self.relative_cone_x = 0.0  # car-frame cone position (from homography)
        self.relative_cone_y = 0.0

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 1)
        self.person_sub = self.create_subscription(
            Bool, "/person_detected", self.person_callback, 10)
        self.red_light_sub = self.create_subscription(
            Bool, "/red_light_detected", self.red_light_callback, 10)
        self.parking_px_sub = self.create_subscription(
            ConeLocationPixel, "/parking_object_px", self.parking_px_callback, 10)
        self.cone_sub = self.create_subscription(
            ConeLocation, "/relative_cone", self.cone_callback, 10)
        # self.initial_pose_sub = self.create_subscription(
            # PoseStamped, "/initialpose", self.initial_pose_callback, 10)
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.initial_pose_callback, 10)
        self.clicked_point_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.clicked_point_callback, 10)

        # Publishers
        # drive_pub is used ONLY during WAIT / BACK_UP / STOPPED
        # During FOLLOW states: trajectory_follower drives
        # During PARKING states: parking_controller drives (via /relative_cone)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        # Relay parking object pixels -> homography_transformer -> /relative_cone -> parking_controller
        self.cone_px_pub = self.create_publisher(
            ConeLocationPixel, "/relative_cone_px", 10)
        self.follower_enable_pub = self.create_publisher(
            Bool, "/trajectory_follower/enabled", 1)

        # 10 Hz state machine tick
        self.create_timer(0.1, self.state_machine_tick)

        self.get_logger().info("State Machine B initialized")
        self.get_logger().info(
            f"Park1: {self.park1_goal}, Park2: {self.park2_goal}, End: {self.end_goal}")

    # Callbacks
    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.car_x = pos.x
        self.car_y = pos.y

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

    def initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.end_goal = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.get_logger().info(f"End goal updated to: {self.end_goal}")
    # def initial_pose_callback(self, msg: PoseStamped):
        # self.end_goal = (msg.pose.position.x, msg.pose.position.y)
        # self.get_logger().info(f"End goal updated to: {self.end_goal}")

    def clicked_point_callback(self, msg: PointStamped):
        """First RViz click sets park1, second sets park2."""
        point = (msg.point.x, msg.point.y)
        if self.park1_goal == (0.0, 0.0):
            self.park1_goal = point
            self.get_logger().info(f"Park1 goal set to: {self.park1_goal}")
            self.publish_goal(*self.park1_goal)
        elif self.park2_goal == (0.0, 0.0):
            self.park2_goal = point
            self.get_logger().info(f"Park2 goal set to: {self.park2_goal}")
            # Will publish after backing up from park1

    # State machine
    def state_machine_tick(self):
        now = self.get_clock().now()

        if self.state == State.DONE:
            return

        parking_visible = (
            self.parking_object_px is not None
            and self.parking_object_time is not None
            and (now - self.parking_object_time).nanoseconds / 1e9 < 0.5
        )

        # Always relay parking object pixels -> homography -> /relative_cone
        # parking_controller receives /relative_cone and drives, but trajectory_follower
        # publishes to a higher-priority mux input (/vesc/input/navigation) so it wins
        # during FOLLOW states. During PARKING states, trajectory_follower is disabled
        if parking_visible:
            self.cone_px_pub.publish(self.parking_object_px)

        follow_states = (State.FOLLOW_TO_PARK1, State.FOLLOW_TO_PARK2, State.FOLLOW_TO_END)

        # Enable trajectory_follower only during FOLLOW states
        self._publish_follower_enable(self.state in follow_states)

        # STOPPED: blocked by pedestrian or red light
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

        # Parking trigger (while following toward a parking area)
        # /relative_cone is now always populated (we relay pixels above), so this works
        if self.state in (State.FOLLOW_TO_PARK1, State.FOLLOW_TO_PARK2) and parking_visible:
            dist = np.hypot(self.relative_cone_x, self.relative_cone_y)
            if 0.01 < dist < self.parking_trigger_dist:
                next_state = (
                    State.PARKING_1 if self.state == State.FOLLOW_TO_PARK1
                    else State.PARKING_2
                )
                self.state = next_state
                self.parked_ticks = 0
                self.get_logger().info(
                    f"Parking object at {dist:.2f} m -> {next_state.name}")
                return

        # FOLLOW_TO_END: check if we reached the end goal
        if self.state == State.FOLLOW_TO_END:
            ex, ey = self.end_goal
            if np.hypot(self.car_x - ex, self.car_y - ey) <= self.goal_threshold:
                self.state = State.DONE
                self.publish_drive(0.0, 0.0)
                self.get_logger().info("DONE: returned to start")
            return

        # PARKING: parking_controller drives; we just watch for "parked"
        if self.state in (State.PARKING_1, State.PARKING_2):
            dist = np.hypot(self.relative_cone_x, self.relative_cone_y)
            if abs(dist - self.parking_distance) < self.parking_tolerance:
                self.parked_ticks += 1
            else:
                self.parked_ticks = 0

            if self.parked_ticks >= self.parked_ticks_needed:
                wait_state = (
                    State.WAIT_AT_PARK1 if self.state == State.PARKING_1
                    else State.WAIT_AT_PARK2
                )
                self.state = wait_state
                self.wait_start_time = now
                self.parked_ticks = 0
                self.get_logger().info(
                    f"Parked -> {wait_state.name} ({self.wait_duration:.0f} s)")
            return

        # WAIT: sit still
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
                self.get_logger().info(f"Wait done -> {backup_state.name}")
            return

        # BACK UP: reverse to clear the object
        if self.state in (State.BACK_UP_FROM_PARK1, State.BACK_UP_FROM_PARK2):
            self.publish_drive(self.backup_speed, 0.0)
            elapsed = (now - self.backup_start_time).nanoseconds / 1e9
            if elapsed >= self.backup_duration:
                if self.state == State.BACK_UP_FROM_PARK1:
                    if self.park2_goal != (0.0, 0.0):
                        self.state = State.FOLLOW_TO_PARK2
                        self.publish_goal(*self.park2_goal)
                        self.get_logger().info(f"Back-up done -> FOLLOW_TO_PARK2")
                    else:
                        self.get_logger().warn("Park2 goal not set yet; waiting for second click")
                else:
                    if self.end_goal != (0.0, 0.0):
                        self.state = State.FOLLOW_TO_END
                        self.publish_goal(*self.end_goal)
                        self.get_logger().info(f"Back-up done -> FOLLOW_TO_END")
                    else:
                        self.get_logger().warn("End goal not set yet; set initial pose in RViz")
                self.parking_object_px = None
                self.parking_object_time = None
                self.get_logger().info(f"Back-up done -> {self.state.name}")
            return

        # FOLLOW_TO_PARK1 / FOLLOW_TO_PARK2 / FOLLOW_TO_END with no trigger yet:
        # trajectory_follower is driving; nothing for the state machine to do

    # Helpers
    def _publish_follower_enable(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.follower_enable_pub.publish(msg)

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


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineB()
    rclpy.spin(node)
    rclpy.shutdown()

'''
[INFO] [launch]: All log files can be found below /root/.ros/log/2026-05-02-17-50-29-718918-racecar-desktop-1573
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [wall_follower-1]: process started with pid [1574]
[INFO] [stop_system-2]: process started with pid [1576]
[INFO] [parking_controller-3]: process started with pid [1578]
[INFO] [cone_detector-4]: process started with pid [1580]
[INFO] [homography_transformer-5]: process started with pid [1582]
[INFO] [republish-6]: process started with pid [1584]
[INFO] [trajectory_planner-7]: process started with pid [1586]
[INFO] [trajectory_follower-8]: process started with pid [1588]
[INFO] [state_machine_b-9]: process started with pid [1590]
[INFO] [particle_filter-10]: process started with pid [1592]
[INFO] [yolo_annotator-11]: process started with pid [1594]
[republish-6] [WARN] [1777758630.223401063] [rcl]: Found remap rule 'in:=/cone_debug_img'. This syntax is deprecated. Use '--ros-args --remap in:=/cone_debug_img' instead.
[republish-6] [WARN] [1777758630.223485865] [rcl]: Found remap rule 'out:=/cone_debug_img_compressed'. This syntax is deprecated. Use '--ros-args --remap out:=/cone_debug_img_compressed' instead.
[parking_controller-3] /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/node.py:440: UserWarning: when declaring parameter named 'drive_topic', declaring a parameter only providing its name is deprecated. You have to either:
[parking_controller-3]  - Pass a name and a default value different to "PARAMETER NOT SET" (and optionally a descriptor).
[parking_controller-3]  - Pass a name and a parameter type.
[parking_controller-3]  - Pass a name and a descriptor with `dynamic_typing=True
[parking_controller-3]   warnings.warn(
[parking_controller-3] [INFO] [1777758631.065343353] [parking_controller]: Parking Controller Initialized
[wall_follower-1] [INFO] [1777758631.580504655] [wall_follower]: Starting node
[wall_follower-1] [INFO] [1777758631.622128840] [wall_follower]: NEW VERSION RUNNING - KDF
[wall_follower-1] [INFO] [1777758631.624743424] [wall_follower]: Wall follower node started
[republish-6] [INFO] [1777758631.775197644] [FFMPEGPublisher]: using libav encoder: libx264
[cone_detector-4] [INFO] [1777758631.777802947] [cone_detector_final_challenge_b]: Cone Detector Initialized
[state_machine_b-9] [INFO] [1777758632.334217703] [state_machine_b]: State Machine B initialized
[state_machine_b-9] [INFO] [1777758632.335466850] [state_machine_b]: Park1: (0.0, 0.0), Park2: (0.0, 0.0), End: (0.0, 0.0)
[cone_detector-4] (360, 640)
[particle_filter-10] [INFO] [1777758632.479120169] [particle_filter]: Motion Model Initialized
[particle_filter-10] [INFO] [1777758632.490339225] [particle_filter]: /map
[particle_filter-10] [INFO] [1777758632.492210466] [particle_filter]: 99
[particle_filter-10] [INFO] [1777758632.493005715] [particle_filter]: 500.0
[particle_filter-10] [INFO] [1777758632.493680385] [particle_filter]: 4.71
[particle_filter-10] [INFO] [1777758632.506233294] [particle_filter]: =============+READY+=============
[particle_filter-10] [INFO] [1777758632.508352284] [particle_filter]: /base_link
[cone_detector-4] Traceback (most recent call last):
[cone_detector-4]   File "/root/racecar_ws/install/visual_servoing/lib/visual_servoing/cone_detector", line 33, in <module>
[cone_detector-4]     sys.exit(load_entry_point('visual-servoing', 'console_scripts', 'cone_detector')())
[cone_detector-4]   File "/root/racecar_ws/build/visual_servoing/visual_servoing/cone_detector.py", line 82, in main
[cone_detector-4]     rclpy.spin(cone_detector)
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py", line 229, in spin
[cone_detector-4]     executor.spin_once()
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 773, in spin_once
[cone_detector-4]     self._spin_once_impl(timeout_sec)
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 770, in _spin_once_impl
[cone_detector-4]     raise handler.exception()
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/task.py", line 254, in __call__
[cone_detector-4]     self._handler.send(None)
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 463, in handler
[cone_detector-4]     await call_coroutine(entity, arg)
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 382, in _execute_subscription
[cone_detector-4]     await await_or_execute(sub.callback, msg)
[cone_detector-4]   File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 107, in await_or_execute
[cone_detector-4]     return callback(*args)
[cone_detector-4]   File "/root/racecar_ws/build/visual_servoing/visual_servoing/cone_detector.py", line 65, in image_callback
[cone_detector-4]     ((xl, yl), (xh, yh)) = cd_color_segmentation(input, input)
[cone_detector-4]   File "/root/racecar_ws/build/visual_servoing/visual_servoing/computer_vision/color_segmentation.py", line 58, in cd_color_segmentation
[cone_detector-4]     return color_segmentation(
[cone_detector-4]   File "/root/racecar_ws/build/visual_servoing/visual_servoing/computer_vision/color_segmentation.py", line 72, in color_segmentation
[cone_detector-4]     x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
[cone_detector-4] ValueError: max() arg is an empty sequence
[homography_transformer-5] [INFO] [1777758632.580359331] [homography_transformer]: Homography Transformer Initialized
[ERROR] [cone_detector-4]: process has died [pid 1580, exit code 1, cmd '/root/racecar_ws/install/visual_servoing/lib/visual_servoing/cone_detector --ros-args -r __node:=cone_detector_final_challenge_b'].
[yolo_annotator-11] [INFO] [1777758636.500210289] [yolo_annotator_final_challenge_b]: Running yolo11n.pt on device cuda:0
[yolo_annotator-11] [INFO] [1777758636.501399243] [yolo_annotator_final_challenge_b]: Confidence threshold: 0.5
[yolo_annotator-11] [INFO] [1777758636.502207036] [yolo_annotator_final_challenge_b]: You've chosen to keep these class IDs: [0, 24, 39, 56, 63]
'''
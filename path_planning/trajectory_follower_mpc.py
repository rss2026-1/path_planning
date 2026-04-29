"""
MPC trajectory follower node.

Subscribes
----------
  <odom_topic>          nav_msgs/Odometry       — robot pose
  /trajectory/current   geometry_msgs/PoseArray — planned path

Publishes
---------
  <drive_topic>   ackermann_msgs/AckermannDriveStamped
  /mpc_ref_point  visualization_msgs/Marker   (first reference point)

Launch
------
  ros2 launch path_planning pf_sim_plan_follow_mpc.launch.xml
"""

import time

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)

from .utils import LineTrajectory
from .resources.mpc_formulation import build_mpc


class MPCFollower(Node):

    def __init__(self):
        super().__init__("trajectory_follower_mpc")

        self.declare_parameter('odom_topic',    "default")
        self.declare_parameter('drive_topic',   "default")
        self.declare_parameter('loop_track',    False)
        self.declare_parameter('smooth_window', 7)   # moving-avg window for ref filtering

        odom_topic  = self.get_parameter('odom_topic').get_parameter_value().string_value
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.loop_track     = self.get_parameter('loop_track').get_parameter_value().bool_value
        self._smooth_window = int(self.get_parameter('smooth_window').value)

        # ---- vehicle / MPC parameters --------------------------------
        self.wheelbase      = 0.34    # m
        self.speed          = 4.0     # m/s  (fixed longitudinal speed)
        self.max_steer      = 0.34    # rad  (~20 deg)
        self.goal_threshold = 0.5     # m
        self.N              = 30      # horizon steps
        self.dt             = 0.1     # s per step  → 2 s / 2 m total look-ahead at 1 m/s

        # ---- state ----------------------------------------------------
        self.initialized_traj = False
        self.stopped          = False
        self._mpc_initialized = False
        self._ref_traj        = np.zeros((self.N + 1, 3))   # (x, y, theta) per step
        self.trajectory       = LineTrajectory(self, "/followed_trajectory")
        self._latest_odom     = None   # cache latest pose; MPC timer reads from here

        # ---- build MPC ------------------------------------------------
        self._mpc, self._tvp_template = build_mpc(
            wheelbase = self.wheelbase,
            speed     = self.speed,
            max_steer = self.max_steer,
            N         = self.N,
            dt        = self.dt,
        )
        self.get_logger().info(
            f"MPC ready  (N={self.N}, dt={self.dt}s, v={self.speed}m/s)"
        )

        # ---- ROS2 wiring -----------------------------------------------
        self.pose_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cache_cb, 1)
        # MPC runs on its own timer at 20 Hz, decoupled from odom rate
        self.create_timer(self.dt, self.pose_callback)
        traj_qos = _LATCHED_QOS if self.loop_track else 1
        self.traj_sub = self.create_subscription(
            PoseArray, "/trajectory/current", self.trajectory_callback, traj_qos)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 1)
        self.ref_marker_pub = self.create_publisher(
            Marker, "/mpc_ref_point", 1)

    # ------------------------------------------------------------------
    # Real-time reference window filtering
    # ------------------------------------------------------------------
    def _filter_ref_window(self) -> None:
        """
        Filter self._ref_traj (N+1 × 3) in-place before each MPC solve.

        Two passes on x and y:
          1. Median filter (window 3) — replaces each point with the median
             of itself and its two neighbours, killing isolated spikes.
          2. Moving-average (window smooth_window) — smooths Gaussian noise.

        Heading (theta) is recomputed from the smoothed x/y so it stays
        consistent with the filtered path.
        """
        ref = self._ref_traj
        n   = ref.shape[0]

        for col in range(2):
            arr = ref[:, col]

            # --- median filter (window 3, edge-padded) ---
            left  = np.pad(arr, (1, 0), mode='edge')[:-1]
            right = np.pad(arr, (0, 1), mode='edge')[1:]
            arr   = np.median(np.stack([left, arr, right], axis=1), axis=1)

            # --- moving average ---
            w   = min(self._smooth_window, n)
            pad = w // 2
            ext = np.pad(arr, (pad, pad + 1), mode='edge')
            arr = np.convolve(ext, np.ones(w) / w, mode='valid')[:n]

            ref[:, col] = arr

        # Recompute heading from smoothed x/y
        dx = np.diff(ref[:, 0])
        dy = np.diff(ref[:, 1])
        ref[:-1, 2] = np.arctan2(dy, dx)
        ref[-1,  2] = ref[-2, 2]

    # ------------------------------------------------------------------
    # Reference trajectory sampling
    # ------------------------------------------------------------------
    def _compute_reference(self, car_x: float, car_y: float) -> bool:
        """
        Fill self._ref_traj with N+1 poses sampled along the stored
        trajectory starting from the car's projection onto the nearest
        segment, spaced speed*dt apart in arc length.

        Projecting onto the segment (rather than snapping to the nearest
        waypoint) ensures the reference advances smoothly as the car
        moves, even when waypoints are far apart.
        """
        points  = np.array(self.trajectory.points)
        if len(points) < 2:
            return False

        # For a closed loop, tile so look-ahead wraps around without special-casing
        if self.loop_track:
            points = np.tile(points, (3, 1))

        car_pos = np.array([car_x, car_y])

        # ---- find nearest segment and project car onto it ----
        starts      = points[:-1]                               # (M-1, 2)
        ends        = points[1:]                                # (M-1, 2)
        segs        = ends - starts                             # (M-1, 2)
        seg_len_sq  = np.maximum(np.sum(segs**2, axis=1), 1e-10)

        t_proj      = np.clip(
            np.sum((car_pos - starts) * segs, axis=1) / seg_len_sq,
            0.0, 1.0
        )
        closest_pts = starts + t_proj[:, None] * segs
        nearest_seg = int(np.argmin(np.linalg.norm(closest_pts - car_pos, axis=1)))

        proj_pt = closest_pts[nearest_seg]                      # car's foot on path

        # ---- build remaining path: proj_pt → rest of waypoints ----
        remaining = np.vstack([proj_pt, points[nearest_seg + 1:]])

        if len(remaining) < 2:
            last = points[-1]
            yaw  = np.arctan2(points[-1, 1] - points[-2, 1],
                              points[-1, 0] - points[-2, 0])
            self._ref_traj[:] = [last[0], last[1], yaw]
            return True

        diffs    = np.diff(remaining, axis=0)                   # (K-1, 2)
        seg_lens = np.linalg.norm(diffs, axis=1)                # (K-1,)
        cum_arc  = np.concatenate([[0.0], np.cumsum(seg_lens)]) # (K,)

        spacing = self.speed * self.dt

        for k in range(self.N + 1):
            s = k * spacing
            if s >= cum_arc[-1]:
                pt  = remaining[-1]
                yaw = np.arctan2(diffs[-1, 1], diffs[-1, 0])
            else:
                idx = int(np.searchsorted(cum_arc, s, side='right')) - 1
                idx = int(np.clip(idx, 0, len(diffs) - 1))
                t   = float(np.clip((s - cum_arc[idx]) / max(seg_lens[idx], 1e-9), 0.0, 1.0))
                pt  = remaining[idx] + t * diffs[idx]
                yaw = np.arctan2(diffs[idx, 1], diffs[idx, 0])
            self._ref_traj[k] = [pt[0], pt[1], yaw]

        return True

    def _odom_cache_cb(self, msg: Odometry):
        self._latest_odom = msg

    # ------------------------------------------------------------------
    # MPC control loop — runs at fixed 20 Hz via timer
    # ------------------------------------------------------------------
    def pose_callback(self):
        if not self.initialized_traj or self.stopped:
            return
        if self._latest_odom is None:
            return

        msg = self._latest_odom

        # Extract state
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        car_x   = pos.x
        car_y   = pos.y
        car_yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Goal check (skipped for closed-loop tracks)
        goal = np.array(self.trajectory.points[-1])
        dist_to_goal = np.hypot(car_x - goal[0], car_y - goal[1])
        if not self.loop_track and dist_to_goal < self.goal_threshold:
            self._publish_drive(0.0, 0.0)
            self.stopped = True
            self.get_logger().info("Goal reached — stopping")
            return

        # Build reference for this solve, then filter it
        if not self._compute_reference(car_x, car_y):
            self.get_logger().warn("Reference computation failed")
            return
        self._filter_ref_window()

        # Write reference into the TVP template
        for k in range(self.N + 1):
            self._tvp_template['_tvp', k, 'x_ref']     = float(self._ref_traj[k, 0])
            self._tvp_template['_tvp', k, 'y_ref']     = float(self._ref_traj[k, 1])
            self._tvp_template['_tvp', k, 'theta_ref'] = float(self._ref_traj[k, 2])

        # State column vector expected by do_mpc: shape (n_x, 1)
        x0 = np.array([[car_x], [car_y], [car_yaw]])

        # Warm-start on first call
        if not self._mpc_initialized:
            self._mpc.x0 = x0
            self._mpc.set_initial_guess()
            self._mpc_initialized = True
            self.get_logger().info("MPC warm-start done")

        # Solve
        try:
            t_start = time.perf_counter()
            u0 = self._mpc.make_step(x0)   # returns (n_u, 1) ndarray
            solve_ms = (time.perf_counter() - t_start) * 1e3
        except Exception as e:
            self.get_logger().error(f"MPC solve failed: {e}")
            return

        delta_cmd = float(u0[0, 0])
        self.get_logger().info(
            f"MPC cmd  delta={delta_cmd:.3f} rad  solve={solve_ms:.1f} ms  "
            f"dist_to_goal={dist_to_goal:.2f} m  "
            f"ref=({self._ref_traj[1,0]:.2f},{self._ref_traj[1,1]:.2f})",
            throttle_duration_sec=0.1
        )
        self._publish_drive(self.speed, delta_cmd)
        self._publish_ref_marker(self._ref_traj[1])

    # ------------------------------------------------------------------
    # Trajectory callback
    # ------------------------------------------------------------------
    def trajectory_callback(self, msg: PoseArray):  # noqa: F821
        self.get_logger().info(f"New trajectory: {len(msg.poses)} points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.initialized_traj = True
        self.stopped          = False
        self._mpc_initialized = False   # reset warm-start for new path

    # Publishing helpers
    def _publish_drive(self, speed: float, steering: float):
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.header.frame_id      = "base_link"
        msg.drive.speed          = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def _publish_ref_marker(self, ref_pt):
        m = Marker()
        m.header.frame_id    = "map"
        m.header.stamp       = self.get_clock().now().to_msg()
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = float(ref_pt[0])
        m.pose.position.y    = float(ref_pt[1])
        m.pose.position.z    = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.5
        m.color.a = 1.0
        self.ref_marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = MPCFollower()
    rclpy.spin(node)
    rclpy.shutdown()

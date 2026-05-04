import argparse
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path


def calculate_metrics(actual_points, reference_points):
    """
    Calculates metrics using point-to-segment distance.
    For each actual point, calculates the shortest distance to the line segments 
    that make up the reference path.
    """
    # Create segment start (A) and end (B) points
    A = reference_points[:-1]
    B = reference_points[1:]
    
    # Vector from A to B for all segments
    AB = B - A
    # Squared length of all segments
    AB_squared = np.sum(AB**2, axis=1)
    # Prevent division by zero if there are duplicate adjacent points in the path
    AB_squared[AB_squared == 0] = 1e-10 
    
    distances = []
    
    for p in actual_points:
        # Vector from segment starts to the current point
        AP = p - A
        
        # Project AP onto AB to find the projection scalar 't'
        dot_product = np.sum(AP * AB, axis=1)
        t = dot_product / AB_squared
        
        # Clamp 't' to the [0, 1] range so the closest point lies strictly on the segment
        t = np.clip(t, 0.0, 1.0)
        
        # Calculate the actual closest points on all segments
        closest_points = A + t[:, np.newaxis] * AB
        
        # Calculate Euclidean distances to all closest points and take the minimum
        dist_to_segments = np.linalg.norm(p - closest_points, axis=1)
        distances.append(np.min(dist_to_segments))
        
    distances = np.array(distances)

    mse = np.mean(np.square(distances))
    avg_cte = np.mean(distances)
    max_cte = np.max(distances)

    return mse, avg_cte, max_cte


def extract_xy_from_message(msg):
    """
    Supports:
    - nav_msgs/msg/Path      -> msg.poses[*].pose.position
    - visualization_msgs/msg/Marker -> msg.points[*]
    - Odometry / PoseStamped / Pose
    """
    # nav_msgs/msg/Path
    if hasattr(msg, "poses"):
        pts = []
        for p in msg.poses:
            if hasattr(p, "pose") and hasattr(p.pose, "position"):
                pts.append([p.pose.position.x, p.pose.position.y])
        return np.array(pts)

    # visualization_msgs/msg/Marker
    if hasattr(msg, "points"):
        pts = []
        for p in msg.points:
            pts.append([p.x, p.y])
        return np.array(pts)

    return np.array([])


def extract_position_from_odom(msg):
    if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
        pos = msg.pose.pose.position
    elif hasattr(msg, "pose") and hasattr(msg.pose, "position"):
        pos = msg.pose.position
    elif hasattr(msg, "position"):
        pos = msg.position
    else:
        raise ValueError(f"Unsupported odom message format: {type(msg)}")

    return np.array([pos.x, pos.y])


def process_bag(bag_path, goal_threshold=0.5):
    typestore = get_typestore(Stores.LATEST)

    path_points = None
    raw_odom = []

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        # Print detected topics and message types
        print("Topics in bag:")
        for c in reader.connections:
            print(f"  {c.topic} -> {c.msgtype}")

        # Read reference path / marker
        path_connections = [c for c in reader.connections if c.topic == "/followed_trajectory/path"]
        if not path_connections:
            print("Could not find /followed_trajectory/path")
        else:
            conn = path_connections[0]
            for connection, timestamp, rawdata in reader.messages(connections=[conn]):
                msg = reader.deserialize(rawdata, connection.msgtype)
                path_points = extract_xy_from_message(msg)
                if len(path_points) > 0:
                    break

        # Read odometry
        odom_connections = [c for c in reader.connections if c.topic == "/pf/pose/odom"]
        if not odom_connections:
            print("Could not find /pf/pose/odom")
        else:
            for connection, timestamp, rawdata in reader.messages(connections=odom_connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                raw_odom.append(extract_position_from_odom(msg))

    raw_odom = np.array(raw_odom)

    if path_points is None or len(path_points) == 0:
        print(f"Error: no path points found in /followed_trajectory/path")
        return

    if len(raw_odom) == 0:
        print(f"Error: no odom points found in /pf/pose/odom")
        return

    # Start when car moves more than 10 cm
    start_idx = 0
    start_pos = raw_odom[0]
    for i, p in enumerate(raw_odom):
        if np.linalg.norm(p - start_pos) > 0.1:
            start_idx = i
            break

    # Stop when it reaches the goal
    goal_point = path_points[-1]
    end_idx = len(raw_odom)
    for i in range(start_idx, len(raw_odom)):
        if np.linalg.norm(raw_odom[i] - goal_point) < goal_threshold:
            end_idx = i + 1
            break

    actual_points = raw_odom[start_idx:end_idx]

    if len(actual_points) == 0:
        print("Error: no actual trajectory points after filtering")
        return

    mse, avg_cte, max_cte = calculate_metrics(actual_points, path_points)

    print("\n--- Analysis Results ---")
    print(f"File: {bag_path}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Avg Cross-Track Error:    {avg_cte:.4f} m")
    print(f"Max Cross-Track Error:    {max_cte:.4f} m")

    plt.figure(figsize=(10, 8))
    plt.plot(path_points[:, 0], path_points[:, 1], "r--", label="Target Path", linewidth=2)
    plt.plot(actual_points[:, 0], actual_points[:, 1], "b-", label="Actual Drive", alpha=0.8)

    plt.scatter(actual_points[0, 0], actual_points[0, 1], c="g", label="Movement Start", s=80)
    plt.scatter(actual_points[-1, 0], actual_points[-1, 1], c="k", marker="X", label="Goal Reached", s=80)

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"Trajectory Comparison\n{Path(bag_path).name}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_file", help="Path to the .db3 file")
    parser.add_argument("--goal_dist", type=float, default=0.5)
    args = parser.parse_args()

    process_bag(args.bag_file, goal_threshold=args.goal_dist)
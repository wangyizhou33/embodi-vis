import argparse
import os
import sys
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from mcap.writer import Writer as McapWriter
from ProtobufWriter import ProtobufWriter
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp

dir_path = os.path.dirname(os.path.realpath(__file__))


def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


def tensor_to_jpeg(tensor: np.ndarray) -> Image.Image:
    """Converts a numpy array to a PIL Image."""
    return Image.fromarray(tensor)


def WriteCamera(protobuf_writer, topic, frame_id, image_tensor, ts_ns):
    image = tensor_to_jpeg(image_tensor)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    msg = CompressedImage(
        timestamp=timestamp(ts_ns),
        frame_id=frame_id,
        data=buffered.getvalue(),
        format="jpeg",
    )

    protobuf_writer.write_message(
        topic=topic,
        message=msg,
        log_time=ts_ns,
        publish_time=ts_ns,
    )


def WriteCameraInfo(protobuf_writer, topic, frame_id, width, height, ts_ns):
    calib = CameraCalibration(
        timestamp=timestamp(ts_ns),
        frame_id=frame_id,
        width=width,
        height=height,
        distortion_model="plumb_bob",
    )
    calib.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    # Estimate intrinsics
    # Assuming fx = fy = width is a reasonable approximation for visualization
    # when the actual FOV is unknown but standard.
    fx = float(width)
    fy = float(width)
    cx = width / 2.0
    cy = height / 2.0

    calib.K.extend([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])
    calib.P.extend([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])

    protobuf_writer.write_message(
        topic=topic,
        message=calib,
        log_time=ts_ns,
        publish_time=ts_ns,
    )


def create_camera_transform(parent_frame_id, child_frame_id, extrinsic):
    """Creates a FrameTransform for a camera based on its extrinsic parameters.

    Args:
        frame_id: The name of the camera's frame (e.g., "/sensors/exterior_1_left").
        extrinsic: A 6-element numpy array [x, y, z, roll, pitch, yaw].

    Returns:
        A FrameTransform object.
    """
    x, y, z, roll, pitch, yaw = (
        extrinsic[0],
        extrinsic[1],
        extrinsic[2],
        extrinsic[3],
        extrinsic[4],
        extrinsic[5],
    )
    translation = Vector3(x=float(x), y=float(y), z=float(z))

    # Convert Euler angles to quaternion using scipy
    # The 'zyx' sequence corresponds to yaw, pitch, roll.
    r = Rotation.from_euler("xyz", [roll, pitch, yaw])
    quat = r.as_quat()  # Returns as [x, y, z, w]

    rotation = Quaternion(
        x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
    )

    return FrameTransform(
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
        translation=translation,
        rotation=rotation,
    )


def process_video(protobuf_writer, video_path, timestamp_path, topic, frame_id):
    """Processes a video file and writes its frames to the mcap file."""
    print(f"Processing video {video_path}")
    timestamps_df = pd.read_csv(timestamp_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index < len(timestamps_df):
            # Get the timestamp for the current frame
            # The question states `aligned_stamp` is in seconds, convert to nanoseconds
            ts_ns = int(timestamps_df["aligned_stamp"][frame_index] * 1_000_000_000)

            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]

            WriteCamera(
                protobuf_writer,
                topic,
                frame_id,
                frame_rgb,
                ts_ns,
            )

            # Write Camera Info
            info_topic = topic.replace("/rgb", "/camera_info")
            if info_topic == topic:
                info_topic = topic + "/camera_info"

            WriteCameraInfo(
                protobuf_writer,
                info_topic,
                frame_id,
                width,
                height,
                ts_ns
            )
        else:
            print(f"Warning: More frames in video than timestamps in {timestamp_path}")
            break

        frame_index += 1

    cap.release()
    print(f"Finished processing video {video_path}")


def process_trajectory(protobuf_writer, trajectory_path, child_frame_id):
    """Processes a trajectory file and writes TF messages to the mcap file."""
    print(f"Processing trajectory {trajectory_path}")
    try:
        # Read the space-separated file without a header
        df = pd.read_csv(
            trajectory_path,
            delim_whitespace=True,
            header=None,
            names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )
    except Exception as e:
        print(f"Error reading {trajectory_path}: {e}")
        return

    # Precompute static transform for camera frame (Z forward, X right, Y down)
    # relative to hand frame (X forward, Y left, Z up)
    # Hand X (Fwd) -> Cam Z (Fwd)
    # Hand Y (Left) -> Cam -X (Left) -> Cam X (Right) = -Hand Y
    # Hand Z (Up)   -> Cam -Y (Up)   -> Cam Y (Down)  = -Hand Z
    # Basis of Cam in Hand:
    # X_c = (0, -1, 0)
    # Y_c = (0, 0, -1)
    # Z_c = (1, 0, 0)
    r_cam = Rotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    q_cam = r_cam.as_quat()

    for _, row in df.iterrows():
        ts_sec = row["timestamp"]
        ts_ns = int(ts_sec * 1_000_000_000)

        # Create the transform message
        tfs = FrameTransforms()

        qx, qy, qz, qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
        tx, ty, tz = np.array([float(row["x"]), float(row["y"]), float(row["z"])])

        tf = FrameTransform(
            timestamp=timestamp(ts_ns),
            parent_frame_id="base_link",
            child_frame_id=child_frame_id,
            translation=Vector3(x = tx, y = ty, z = tz),
            rotation=Quaternion(x = qx, y = qy, z = qz, w = qw),
        )
        tfs.transforms.append(tf)

        # Add static camera transform
        tf_cam = FrameTransform(
            timestamp=timestamp(ts_ns),
            parent_frame_id=child_frame_id,
            child_frame_id=child_frame_id + "_cam",
            translation=Vector3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(x=float(q_cam[0]), y=float(q_cam[1]), z=float(q_cam[2]), w=float(q_cam[3])),
        )
        tfs.transforms.append(tf_cam)

        # Write to the /tf topic
        protobuf_writer.write_message(
            topic="/tf",
            message=tfs,
            log_time=ts_ns,
            publish_time=ts_ns,
        )
    print(f"Finished processing trajectory {trajectory_path}")


def get_session_start_time(session_path):
    """Finds the earliest timestamp in the session across all hands and data sources."""
    start_times = []
    for hand in ["left", "right"]:
        hand_folder = ""
        for item in os.listdir(session_path):
            if item.startswith(f"{hand}_hand"):
                hand_folder = os.path.join(session_path, item)
                break
        if not hand_folder:
            continue

        # Check trajectory
        traj_path = os.path.join(hand_folder, "Merged_Trajectory", "merged_trajectory.txt")
        if os.path.exists(traj_path):
            try:
                df = pd.read_csv(traj_path, delim_whitespace=True, header=None, nrows=1)
                if not df.empty:
                    start_times.append(df.iloc[0, 0])
            except:
                pass

        # Check video timestamps
        ts_path = os.path.join(hand_folder, "RGB_Images", "timestamps.csv")
        if os.path.exists(ts_path):
            try:
                df = pd.read_csv(ts_path)
                if not df.empty and "aligned_stamp" in df.columns:
                    start_times.append(df["aligned_stamp"].min())
            except:
                pass

    return min(start_times) if start_times else None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output", type=str, default="fastumi.mcap")
    argparser.add_argument("--task", type=str, default="task1")
    argparser.add_argument("--session", type=str, default="session_1")
    args = argparser.parse_args()

    # Get the path to the session folder
    session_path = os.path.normpath(os.path.join(dir_path, "..", "fastumi_sample", args.task, args.session))

    session_start_time = get_session_start_time(session_path)
    if session_start_time is not None:
        print(f"Session start time: {session_start_time}")
    else:
        print("Warning: Could not determine session start time.")

    with open(args.output, "wb") as f:
        writer = McapWriter(f)
        writer.start()
        protobuf_writer = ProtobufWriter(writer)

        # Process left and right hand videos
        for hand in ["left", "right"]:
            # Find the hand folder
            hand_folder = ""
            for item in os.listdir(session_path):
                if item.startswith(f"{hand}_hand"):
                    hand_folder = os.path.join(session_path, item)
                    break

            if not hand_folder:
                print(f"Could not find {hand}_hand folder in {session_path}")
                continue

            video_path = os.path.join(hand_folder, "RGB_Images", "video.mp4")
            timestamp_path = os.path.join(hand_folder, "RGB_Images", "timestamps.csv")
            trajectory_path = os.path.join(hand_folder, "Merged_Trajectory", "merged_trajectory.txt")

            if os.path.exists(video_path) and os.path.exists(timestamp_path):
                process_video(
                    protobuf_writer,
                    video_path,
                    timestamp_path,
                    topic=f"/fastumi/{hand}_hand/rgb",
                    frame_id=f"{hand}_hand_cam",
                )
            else:
                print(f"video.mp4 or timestamps.csv not found in {os.path.join(hand_folder, 'RGB_Images')}")

            if os.path.exists(trajectory_path):
                process_trajectory(
                    protobuf_writer,
                    trajectory_path,
                    child_frame_id=f"{hand}_hand",
                )
            else:
                print(f"merged_trajectory.txt not found in {os.path.join(hand_folder, 'Merged_Trajectory')}")

        writer.finish()
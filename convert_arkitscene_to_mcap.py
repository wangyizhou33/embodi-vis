from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from mcap.writer import Writer as McapWriter
from ProtobufWriter import ProtobufWriter
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.SceneEntityDeletion_pb2 import SceneEntityDeletion
from foxglove_schemas_protobuf.Color_pb2 import Color
from google.protobuf.timestamp_pb2 import Timestamp
import argparse
import os
from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import io
import json
from time import time_ns

DATA_DIR = "./arkit_scenes/raw/Validation/48458663"
EPISODE_NAME = "48458663"


def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


def load_intrinsics(dir):
    """
    Load the first file in this directory. It's a .pincam file
    The content is like "256 192 212.088 212.088 127.4 98.6992"
    and should be interpreted as
    "width height focal_length_x focal_length_y principal_point_x principal_point_y"
    Output a numpy 3x3 intrinsic matrix
    """
    files = os.listdir(dir)
    if not files:
        raise FileNotFoundError("No files found in the directory.")

    # Assuming the first file is the correct one as per the comment.
    pincam_file = os.path.join(dir, files[0])

    with open(pincam_file, "r") as f:
        content = f.read().strip().split()
        width, height, fx, fy, cx, cy = [float(x) for x in content]

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return int(width), int(height), intrinsic_matrix


def load_camera_poses(file_path):
    """
    Load the camera pose trajectory file.
    Each line is parsed to extract timestamp, translation, and rotation (as quaternion).
    Returns an ordered dictionary mapping timestamps to (translation, quaternion) tuples.
    """
    poses = OrderedDict()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts, t, quat = parse_traj_line(line)
            poses[ts] = (t, quat)
    return poses


def parse_traj_line(line: str):
    tokens = line.strip().split()
    if len(tokens) != 7:
        raise ValueError("Unexpected number of tokens in traj line")
    ts = int(float(tokens[0]) * 1e9)
    rx, ry, rz = map(float, tokens[1:4])
    tx, ty, tz = map(float, tokens[4:7])
    # Rotation vector (axis-angle): magnitude = angle, direction = axis
    rot = R.from_rotvec([rx, ry, rz])  # gives a Rotation object (scipy)
    quat = rot.as_quat()  # (x, y, z, w) quaternion
    # Translation vector:
    t = np.array([tx, ty, tz])
    return ts, t, quat


def load_image_folder(dir_path):
    """
    Loads image file paths from a directory and extracts timestamps from filenames.
    Filenames are expected in the format 'EPISODE_NAME_timestamp.png'.
    Returns a sorted list of (timestamp_ns, file_path) tuples.
    """
    image_files = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".png"):
            parts = filename.split("_")
            if len(parts) > 1:
                try:
                    # The timestamp is the last part before the extension
                    ts_sec_str = parts[-1].replace(".png", "")
                    ts_sec = float(ts_sec_str)
                    ts_ns = int(ts_sec * 1e9)
                    image_files.append((ts_ns, os.path.join(dir_path, filename)))
                except ValueError:
                    # Ignore files that don't match the timestamp pattern
                    continue
    # Sort by timestamp
    image_files.sort()
    return image_files


def load_ply(file_path):
    mesh = trimesh.load(file_path)

    points = np.array(mesh.vertices)
    # trimesh loads colors into mesh.visual.vertex_colors
    # These are usually RGBA with uint8 values.
    colors = np.array(mesh.visual.vertex_colors)

    # The colors are RGBA, let's get RGB and normalize to [0, 1]
    if colors.shape[1] == 4:
        colors = colors[:, :3]

    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0

    structured_data = np.zeros(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("r", "f4"),
            ("g", "f4"),
            ("b", "f4"),
            ("a", "f4"),
        ],
    )
    # Fill the structured array
    structured_data["x"] = points[:, 0]
    structured_data["y"] = points[:, 1]
    structured_data["z"] = points[:, 2]
    structured_data["r"] = colors[:, 0]
    structured_data["g"] = colors[:, 1]
    structured_data["b"] = colors[:, 2]
    structured_data["a"] = 1.0  # Set alpha to 1.0 for all points

    data = io.BytesIO()
    # Write all data at once
    data.write(structured_data.tobytes())
    pcd_msg = PointCloud(
        timestamp=timestamp(time_ns()),
        frame_id="/world",
        point_stride=28,
        data=data.getvalue(),
        fields=[
            PackedElementField(name="x", offset=0, type=PackedElementField.FLOAT32),
            PackedElementField(name="y", offset=4, type=PackedElementField.FLOAT32),
            PackedElementField(name="z", offset=8, type=PackedElementField.FLOAT32),
            PackedElementField(name="red", offset=12, type=PackedElementField.FLOAT32),
            PackedElementField(
                name="green", offset=16, type=PackedElementField.FLOAT32
            ),
            PackedElementField(name="blue", offset=20, type=PackedElementField.FLOAT32),
            PackedElementField(
                name="alpha", offset=24, type=PackedElementField.FLOAT32
            ),
        ],
        pose=Pose(
            position=Vector3(x=0, y=0, z=0),
            orientation=Quaternion(w=1, x=0, y=0, z=0),
        ),
    )
    return pcd_msg


def colorize_depth_image(depth_img_raw):
    """Applies a colormap to a grayscale depth image to improve contrast."""
    depth_array = np.array(depth_img_raw)

    # Normalize array to 0-1 range for colormap
    if np.max(depth_array) > np.min(depth_array):
        normalized_depth = (depth_array - np.min(depth_array)) / (
            np.max(depth_array) - np.min(depth_array)
        )
    else:
        normalized_depth = np.zeros_like(depth_array, dtype=float)

    # Apply colormap
    colored_depth = plt.cm.plasma(normalized_depth)

    # Convert from RGBA (float 0-1) to RGB (uint8 0-255)
    colored_depth_rgb_uint8 = (colored_depth[:, :, :3] * 255).astype(np.uint8)

    # Create a new PIL image from the colored numpy array
    return Image.fromarray(colored_depth_rgb_uint8)


def load_annotation(file_path):
    """
    Loads 3D annotations from a JSON file, parses OBB data, and returns a list of obstacle dictionaries.
    """
    obstacles = []
    with open(file_path, "r") as f:
        annotations = json.load(f)

    for annotation_data in annotations.get("data", []):
        obb_aligned = annotation_data.get("segments", {}).get("obbAligned")
        if not obb_aligned:
            continue

        centroid = obb_aligned.get("centroid")
        axes_lengths = obb_aligned.get("axesLengths")
        normalized_axes = obb_aligned.get("normalizedAxes")
        uid = annotation_data.get("uid", "")
        label = annotation_data.get("label", "unknown")

        if not all([centroid, axes_lengths, normalized_axes]):
            continue

        # The normalizedAxes is a flattened 3x3 rotation matrix.
        # Reshape it to a 3x3 matrix and transpose it, as per the suggestion.
        rotation_matrix = np.array(normalized_axes).reshape(3, 3).T

        # Convert rotation matrix to quaternion (x, y, z, w)
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # returns (x, y, z, w)

        obstacle = {
            "id": uid,
            "label": label,
            "pos": centroid,
            "size": axes_lengths,
            "orientation": quaternion,
        }
        obstacles.append(obstacle)

    return obstacles


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default=DATA_DIR)
    argparser.add_argument("--output", type=str, default="arkitscene.mcap")
    args = argparser.parse_args()

    width, height, intrinsics = load_intrinsics(
        os.path.join(args.data_dir, "lowres_wide_intrinsics")
    )
    cam_poses = load_camera_poses(os.path.join(args.data_dir, "lowres_wide.traj"))
    pcd = load_ply(os.path.join(DATA_DIR, "48458663_3dod_mesh.ply"))
    image_files = load_image_folder(os.path.join(args.data_dir, "lowres_wide"))
    image_timestamps = np.array([t for t, p in image_files])

    # Create CameraCalibration message
    first_ts_ns = next(iter(cam_poses))
    calib_ts_ns = timestamp(first_ts_ns)
    calib_msg = CameraCalibration(
        timestamp=calib_ts_ns,
        frame_id="/sensors/camera",
        width=width,
        height=height,
        distortion_model="plumb_bob",
    )
    calib_msg.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    calib_msg.K.extend(intrinsics.flatten())

    P = np.zeros((3, 4))
    P[:3, :3] = intrinsics
    calib_msg.P.extend(P.flatten())

    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    # write to mcap

    protobuf_writer.write_message(
        "/sensors/camera/intrinsics",
        calib_msg,
        publish_time=first_ts_ns,
        log_time=first_ts_ns,
    )

    protobuf_writer.write_message(
        "/scene/mesh",
        pcd,
        publish_time=first_ts_ns,
        log_time=first_ts_ns,
    )

    # Load and write annotations
    annotation_file = os.path.join(
        args.data_dir, f"{EPISODE_NAME}_3dod_annotation.json"
    )
    if os.path.exists(annotation_file):
        obstacles = load_annotation(annotation_file)

        scene_update = SceneUpdate()
        # Clear any existing entities
        scene_update.deletions.add(type=SceneEntityDeletion.Type.ALL)

        for obstacle in obstacles:
            entity = scene_update.entities.add()
            entity.id = obstacle["id"]
            entity.frame_id = "/world"
            entity.frame_locked = True
            entity.timestamp.CopyFrom(timestamp(first_ts_ns))

            pos = obstacle["pos"]
            size = obstacle["size"]
            quat = obstacle["orientation"]

            entity.cubes.add(
                pose=Pose(
                    position=Vector3(x=pos[0], y=pos[1], z=pos[2]),
                    orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                ),
                size=Vector3(x=size[0], y=size[1], z=size[2]),
                color=Color(r=0, g=0, b=1, a=0.4),
            )

            # Add text label
            text_primitive = entity.texts.add()
            text_primitive.text = obstacle["label"]
            # Position the text slightly above the cube
            text_primitive.pose.position.x = pos[0]
            text_primitive.pose.position.y = pos[1]
            text_primitive.pose.position.z = pos[2] + size[2] / 2.0 + 0.1
            text_primitive.billboard = True
            text_primitive.font_size = 16.0
            text_primitive.scale_invariant = True
            text_primitive.color.r = 1.0
            text_primitive.color.g = 1.0
            text_primitive.color.b = 1.0
            text_primitive.color.a = 1.0

        protobuf_writer.write_message(
            "/scene/annotations",
            scene_update,
            publish_time=first_ts_ns,
            log_time=first_ts_ns,
        )

    # time loop
    for ts_ns, (t, quat) in tqdm(cam_poses.items(), desc="Processing frames"):

        ft_msg = FrameTransform(
            timestamp=timestamp(ts_ns),
            parent_frame_id="/sensors/camera",
            child_frame_id="/world",
            translation=Vector3(x=t[0], y=t[1], z=t[2]),
            rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )

        protobuf_writer.write_message(
            "/tf",
            ft_msg,
            log_time=ts_ns,
            publish_time=ts_ns,
        )

        # Find and write image
        closest_idx = np.abs(image_timestamps - ts_ns).argmin()
        image_ts, image_path = image_files[closest_idx]

        # Only write image if it's close enough to the pose timestamp
        # within half a frame time at 30fps
        if abs(image_ts - ts_ns) < (0.5 / 30 * 1e9):
            img = Image.open(image_path)
            buffered = io.BytesIO()
            # The images are PNG, let's make sure they are RGB before saving as JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buffered, format="jpeg")

            img_msg = CompressedImage(
                timestamp=timestamp(ts_ns),
                frame_id="/sensors/camera",
                data=buffered.getvalue(),
                format="jpeg",
            )
            protobuf_writer.write_message(
                "/image",
                img_msg,
                publish_time=ts_ns,
                log_time=ts_ns,
            )

            # Process depth image
            depth_image_path = image_path.replace("lowres_wide", "lowres_depth")
            if os.path.exists(depth_image_path):
                depth_img_raw = Image.open(depth_image_path)
                depth_img = colorize_depth_image(depth_img_raw)

                buffered_depth = io.BytesIO()
                depth_img.save(buffered_depth, format="png")

                depth_msg = CompressedImage(
                    timestamp=timestamp(ts_ns),
                    frame_id="/sensors/camera",
                    data=buffered_depth.getvalue(),
                    format="png",
                )
                protobuf_writer.write_message(
                    "/depth",
                    depth_msg,
                    publish_time=ts_ns,
                    log_time=ts_ns,
                )

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

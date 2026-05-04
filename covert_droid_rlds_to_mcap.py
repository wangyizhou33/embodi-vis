import argparse
import os
from io import BytesIO

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from urdfpy import URDF

from mcap.writer import Writer as McapWriter
from ProtobufWriter import ProtobufWriter
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp


URDF_FILE = "urdf/droid.urdf"
DATASET_DIR = "droid_100/1.0.0"

CAMERAS = {
    "exterior_image_1_left": {
        "topic": "/camera/exterior_1_left",
        "frame_id": "/sensors/exterior_1_left",
    },
    "exterior_image_2_left": {
        "topic": "/camera/exterior_2_left",
        "frame_id": "/sensors/exterior_2_left",
    },
    "wrist_image_left": {
        "topic": "/camera/wrist_left",
        "frame_id": "/sensors/wrist_left",
    },
}


def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


def rot_matrix_to_quat(rot):
    return Rotation.from_matrix(rot).as_quat()


def tensor_to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def image_from_tensor(value):
    image = tensor_to_numpy(value)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def write_camera(protobuf_writer, topic, frame_id, image, ts_ns):
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


def build_robot_cfg(joint_position, gripper_position):
    joint_position = np.asarray(joint_position, dtype=np.float64).reshape(-1)
    gripper_position = np.asarray(gripper_position, dtype=np.float64).reshape(-1)

    cfg = {}
    for i, value in enumerate(joint_position[:7]):
        cfg[f"panda_joint{i + 1}"] = float(value)

    if gripper_position.size > 0:
        finger = float(np.clip(gripper_position[0], 0.0, 0.725))
        cfg["finger_joint"] = finger
        cfg["left_inner_knuckle_joint"] = finger
        cfg["left_inner_finger_joint"] = -finger
        cfg["right_inner_knuckle_joint"] = -finger
        cfg["right_inner_finger_joint"] = finger
        cfg["right_outer_knuckle_joint"] = -finger

    return cfg


def write_robot_transforms(protobuf_writer, robot, step, ts_ns):
    observation = step["observation"]
    cfg = build_robot_cfg(
        tensor_to_numpy(observation["joint_position"]),
        tensor_to_numpy(observation["gripper_position"]),
    )

    tfs = FrameTransforms()
    tfs.transforms.append(
        FrameTransform(
            parent_frame_id="scene",
            child_frame_id=robot.base_link.name,
            translation=Vector3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    for joint in robot.joints:
        parent_link = joint.parent
        child_link = joint.child
        local_pose = joint.get_child_pose(cfg=cfg.get(joint.name, 0.0))
        trans = local_pose[:3, 3]
        quat = rot_matrix_to_quat(local_pose[:3, :3])

        tfs.transforms.append(
            FrameTransform(
                parent_frame_id=parent_link,
                child_frame_id=child_link,
                translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                rotation=Quaternion(
                    x=float(quat[0]),
                    y=float(quat[1]),
                    z=float(quat[2]),
                    w=float(quat[3]),
                ),
            )
        )

    protobuf_writer.write_message(
        topic="/tf",
        message=tfs,
        log_time=ts_ns,
        publish_time=ts_ns,
    )


def load_dataset(dataset_dir, split):
    if os.path.exists(os.path.join(dataset_dir, "dataset_info.json")):
        builder = tfds.builder_from_directory(dataset_dir)
        return builder.as_dataset(split=split)

    tfds_data_dir = dataset_dir
    if os.path.basename(tfds_data_dir) == "1.0.0":
        tfds_data_dir = os.path.dirname(os.path.dirname(tfds_data_dir))
    return tfds.load("droid_100", data_dir=tfds_data_dir, split=split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--urdf", type=str, default=URDF_FILE)
    parser.add_argument("--output", type=str, default="droid_rlds.mcap")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    ds = load_dataset(args.dataset_dir, args.split)

    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    step_dt_ns = int(1e9 / args.fps)
    ts_ns = 0

    try:
        episodes = ds.take(args.num_episodes)
        for episode_idx, episode in enumerate(tqdm(episodes, total=args.num_episodes)):
            steps = episode["steps"]
            if args.max_steps is not None and args.max_steps > 0:
                steps = steps.take(args.max_steps)

            for step_idx, step in enumerate(steps):
                ts_ns += step_dt_ns
                observation = step["observation"]

                write_robot_transforms(protobuf_writer, robot, step, ts_ns)

                for feature_name, camera in CAMERAS.items():
                    write_camera(
                        protobuf_writer,
                        camera["topic"],
                        camera["frame_id"],
                        image_from_tensor(observation[feature_name]),
                        ts_ns,
                    )

            print(f"Finished episode {episode_idx} at timestamp {ts_ns} ns.")
    finally:
        writer.finish()
        stream.close()

    print(f"The mcap file is saved at {args.output}.")


if __name__ == "__main__":
    main()

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from urdfpy import URDF
from io import BytesIO
import base64
import numpy as np
from scipy.spatial.transform import Rotation
import os
import argparse
from PIL import Image
from tqdm import tqdm
from mcap.writer import Writer as McapWriter
from ProtobufWriter import ProtobufWriter
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from google.protobuf.timestamp_pb2 import Timestamp

dir_path = os.path.dirname(os.path.realpath(__file__))

REPO_ID = "lerobot/droid_1.0.1"
URDF_FILE = "urdf/droid.urdf"


def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


def tensor_to_jpeg(tensor):
    # 1. 预处理：张量→numpy数组（调整形状+数据类型）
    # - 张量形状：[C, H, W] → 转为 [H, W, C]
    # - 数据范围：若为归一化张量（0-1），需乘255转为0-255；若已为0-255则跳过
    img_np = (
        tensor.permute(1, 2, 0).cpu().detach().numpy()
    )  # 调整通道顺序（C→最后一维）
    if img_np.max() <= 1.0:  # 假设输入为归一化张量（如经过ToTensor()）
        img_np = (img_np * 255).astype(np.uint8)  # 转为uint8类型（JPEG要求0-255整数）
    else:
        img_np = img_np.astype(np.uint8)  # 若已为0-255，直接转类型

    # 2. numpy数组→PIL图像
    img_pil = Image.fromarray(img_np)

    # # 3. 保存为JPEG
    # img_pil.save(save_path, "JPEG")  # 可指定质量参数：quality=95（默认75）
    return img_pil


def rot_matrix_to_quat(R):
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


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
        log_time=ts_ns,  # to microseconds
        publish_time=ts_ns,  # to microseconds
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


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--urdf", type=str, default=URDF_FILE)
    argparser.add_argument("--output", type=str, default="droid_101.mcap")
    args = argparser.parse_args()

    # 1) Load from the Hub (cached locally)
    cache_dir = os.path.join(dir_path, REPO_ID)
    dataset = LeRobotDataset(REPO_ID, root=cache_dir, episodes=[0])

    # 2) Random access by index
    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    calib_ts_ns = int(dataset[0]["timestamp"] * 1e9)
    # Camera calibration for exterior_1_left
    calib = CameraCalibration(
        timestamp=timestamp(calib_ts_ns),
        frame_id="/sensors/exterior_1_left",
        width=640,
        height=360,
        distortion_model="plumb_bob",
    )
    calib.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    calib.K.extend(
        [
            531.0409545898438,
            0.0,
            651.0386352539062,
            0.0,
            531.0409545898438,
            351.9400939941406,
            0.0,
            0.0,
            1.0,
        ]
    )
    calib.P.extend(
        [
            531.0409545898438,
            0.0,
            651.0386352539062,
            0.0,
            531.0409545898438,
            351.9400939941406,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    protobuf_writer.write_message(
        "/sensors/exterior_1_left/intrinsics",
        calib,
        publish_time=calib_ts_ns,
        log_time=calib_ts_ns,
    )

    calib = CameraCalibration(
        timestamp=timestamp(calib_ts_ns),
        frame_id="/sensors/exterior_2_left",
        width=640,
        height=360,
        distortion_model="plumb_bob",
    )
    calib.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    calib.K.extend(
        [
            531.0409545898438,
            0.0,
            651.0386352539062,
            0.0,
            531.0409545898438,
            351.9400939941406,
            0.0,
            0.0,
            1.0,
        ]
    )
    calib.P.extend(
        [
            531.0409545898438,
            0.0,
            651.0386352539062,
            0.0,
            531.0409545898438,
            351.9400939941406,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    protobuf_writer.write_message(
        "/sensors/exterior_2_left/intrinsics",
        calib,
        publish_time=calib_ts_ns,
        log_time=calib_ts_ns,
    )

    calib = CameraCalibration(
        timestamp=timestamp(calib_ts_ns),
        frame_id="/sensors/wrist_left",
        width=640,
        height=360,
        distortion_model="plumb_bob",
    )
    calib.D.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    calib.K.extend(
        [
            531.0409545898438,
            0.0,
            651.0386352539062,
            0.0,
            531.0409545898438,
            351.9400939941406,
            0.0,
            0.0,
            1.0,
        ]
    )
    calib.P.extend(
        [
            531.0409545898438,
            0.0,
            651.0386352539062,
            0.0,
            531.0409545898438,
            351.9400939941406,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    protobuf_writer.write_message(
        "/sensors/wrist_left/intrinsics",
        calib,
        publish_time=calib_ts_ns,
        log_time=calib_ts_ns,
    )

    for i, joint in enumerate(robot.actuated_joints):
        print(f"{i}. {joint.name} links {joint.parent} and {joint.child}")

    # for i in range(0, len(dataset)):
    for i in tqdm(range(0, 1000)):
        sample = dataset[i]
        ts_ns = int(sample["timestamp"] * 1e9)

        # transforms
        tfs = FrameTransforms()
        tfs.transforms.append(
            FrameTransform(
                parent_frame_id="scene",
                child_frame_id="panda_link0",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        for i, joint in enumerate(robot.actuated_joints):
            parent_link = joint.parent
            child_link = joint.child
            T_local = joint.get_child_pose(cfg=sample["observation.state"][i % 8])
            trans = T_local[:3, 3]
            quat = rot_matrix_to_quat(T_local[:3, :3])
            tfs.transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    translation=Vector3(
                        x=float(trans[0]), y=float(trans[1]), z=float(trans[2])
                    ),
                    rotation=Quaternion(
                        x=float(quat[0]),
                        y=float(quat[1]),
                        z=float(quat[2]),
                        w=float(quat[3]),
                    ),
                )
            )

        # camera extrinsics and intrinsics
        tfs.transforms.append(
            create_camera_transform(
                "scene",
                "/sensors/exterior_1_left",
                sample["camera_extrinsics.exterior_1_left"],
            )
        )
        tfs.transforms.append(
            create_camera_transform(
                "scene",
                "/sensors/exterior_2_left",
                sample["camera_extrinsics.exterior_2_left"],
            )
        )
        tfs.transforms.append(
            create_camera_transform(
                "panda_link7",
                "/sensors/wrist_left",
                sample["camera_extrinsics.wrist_left"],
            )
        )

        protobuf_writer.write_message(
            topic="/tf",
            message=tfs,
            log_time=ts_ns,  # to microseconds
            publish_time=ts_ns,  # to microseconds
        )

        # camera
        WriteCamera(
            protobuf_writer,
            "/camera/exterior_1_left",
            "/sensors/exterior_1_left",
            sample["observation.images.exterior_1_left"],
            ts_ns,
        )

        WriteCamera(
            protobuf_writer,
            "/camera/exterior_2_left",
            "/sensors/exterior_2_left",
            sample["observation.images.exterior_2_left"],
            ts_ns,
        )

        WriteCamera(
            protobuf_writer,
            "/camera/wrist_left",
            "/sensors/wrist_left",
            sample["observation.images.wrist_left"],
            ts_ns,
        )

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

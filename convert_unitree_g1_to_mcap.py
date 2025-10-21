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
import re


REPO_ID = "./unitreerobotics/G1_Dex3_ToastedBread_Dataset"
URDF_FILE = "./urdf/g1_29dof_rev_1_0.urdf"

joint_names = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]


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


def convert_joint_name_lerobot_to_urdf(name_in):
    """
    Convert lerobot joint names to urdf joint names
    E.g. kLeftShoulderPitch -> left_shoulder_pitch_joint
    """

    camel_case_part = name_in[1:]
    # convert camel case to snake case
    snake_case_part = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_part).lower()

    name_out = f"{snake_case_part}_joint"

    return name_out


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


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--urdf", type=str, default=URDF_FILE)
    argparser.add_argument("--output", type=str, default="unitree_g1.mcap")
    args = argparser.parse_args()

    # 1) Load from the Hub (cached locally)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(dir_path, REPO_ID)
    dataset = LeRobotDataset(REPO_ID, root=cache_dir, episodes=[0])

    # 2) Random access by index
    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    # 3) Load the robot configuration from the urdf file
    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    # 4) Establish joint name mapping
    # Dicard joint measurements that are "fixed" in the urdf
    matches = []
    actuated_joint_names = [j.name for j in robot.actuated_joints]

    for name in joint_names:
        if not name.startswith("k"):
            continue
        converted_name = convert_joint_name_lerobot_to_urdf(name)
        if converted_name in actuated_joint_names:
            matches.append((name, converted_name))

    lerobot_to_urdf_map = {
        lerobot_name: urdf_name for lerobot_name, urdf_name in matches
    }

    joint_positions = {}
    for joint in robot.joints:
        joint_positions[joint.name] = 0.0

    # for i in tqdm(range(0, len(dataset))):
    for i in tqdm(range(1000, 5000)):
        sample = dataset[i]
        state = sample["observation.state"]  # all joint measurements

        ts_ns = int(sample["timestamp"] * 1e9)

        # for urdf configuration before calling forward kinematics
        for k, value in enumerate(state):
            if k < len(joint_names):
                lerobot_name = joint_names[k]
                if lerobot_name in lerobot_to_urdf_map:
                    urdf_name = lerobot_to_urdf_map[lerobot_name]
                    joint_positions[urdf_name] = float(value)

        # forward kinematics
        # the "base_link" of the robot is "pelvis"
        # fix the pelvis in the world right now
        fk_poses = robot.link_fk(cfg=joint_positions)

        # transforms
        tfs = FrameTransforms()
        tfs.transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="pelvis",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )

        for j, joint in enumerate(robot.joints):
            parent_link = "pelvis"
            child_link = joint.child
            # print(f"{parent_link} links to {child_link} by {joint.name}")
            T_local = fk_poses[robot.link_map[child_link]]
            trans = T_local[:3, 3]
            quat = rot_matrix_to_quat(T_local[:3, :3])
            tfs.transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    timestamp=Timestamp(
                        seconds=int(ts_ns // 1e9), nanos=int(ts_ns % 1e9)
                    ),
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

        protobuf_writer.write_message(
            topic="/tf",
            message=tfs,
            log_time=ts_ns,  # to microseconds
            publish_time=ts_ns,  # to microseconds
        )

        # camera
        WriteCamera(
            protobuf_writer,
            "/camera/cam_left_high",
            "/sensors/cam_left_high",
            sample["observation.images.cam_left_high"],
            ts_ns,
        )
        WriteCamera(
            protobuf_writer,
            "/camera/cam_right_high",
            "/sensors/cam_right_high",
            sample["observation.images.cam_right_high"],
            ts_ns,
        )
        WriteCamera(
            protobuf_writer,
            "/camera/cam_left_wrist",
            "/sensors/cam_left_wrist",
            sample["observation.images.cam_left_wrist"],
            ts_ns,
        )
        WriteCamera(
            protobuf_writer,
            "/camera/cam_right_wrist",
            "/sensors/cam_right_wrist",
            sample["observation.images.cam_right_wrist"],
            ts_ns,
        )

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

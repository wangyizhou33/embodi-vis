# For mac ffmpeg and libtorchcodec issue
# Also make sure: pip uninstall torchcodec -y     
import os
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from urdfpy import URDF
from io import BytesIO
import base64
import numpy as np
from scipy.spatial.transform import Rotation
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

REPO_ID = "OpenGalaxea/Galaxea-Open-World-Dataset"
URDF_FILE = "urdf/r1_lite.urdf"

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
    argparser.add_argument("--output", type=str, default="galaxea_r1lite_lerobot.mcap")
    args = argparser.parse_args()

    # 1) Load from the Hub (cached locally)
    # Note the dataset is in v2.1 format
    # Downgrade by pip install lerobot==0.3.2
    cache_dir = os.path.join(dir_path, REPO_ID)
    dataset = LeRobotDataset(REPO_ID, root=cache_dir)

    # 2) Random access by index
    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    joint_positions = {}
    for joint in robot.joints:
        joint_positions[joint.name] = 0.0


    base_link = robot.base_link.name

    # for i in range(0, len(dataset)):
    for i in tqdm(range(0, 200)):
        sample = dataset[i]
        ts_ns = int(sample["timestamp"] * 1e9)

        # Read joint positions from the data
        left_arm_positions = sample["observation.state.left_arm"]
        right_arm_positions = sample["observation.state.right_arm"]
        torso_positions = sample["observation.state.torso"]

        # Assign left arm joint positions
        joint_positions["left_arm_joint1"] = float(left_arm_positions[0])
        joint_positions["left_arm_joint2"] = float(left_arm_positions[1])
        joint_positions["left_arm_joint3"] = float(left_arm_positions[2])
        joint_positions["left_arm_joint4"] = float(left_arm_positions[3])
        joint_positions["left_arm_joint5"] = float(left_arm_positions[4])
        joint_positions["left_arm_joint6"] = float(left_arm_positions[5])

        # Assign right arm joint positions
        joint_positions["right_arm_joint1"] = float(right_arm_positions[0])
        joint_positions["right_arm_joint2"] = float(right_arm_positions[1])
        joint_positions["right_arm_joint3"] = float(right_arm_positions[2])
        joint_positions["right_arm_joint4"] = float(right_arm_positions[3])
        joint_positions["right_arm_joint5"] = float(right_arm_positions[4])
        joint_positions["right_arm_joint6"] = float(right_arm_positions[5])

        joint_positions["torso_joint1"] = float(torso_positions[0])
        joint_positions["torso_joint2"] = float(torso_positions[1])
        joint_positions["torso_joint3"] = float(torso_positions[2])
        # however sample["observation.state.torso"] is shape 4

        # forward kinematics
        # the "base_link" of the robot is "pelvis"
        # fix the pelvis in the world right now
        fk_poses = robot.link_fk(cfg=joint_positions)

        # transforms
        tfs = FrameTransforms()
        tfs.transforms.append(
            FrameTransform(
                timestamp=timestamp(ts_ns),
                parent_frame_id="world",
                child_frame_id=base_link,
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )

        for j, joint in enumerate(robot.joints):
            parent_link = base_link
            child_link = joint.child
            # print(f"{parent_link} links to {child_link} by {joint.name}")
            T_local = fk_poses[robot.link_map[child_link]]
            trans = T_local[:3, 3]
            r = Rotation.from_matrix(T_local[:3, :3])
            quat = r.as_quat()
            tfs.transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    timestamp=timestamp(ts_ns),
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
            "/camera/image_camera_head",
            "/sensors/camera_head",
            sample["observation.images.head_rgb"],
            ts_ns,
        )

        # camera
        WriteCamera(
            protobuf_writer,
            "/camera/image_camera_head_right",
            "/sensors/camera_head_right",
            sample["observation.images.head_right_rgb"],
            ts_ns,
        )

        WriteCamera(
            protobuf_writer,
            "/camera/wrist_left",
            "/sensors/wrist_left",
            sample["observation.images.left_wrist_rgb"],
            ts_ns,
        )

        WriteCamera(
            protobuf_writer,
            "/camera/wrist_right",
            "/sensors/wrist_right",
            sample["observation.images.right_wrist_rgb"],
            ts_ns,
        )

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

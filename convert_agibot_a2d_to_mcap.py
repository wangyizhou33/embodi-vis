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

URDF_FILE = "./urdf/A2D.urdf"

def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--urdf", type=str, default=URDF_FILE)
    argparser.add_argument("--output", type=str, default="agibot_a2d.mcap")
    args = argparser.parse_args()

    # 2) Random access by index
    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    # 3) Load the robot configuration from the urdf file
    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    joint_positions = {}
    for joint in robot.joints:
        joint_positions[joint.name] = 0.0

    for name in robot.actuated_joint_names:
        print(name)


    base_link = robot.base_link.name

    ts_ns = int(0)
    # for i in tqdm(range(0, len(dataset))):
    for i in tqdm(range(1, 1000)):
        ts_ns = ts_ns + int(1e6)

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
                child_frame_id="base_link",
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

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

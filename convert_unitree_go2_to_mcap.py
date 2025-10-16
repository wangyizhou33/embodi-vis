from urdfpy import URDF
import argparse
import pyrender
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mcap.writer import Writer as McapWriter
from ProtobufWriter import ProtobufWriter
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from google.protobuf.timestamp_pb2 import Timestamp
import time

URDF_FILE = "urdf/go2_description.urdf"
CSV_FILE = "go2_motor_states.csv"
DT = 0.002

# Mapping from URDF joint name to CSV column index
# Only map the "actuated" joints
# 12 DOF for go2
joint_name_to_csv_index = {
    "FL_hip_joint": 3,
    "FR_hip_joint": 0,
    "RL_hip_joint": 9,
    "RR_hip_joint": 6,
    "FL_thigh_joint": 4,
    "FR_thigh_joint": 1,
    "RL_thigh_joint": 10,
    "RR_thigh_joint": 7,
    "FL_calf_joint": 5,
    "FR_calf_joint": 2,
    "RL_calf_joint": 11,
    "RR_calf_joint": 8,
}


def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


def load_data(csv_file):
    import csv

    all_joint_values = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                all_joint_values.append([float(v) for v in row])

    # For now, use the first row of values for visualization
    if all_joint_values:
        joint_values = all_joint_values[0]
    else:
        print("Warning: go2_motor_states.csv contains no data rows. Using zeros.")
        joint_values = [0.0] * 12
    return all_joint_values


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


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--urdf", type=str, default=URDF_FILE)
    argparser.add_argument("--output", type=str, default="test.mcap")
    args = argparser.parse_args()

    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    # load the csv file
    all_joint_values = load_data(CSV_FILE)
    joint_positions = {}

    ts_ns = time.time_ns()

    for i in tqdm(range(len(all_joint_values))):
        ts_ns += int(DT * 1e9)

        joint_values = all_joint_values[i]

        # robot forward kinematics
        for joint in robot.joints:
            csv_index = joint_name_to_csv_index.get(joint.name)
            if csv_index is not None and csv_index < len(joint_values):
                joint_positions[joint.name] = joint_values[csv_index]
            else:
                joint_positions[joint.name] = 0.0

        fk_poses = robot.link_fk(cfg=joint_positions)

        # transforms
        base_link = robot.base_link.name
        tfs = FrameTransforms()
        tfs.transforms.append(
            FrameTransform(
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

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

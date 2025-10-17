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
import bisect
from scipy.spatial.transform import Rotation

URDF_FILE = "urdf/go2_description.urdf"
JOINT_FILE = "go2_motor_states.csv"
POSE_FILE = "go2_base_pose.csv"
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


def load_joint_data(csv_file):
    import csv

    all_joint_values = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                all_joint_values.append((int(row[0]), [float(v) for v in row[1:]]))

    if not all_joint_values:
        print("Warning: go2_motor_states.csv contains no data rows. Using zeros.")
        all_joint_values = [(0, [0.0] * 12)]
    return all_joint_values


def load_pose_data(csv_file):
    import csv

    pose_data = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                # ts, x, y, z, roll, pitch, yaw
                pose_data.append([int(row[0])] + [float(v) for v in row[1:]])
    if not pose_data:
        print(f"Warning: {csv_file} contains no data rows.")
    return pose_data


def find_closest_pose(pose_data, target_ts):
    # pose_data is sorted by timestamp (the first element of each sublist)
    timestamps = [p[0] for p in pose_data]

    # Find insertion point
    idx = bisect.bisect_left(timestamps, target_ts)

    if idx == 0:
        return pose_data[0]
    if idx == len(timestamps):
        return pose_data[-1]

    # Check neighbors
    before = pose_data[idx - 1]
    after = pose_data[idx]

    if target_ts - before[0] < after[0] - target_ts:
        return before
    else:
        return after


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

    # load the csv files
    all_joint_values = load_joint_data(JOINT_FILE)
    all_pose_values = load_pose_data(POSE_FILE)
    joint_positions = {}

    for i in tqdm(range(len(all_joint_values))):
        ts_ns, joint_values = all_joint_values[i]

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

        # Find the closest pose and set the world -> base_link transform
        closest_pose = find_closest_pose(all_pose_values, ts_ns)
        _, x, y, z, roll, pitch, yaw = closest_pose

        # Reason to use the following formula is that
        # at the recording side, rpy were calculated using
        # pitch = asin(-mat[2]);
        # roll  = atan2(mat[5], mat[8]);
        # yaw   = atan2(mat[1], mat[0]);
        # so converting to scipy convention, we get the following
        r = Rotation.from_euler("xyz", [roll, pitch, yaw])
        quat_xyzw = r.inv().as_quat()
        q = Quaternion(x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3])

        tfs.transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id=base_link,
                timestamp=timestamp(ts_ns),
                translation=Vector3(x=x, y=y, z=z),
                rotation=q,
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
            log_time=ts_ns,
            publish_time=ts_ns,
        )

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

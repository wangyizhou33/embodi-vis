from urdfpy import URDF
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
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Color_pb2 import Color
from google.protobuf.timestamp_pb2 import Timestamp

URDF_FILE = "./urdf/g1_29dof_rev_1_0.urdf"
DATA_FILE = "./lafan_data/dance2_subject1.npz"

def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)


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
    argparser.add_argument("--output", type=str, default="lafan_retargeted.mcap")
    argparser.add_argument("--urdf", type=str, default=URDF_FILE)
    argparser.add_argument("--data", type=str, default=DATA_FILE)
    argparser.add_argument("--up-axis", type=str, choices=['z', 'y'], default='z', help="The up axis of the input data (default: z). If 'y', data will be converted to z-up.")
    args = argparser.parse_args()

    # 1) Load from lafan dataset
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        exit(1)

    print(f"Loading data from {args.data} ...")
    with np.load(args.data) as data:
        qpos_all = data["qpos"]  # (T, D)
        human_joints = data["human_joints"]

    # 2) Random access by index
    stream = open(args.output, "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    # 3) Load the robot configuration from the urdf file
    print(f"Loading URDF from {args.urdf} ...")
    robot = URDF.load(args.urdf)

    base_link = robot.base_link.name

    # Verify joint count matches data
    # qpos structure: [qw, qx, qy, qz, x, y, z] (7D) + Joint Positions (29D)
    expected_joints = 29
    if len(robot.actuated_joints) != expected_joints:
        print(f"Warning: URDF has {len(robot.actuated_joints)} actuated joints, but data expects {expected_joints}.")

    # Joint mapping based on the provided order
    LAFAN_JOINT_NAMES = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]

    for i in tqdm(range(len(qpos_all))):
        # process data rows
        qpos = qpos_all[i]
        
        # Floating Base [qw, qx, qy, qz, x, y, z] (7D)
        # Note: qpos has qw first.
        floating_base_quat = [qpos[4], qpos[5], qpos[6], qpos[3]] # x, y, z, w
        floating_base_pos = [qpos[0], qpos[1], qpos[2]]

        if args.up_axis == 'y':
            r_corr = Rotation.from_euler('x', 90, degrees=True)
            floating_base_pos = r_corr.apply(floating_base_pos).tolist()
            floating_base_quat = (r_corr * Rotation.from_quat(floating_base_quat)).as_quat().tolist()
        
        joint_positions = qpos[7:]
        
        # Configure robot joints
        cfg = {}
        for j, joint_name in enumerate(LAFAN_JOINT_NAMES):
            if joint_name and j < len(joint_positions):
                cfg[joint_name] = joint_positions[j]
        
        fk_poses = robot.link_fk(cfg=cfg)
        
        ts_ns = i * 33_333_333 # ~30 FPS
        
        # transforms
        tfs = FrameTransforms()
        tfs.transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id=base_link,
                timestamp=timestamp(ts_ns),
                translation=Vector3(x=float(floating_base_pos[0]), y=float(floating_base_pos[1]), z=float(floating_base_pos[2])),
                rotation=Quaternion(x=float(floating_base_quat[0]), y=float(floating_base_quat[1]), z=float(floating_base_quat[2]), w=float(floating_base_quat[3])),
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
        
        protobuf_writer.write_message("tf", tfs, ts_ns)

        # Scene Update for human joints
        scene_update = SceneUpdate()
        entity = scene_update.entities.add()
        entity.id = "human_joints"
        entity.frame_id = "world"
        entity.timestamp.CopyFrom(timestamp(ts_ns))

        current_human_joints = human_joints[i]

        if args.up_axis == 'y':
            r_corr = Rotation.from_euler('x', 90, degrees=True)
            current_human_joints = r_corr.apply(current_human_joints)

        for k in range(len(current_human_joints)):
            pos = current_human_joints[k]
            entity.spheres.add(
                pose=Pose(
                    position=Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                    orientation=Quaternion(x=0, y=0, z=0, w=1),
                ),
                size=Vector3(x=0.05, y=0.05, z=0.05),
                color=Color(r=0, g=1, b=0, a=1.0),
            )
        
        protobuf_writer.write_message("/human_joints", scene_update, ts_ns)

    print(f"The mcap file is saved at {args.output}.")
    writer.finish()
    stream.close()

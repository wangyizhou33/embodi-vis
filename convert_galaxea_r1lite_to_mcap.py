import torch
import tensorflow_datasets as tfds
import tyro
import imageio
from huggingface_hub import HfApi, hf_hub_download
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
from PIL import Image

REPO_ID = "OpenGalaxea/Galaxea-Open-World-Dataset"
URDF_FILE = "./urdf/r1_lite.urdf"

def download_sample_dataset(repo_id, data_dir):
    """
    Download sample dataset files for testing.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if dataset is already downloaded
    info_path = os.path.join(data_dir, "rlds", "part1_r1_lite", "1.0.0", "dataset_info.json")
    if os.path.exists(info_path):
        print(f"Dataset already exists at {data_dir}")
        return
    
    print(f"Downloading sample dataset to {data_dir}...")
    print(f"Please make sure you have accepted the terms at:")
    print(f"https://huggingface.co/datasets/{repo_id}")
    print(f"And set up your Hugging Face token with: export HF_TOKEN=your_token")
    
    # Download the first TFRecord file
    hf_hub_download(
        repo_id=repo_id,
        filename="rlds/part1_r1_lite/1.0.0/merged_dataset_large_r1_lite-train.tfrecord-00000-of-02048",
        repo_type="dataset",
        local_dir=data_dir
    )
    
    # Download dataset metadata files
    hf_hub_download(
        repo_id=repo_id,
        filename="rlds/part1_r1_lite/1.0.0/dataset_info.json",
        repo_type="dataset",
        local_dir=data_dir
    )
    hf_hub_download(
        repo_id=repo_id,
        filename="rlds/part1_r1_lite/1.0.0/features.json",
        repo_type="dataset",
        local_dir=data_dir
    )
    
    # Rename rlds/sample folder to rlds/open_galaxea for tfds recognition
    sample_path = os.path.join(data_dir, "rlds", "part1_r1_lite")
    tfds_path = os.path.join(data_dir, "rlds", "open_galaxea")
    
    if os.path.exists(sample_path) and not os.path.exists(tfds_path):
        os.rename(sample_path, tfds_path)
    
    print("Sample dataset downloaded successfully!")

def timestamp(time_ns: int) -> Timestamp:
    return Timestamp(seconds=time_ns // 1_000_000_000, nanos=time_ns % 1_000_000_000)

def WriteCamera(protobuf_writer, topic, frame_id, image, ts_ns):
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


def main(
    dataset_name: str = "open_galaxea", 
    data_dir: str = "galaxea_data",
    output_dir: str = "extracted_videos",
    num_trajs: int = 5
):
    # First download sample dataset
    download_sample_dataset(REPO_ID, data_dir)
    
    # Load using tfds builder from local directory (pointing to version folder)
    builder = tfds.builder_from_directory(os.path.join(data_dir, "rlds", "part1_r1_lite", "1.0.0"))
    ds = builder.as_dataset(split='train')
    print(f"Successfully loaded dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Videos will be saved to: {output_dir}")

    # 2) Random access by index
    stream = open("galaxea_r1lite.mcap", "wb")
    writer = McapWriter(stream)
    writer.start()
    protobuf_writer = ProtobufWriter(writer)

    # 3) Load the robot configuration from the urdf file
    print(f"Loading URDF from {URDF_FILE} ...")
    robot = URDF.load(URDF_FILE)

    joint_positions = {}
    for joint in robot.joints:
        joint_positions[joint.name] = 0.0

    for name in robot.actuated_joint_names:
        print(name)


    base_link = robot.base_link.name

    # There is no timestamp information in RLDS format
    # so we start from 0 and increment by 0.1sec
    ts_ns = int(0)

    for i, episode in enumerate(tqdm(ds.take(num_trajs), total=num_trajs, desc="Exporting videos")):
        head_frames = []
        instruction = ""
        first_step_printed = False
        
        for step in episode['steps']:
            ts_ns = ts_ns + int(1e8)

            if i == 0 and not first_step_printed:
                # Print keys only for the first step of the first episode
                print(f"step['observation'] keys: {list(step['observation'].keys())}")
                print(f"step keys: {list(step.keys())}")

                first_step_printed = True

            # head_rgb_image = step['observation']['image_camera_head'].numpy()
            # head_frames.append(head_rgb_image)

            # write this instruction to mcap later
            instruction = step['language_instruction'].numpy().decode('utf-8')

            # three RGB cameras
            WriteCamera(
                protobuf_writer,
                "/camera/image_camera_head",
                "/sensors/camera_head",
                Image.fromarray(step['observation']['image_camera_head'].numpy()),
                ts_ns,
            )

            WriteCamera(
                protobuf_writer,
                "/camera/image_camera_wrist_left",
                "/sensors/wrist_left",
                Image.fromarray(step['observation']['image_camera_wrist_left'].numpy()),
                ts_ns,
            )

            WriteCamera(
                protobuf_writer,
                "/camera/image_camera_wrist_right",
                "/sensors/wrist_right",
                Image.fromarray(step['observation']['image_camera_wrist_right'].numpy()),
                ts_ns,
            )

            # Depth images need normalization for JPEG compression
            depth_left = step['observation']['depth_camera_wrist_left'].numpy()
            # Squeeze to remove channel dimension, then normalize to 0-255 for JPEG
            depth_left_normalized = np.squeeze(depth_left)
            # Normalize to 0-255 range based on typical depth values (0-10m mapped to 0-255)
            depth_left_normalized = np.clip(depth_left_normalized, 0, 10)  # Clip at 10 meters
            depth_left_normalized = (depth_left_normalized /10.0 * 255).astype(np.uint8)
            WriteCamera(
                protobuf_writer,
                "/camera/depth_wrist_left",
                "/sensors/wrist_left",
                Image.fromarray(depth_left_normalized, mode="L"),
                ts_ns,
            )

            # Depth images need normalization for JPEG compression
            depth_right = step['observation']['depth_camera_wrist_right'].numpy()
            # Squeeze to remove channel dimension, then normalize to 0-255 for JPEG
            depth_right_normalized = np.squeeze(depth_right)
            # Normalize to 0-255 range based on typical depth values (0-10m mapped to 0-255)
            depth_right_normalized = np.clip(depth_right_normalized, 0, 10)  # Clip at 10 meters
            depth_right_normalized = (depth_right_normalized / 10.0 * 255).astype(np.uint8)
            WriteCamera(
                protobuf_writer,
                "/camera/depth_wrist_right",
                "/sensors/wrist_right",
                Image.fromarray(depth_right_normalized, mode="L"),
                ts_ns,
            )

            # Read joint positions from the data
            left_arm_positions = step['observation']['joint_position_arm_left'].numpy()
            right_arm_positions = step['observation']['joint_position_arm_right'].numpy()

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


        # video_path = os.path.join(output_dir, f"traj_{i}_head_rgb.mp4")
        # try:
        #     imageio.mimsave(video_path, head_frames, fps=15)
        #     print(f"Saved video for episode {i} to {video_path} with instruction: '{instruction}'")
        # except Exception as e:
        #     print(f"Error saving video for episode {i}: {e}")
    writer.finish()
    stream.close()

if __name__ == '__main__':
    tyro.cli(main)

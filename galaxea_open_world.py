import tensorflow_datasets as tfds
import tyro
import os
import imageio
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "OpenGalaxea/Galaxea-Open-World-Dataset"

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
    # hf_hub_download(
    #     repo_id=repo_id,
    #     filename="rlds/part1_r1_lite/1.0.0/text.json",
    #     repo_type="dataset",
    #     local_dir=data_dir
    # )
    
    # Rename rlds/sample folder to rlds/open_galaxea for tfds recognition
    sample_path = os.path.join(data_dir, "rlds", "part1_r1_lite")
    tfds_path = os.path.join(data_dir, "rlds", "open_galaxea")
    
    if os.path.exists(sample_path) and not os.path.exists(tfds_path):
        os.rename(sample_path, tfds_path)
    
    print("Sample dataset downloaded successfully!")


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

    for i, episode in enumerate(tqdm(ds.take(num_trajs), total=num_trajs, desc="Exporting videos")):
        head_frames = []
        instruction = ""
        first_step_printed = False
        
        for step in episode['steps']:
            if i == 0 and not first_step_printed:
                # Print keys only for the first step of the first episode
                print(f"step['observation'] keys: {list(step['observation'].keys())}")
                first_step_printed = True

            head_rgb_image = step['observation']['image_camera_head'].numpy()
            head_frames.append(head_rgb_image)
            instruction = step['language_instruction'].numpy().decode('utf-8')

        video_path = os.path.join(output_dir, f"traj_{i}_head_rgb.mp4")
        try:
            imageio.mimsave(video_path, head_frames, fps=15)
            print(f"Saved video for episode {i} to {video_path} with instruction: '{instruction}'")
        except Exception as e:
            print(f"Error saving video for episode {i}: {e}")


if __name__ == '__main__':
    tyro.cli(main)

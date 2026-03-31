import os
import concurrent.futures
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download, login

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

def download_prithivMLmods_v2_model():
    print("Downloading prithivMLmods/Deep-Fake-Detector-v2-Model...")
    try:
        os.makedirs("models/Deep-Fake-Detector-v2-Model", exist_ok=True)
        snapshot_download(
            repo_id="prithivMLmods/Deep-Fake-Detector-v2-Model", 
            local_dir="models/Deep-Fake-Detector-v2-Model"
        )
        print("Successfully downloaded/cached Deep-Fake-Detector-v2-Model to models/")
    except Exception as e:
        print(f"Failed to download Deep-Fake-Detector-v2-Model: {e}")

def download_prithivMLmods_model():
    print("Downloading prithivMLmods/Deep-Fake-Detector-Model...")
    try:
        os.makedirs("models/Deep-Fake-Detector-Model", exist_ok=True)
        snapshot_download(
            repo_id="prithivMLmods/Deep-Fake-Detector-Model",
            local_dir="models/Deep-Fake-Detector-Model"
        )
        print("Successfully downloaded/cached Deep-Fake-Detector-Model to models/")
    except Exception as e:
        print(f"Failed to download Deep-Fake-Detector-Model: {e}")

def download_wvolf_model():
    print("Downloading Wvolf/ViT_Deepfake_Detection...")
    try:
        os.makedirs("models/ViT_Deepfake_Detection", exist_ok=True)
        snapshot_download(
            repo_id="Wvolf/ViT_Deepfake_Detection",
            local_dir="models/ViT_Deepfake_Detection"
        )
        print("Successfully downloaded/cached ViT_Deepfake_Detection to models/")
    except Exception as e:
        print(f"Failed to download ViT_Deepfake_Detection: {e}")

def download_benchmark_models():
    print("Starting concurrent model downloads...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_prithivMLmods_v2_model),
            executor.submit(download_prithivMLmods_model),
            executor.submit(download_wvolf_model),
        ]
        concurrent.futures.wait(futures)
    print("All models downloaded.")

if __name__ == "__main__":
    download_benchmark_models()

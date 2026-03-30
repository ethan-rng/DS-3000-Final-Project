import os
import concurrent.futures
from huggingface_hub import hf_hub_download, snapshot_download

def download_hugging_face_model():
    print("Downloading brmk/deepfake-detection-model...")
    try:
        os.makedirs("models", exist_ok=True)
        hf_hub_download(
            repo_id="brmk/deepfake-detection-model",
            filename="deepfake_detection_model.h5",
            local_dir="models"
        )
        print("Successfully downloaded/cached brmk model to models/")
    except Exception as e:
        print(f"Failed to download brmk model: {e}")

def download_prithivMLmods_model():
    print("Downloading prithivMLmods/Deep-Fake-Detector-v2-Model...")
    try:
        os.makedirs("models/Deep-Fake-Detector-v2-Model", exist_ok=True)
        snapshot_download(
            repo_id="prithivMLmods/Deep-Fake-Detector-v2-Model", 
            local_dir="models/Deep-Fake-Detector-v2-Model"
        )
        print("Successfully downloaded/cached prithivMLmods model to models/")
    except Exception as e:
        print(f"Failed to download prithivMLmods model: {e}")

def download_benchmark_models():
    print("Starting concurrent model downloads...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(download_hugging_face_model)
        f2 = executor.submit(download_prithivMLmods_model)
        concurrent.futures.wait([f1, f2])
    print("All models downloaded.")

if __name__ == "__main__":
    download_benchmark_models()

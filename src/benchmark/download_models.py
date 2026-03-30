import os
import concurrent.futures
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download, login

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

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
        f2 = executor.submit(download_prithivMLmods_model)
        concurrent.futures.wait([f2])
    print("All models downloaded.")

if __name__ == "__main__":
    download_benchmark_models()

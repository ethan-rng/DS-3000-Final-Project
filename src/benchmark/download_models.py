import os
import concurrent.futures
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download, login

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

def download_hugging_face_model():
    print("Downloading brmk/deepfake-detection...")
    try:
        import os
        os.makedirs("models", exist_ok=True)
        # We must use hf_hub_download to cache the file locally FIRST without loading it. 
        # using keras.saving.load_model directly crashes because the raw Keras 2 metadata 
        # needs to be patched by evaluate.py before loading.
        hf_hub_download(
            repo_id="brmk/deepfake-detection",
            filename="deepfake_detection_model.keras",
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
        # f2 = executor.submit(download_prithivMLmods_model)
        concurrent.futures.wait([f1])
    print("All models downloaded.")

if __name__ == "__main__":
    download_benchmark_models()

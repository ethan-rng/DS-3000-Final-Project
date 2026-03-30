import os
import sys
import glob
import time
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

import requests
from PIL import Image

load_dotenv()

# Constants
HF_TOKEN = os.environ.get("HF_TOKEN")
EDENAI_API_KEY = os.environ.get("EDENAI_API_KEY")

MODEL_CONFIGS = [
    {
        "name": "edenai/deepfake_detection/sightengine",
        "type": "edenai_api",
        "model_string": "image/deepfake_detection/sightengine",
        "provider": "sightengine",
    },
    {
        "name": "edenai/ai_detection/winstonai",
        "type": "edenai_api",
        "model_string": "image/ai_detection/winstonai",
        "provider": "winstonai",
    },
    {
        "name": "brmk/deepfake-detection-model",
        "type": "keras_h5",
        "hf_repo": "brmk/deepfake-detection-model",
        "hf_filename": "deepfake_detection_model.h5",
    },
    {
        "name": "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "type": "hf_cloud",
        "hf_repo": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    },
]

def edenai_inference(filepath, provider):
    url = "https://api.edenai.run/v2/image/deepfake_detection"
    headers = {"Authorization": f"Bearer {EDENAI_API_KEY}"}
    
    with open(filepath, "rb") as file_:
        files = {'file': file_}
        data = {'providers': provider}
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            res_json = response.json()
            # print("DEBUG EDENAI", res_json)
            
            provider_res = res_json.get(provider, {})
            if provider_res.get('status') == 'success':
                score = provider_res.get('deepfake_score', 0)
                # Some providers might use different keys, but deepfake_score is standard.
                return score
            else:
                return None
        except Exception as e:
            print(f"[{provider}] API Error: {e}")
            return None

def hf_cloud_inference(filepath, repo_id):
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    with open(filepath, "rb") as file_:
        data = file_.read()
        try:
            response = requests.post(url, headers=headers, data=data)
            res = response.json()
            if isinstance(res, list):
                # parse [{'label': 'real', 'score': 0.1}, {'label': 'fake', 'score': 0.9}]
                for entry in res:
                    if str(entry.get('label')).lower() in ['fake', 'deepfake', 'ai-generated', 'ai']:
                        return entry.get('score', 0.0)
                # if not matched, maybe it's just probability of fake. Return 1 - real_score
                for entry in res:
                    if str(entry.get('label')).lower() in ['real', 'human']:
                        return 1.0 - entry.get('score', 1.0)
                return res[0].get('score', 0.5) 
            elif isinstance(res, dict) and 'error' in res:
                if 'loading' in res.get('error'):
                    print(f"HF Model loading... sleeping for {res.get('estimated_time', 20)} seconds")
                    time.sleep(res.get('estimated_time', 20))
                    # simple retry
                    return hf_cloud_inference(filepath, repo_id)
            return None
        except Exception as e:
            print(f"[{repo_id}] HF Cloud Error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/cleaned/Data Set 4/Data Set 4/validation", help="Directory containing real/ and fake/ subdirs")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of samples per class")
    args = parser.parse_args()

    # Create figure directory
    os.makedirs("figures", exist_ok=True)

    # Load filepaths
    real_paths = glob.glob(os.path.join(args.dataset_dir, "real", "*.*"))
    fake_paths = glob.glob(os.path.join(args.dataset_dir, "fake", "*.*"))

    if not real_paths and not fake_paths:
        print("No images found in", args.dataset_dir)
        sys.exit(1)

    real_paths = real_paths[:args.limit]
    fake_paths = fake_paths[:args.limit]

    # Combine
    all_paths = real_paths + fake_paths
    # 0 = Real, 1 = Fake
    y_true = np.array([0] * len(real_paths) + [1] * len(fake_paths))

    print(f"Benchmarking on {len(all_paths)} images (Real: {len(real_paths)}, Fake: {len(fake_paths)})")

    # Store predicted probabilities
    y_scores = {cfg["name"]: np.zeros(len(all_paths)) for cfg in MODEL_CONFIGS}
    y_valid = {cfg["name"]: np.zeros(len(all_paths), dtype=bool) for cfg in MODEL_CONFIGS}

    # Keras setup
    keras_model = None
    if any(cfg['type'] == 'keras_h5' for cfg in MODEL_CONFIGS):
        print("Loading local Keras Model...")
        try:
            import tensorflow as tf
            from huggingface_hub import hf_hub_download
            hf_cfg = next(cfg for cfg in MODEL_CONFIGS if cfg['type'] == 'keras_h5')
            model_path = hf_hub_download(repo_id=hf_cfg["hf_repo"], filename=hf_cfg["hf_filename"])
            keras_model = tf.keras.models.load_model(model_path, compile=False)
            print("Loaded Keras model successfully.")
        except Exception as e:
            print("Failed to load Keras model:", e)
            keras_model = None

    for i, img_path in tqdm(enumerate(all_paths), total=len(all_paths)):
        for cfg in MODEL_CONFIGS:
            score = None
            if cfg["type"] == "edenai_api":
                score = edenai_inference(img_path, cfg["provider"])
            elif cfg["type"] == "hf_cloud":
                score = hf_cloud_inference(img_path, cfg["hf_repo"])
            elif cfg["type"] == "keras_h5":
                if keras_model is not None:
                    try:
                        import tensorflow as tf
                        from PIL import Image
                        # Preprocess for Keras (Assuming 256x256 and normalized)
                        img = Image.open(img_path).convert("RGB").resize((256, 256))
                        img_arr = np.array(img) / 255.0
                        img_arr = np.expand_dims(img_arr, axis=0)
                        # Predict
                        pred = keras_model.predict(img_arr, verbose=0)
                        score = float(pred[0][0]) if pred.size == 1 else float(pred[0][1]) # usually second class is fake
                    except Exception as e:
                        score = None
            
            if score is not None:
                y_scores[cfg["name"]][i] = score
                y_valid[cfg["name"]][i] = True
            
            # API rate limits wait
            time.sleep(0.5)
            
    # Calculate metrics
    plt.figure(figsize=(10, 8))
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        valid_idx = y_valid[name]
        
        if not np.any(valid_idx):
            print(f"Model: {name} | FAILED TO GET VALID PREDICTIONS")
            continue
            
        y_t = y_true[valid_idx]
        y_s = y_scores[name][valid_idx]
        y_pred = (y_s >= 0.5).astype(int)
        
        acc = accuracy_score(y_t, y_pred)
        prec = precision_score(y_t, y_pred, zero_division=0)
        rec = recall_score(y_t, y_pred, zero_division=0)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        
        # ROC AUC
        if len(np.unique(y_t)) > 1:
            fpr, tpr, _ = roc_curve(y_t, y_s)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        else:
            roc_auc = float('nan')
        
        print(f"Model: {name}")
        print(f"  Accuracy:  {acc:.4f} ({np.sum(valid_idx)}/{len(all_paths)} valid samples)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}\n")
        
        # Plot Confusion matrix
        cm = confusion_matrix(y_t, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'figures/cm_{name.replace("/", "_")}.png')
        plt.close()

    plt.figure(1) # Switch back to ROC curve figure
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('figures/roc_curves.png')
    plt.close()
    
    print("All figures saved to figures/ output directory.")

if __name__ == "__main__":
    main()

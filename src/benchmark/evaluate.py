import os
# Force Keras 2 to prevent TimeDistributed parsing errors inside Keras 3
os.environ["TF_USE_LEGACY_KERAS"] = "1"
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
        "type": "hf_local",
        "hf_repo": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    },
]

def edenai_inference(filepath, provider, feature="deepfake_detection"):
    url = f"https://api.edenai.run/v2/image/{feature}"
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
                score = provider_res.get('deepfake_score', provider_res.get('ai_score', 0.5))
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
    parser.add_argument("--dataset_dir", default="dataset/cleaned/Data Set 4/validation", help="Directory containing real/ and fake/ subdirs")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of samples per class")
    args = parser.parse_args()

    # Pre-download missing models concurrently
    from src.benchmark.download_models import download_benchmark_models
    download_benchmark_models()

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
            hf_cfg = next(cfg for cfg in MODEL_CONFIGS if cfg['type'] == 'keras_h5')
            model_path = os.path.join("models", hf_cfg["hf_filename"])
            
            # Patch batch_shape in H5 file before loading to fix Keras 3 deserialization errors
            try:
                import h5py, json
                with h5py.File(model_path, "r+") as f:
                    model_config_raw = f.attrs.get("model_config")
                    if model_config_raw:
                        if isinstance(model_config_raw, bytes):
                            model_config_raw = model_config_raw.decode("utf-8")
                        model_config = json.loads(model_config_raw)
                        
                        def patch_dict(d):
                            mod = False
                            if isinstance(d, dict):
                                # Fix batch_shape to batch_input_shape mapping
                                if d.get("class_name") == "InputLayer" and isinstance(d.get("config"), dict) and "batch_shape" in d["config"]:
                                    d["config"]["batch_input_shape"] = d["config"].pop("batch_shape")
                                    mod = True
                                    
                                # Fix Keras 1.x / improperly formatted inbound_nodes that crash Keras 3 deserialization
                                if "inbound_nodes" in d and isinstance(d["inbound_nodes"], list):
                                    nodes = d["inbound_nodes"]
                                    if nodes and isinstance(nodes[0], list):
                                        if len(nodes[0]) > 0 and isinstance(nodes[0][0], str):
                                            # It's a list of tensors instead of list of nodes. Wrap it.
                                            d["inbound_nodes"] = [nodes]
                                            mod = True
                                    elif nodes and isinstance(nodes[0], str):
                                        d["inbound_nodes"] = [[nodes]]
                                        mod = True
                                        
                                for k, v in d.items():
                                    if patch_dict(v): mod = True
                            elif isinstance(d, list):
                                for item in d:
                                    if patch_dict(item): mod = True
                            return mod
                            
                        if patch_dict(model_config):
                            f.attrs["model_config"] = json.dumps(model_config).encode("utf-8")
                            print("Patched 'batch_shape' to 'batch_input_shape' in H5 configuration.")
            except Exception as e:
                print(f"H5Patch Warning: Could not analyze/patch H5 configuration: {e}")

            print(model_path)
            keras_model = tf.keras.models.load_model("./"+model_path, compile=False)
            print("Loaded Keras model successfully.")
        except Exception as e:
            import traceback
            err_str = f"Failed to load Keras model: {e}\n{traceback.format_exc()}"
            print(err_str)
            with open("debug_load_errors.txt", "a") as f:
                f.write(err_str + "\n\n")
            sys.exit(1)

    # HF Local setup
    hf_local_pipelines = {}
    if any(cfg['type'] == 'hf_local' for cfg in MODEL_CONFIGS):
        print("Loading local HuggingFace Pipelines...")
        try:
            from transformers import pipeline
            for cfg in MODEL_CONFIGS:
                if cfg['type'] == 'hf_local':
                    custom_local_path = os.path.join("models", cfg["hf_repo"].split("/")[-1])
                    hf_local_pipelines[cfg["name"]] = pipeline("image-classification", model=custom_local_path, device=-1)
                    print(f"Loaded HF pipeline for {cfg['name']}")
        except Exception as e:
            import traceback
            err_str = f"Failed to load HF pipeline: {e}\n{traceback.format_exc()}"
            print(err_str)
            with open("debug_load_errors.txt", "a") as f:
                f.write(err_str + "\n\n")
            sys.exit(1)

    import concurrent.futures

    def evaluate_model(cfg):
        scores = np.zeros(len(all_paths))
        valid = np.zeros(len(all_paths), dtype=bool)
        
        km = keras_model if cfg["type"] == "keras_h5" else None
        hp = hf_local_pipelines.get(cfg["name"]) if cfg["type"] == "hf_local" else None
        
        # We can use tqdm but it might overlap if too many threads. Since there are 4 models, position based on index helps.
        idx = MODEL_CONFIGS.index(cfg)
        for i, img_path in tqdm(enumerate(all_paths), total=len(all_paths), desc=cfg["name"], position=idx, leave=True):
            score = None
            if cfg["type"] == "edenai_api":
                feature = cfg.get("model_string", "image/deepfake_detection/sightengine").split("/")[1]
                score = edenai_inference(img_path, cfg["provider"], feature)
                time.sleep(0.5)
            elif cfg["type"] == "hf_cloud":
                score = hf_cloud_inference(img_path, cfg["hf_repo"])
                time.sleep(0.5)
            elif cfg["type"] == "hf_local":
                if hp is not None:
                    try:
                        from PIL import Image
                        pil_img = Image.open(img_path).convert("RGB")
                        res = hp(pil_img)
                        for entry in res:
                            val = str(entry.get('label')).lower()
                            if val in ['fake', 'deepfake', 'ai-generated', 'ai', 'fakes']:
                                score = entry.get('score', 0.0)
                                break
                        if score is None:
                            for entry in res:
                                val = str(entry.get('label')).lower()
                                if val in ['real', 'human', 'original', 'reals', 'realism']:
                                    score = 1.0 - entry.get('score', 1.0)
                                    break
                        if score is None:
                            score = res[0].get('score', 0.5)
                    except Exception as e:
                        print(f"[{cfg['name']}] Local Pipeline Error: {e}")
            elif cfg["type"] == "keras_h5":
                if km is not None:
                    try:
                        import tensorflow as tf
                        from PIL import Image
                        img = Image.open(img_path).convert("RGB")
                        
                        # Try predicting with 256x256, and if it fails, retry with 224x224
                        def infer_keras_with_size(size, use_predict=False):
                            img_resized = img.resize((size, size))
                            img_arr = np.array(img_resized) / 255.0
                            
                            try:
                                input_shape = km.input_shape
                                if len(input_shape) == 5:
                                    frames = input_shape[1] if input_shape[1] is not None else 10
                                    img_arr = np.repeat(np.expand_dims(img_arr, axis=0), frames, axis=0)
                            except Exception:
                                pass
                                
                            img_arr = np.expand_dims(img_arr, axis=0)
                            if use_predict:
                                pred = km.predict(img_arr, verbose=0)
                            else:
                                pred = km(img_arr, training=False).numpy()
                                
                            return float(pred[0][0]) if pred.size == 1 else float(pred[0][1])
                        
                        try:
                            score = infer_keras_with_size(256, use_predict=True)
                        except Exception:
                            score = infer_keras_with_size(224, use_predict=True)
                    except Exception as e:
                        print(f"[{cfg['name']}] Keras inference error: {e}")
            
            if score is not None:
                scores[i] = score
                valid[i] = True
                
        return cfg["name"], scores, valid

    print("\nRunning inference across all models in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_CONFIGS)) as executor:
        futures = {executor.submit(evaluate_model, cfg): cfg for cfg in MODEL_CONFIGS}
        for future in concurrent.futures.as_completed(futures):
            name, scores, valid = future.result()
            y_scores[name] = scores
            y_valid[name] = valid
            print(f"Finished evaluating {name}")
            
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

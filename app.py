"""
Flask web UI for training & evaluating the real-vs-fake face classifier.

Run:
    python app.py          # starts on http://localhost:5000
"""

import os
import threading
import time
import traceback
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# ── Repo paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "figures"

# ── Global training state (single-user / single-run) ─────────────────────
_state = {
    "status": "idle",       # idle | training | complete | error
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": None,
    "val_metrics": None,    # dict after each epoch (accuracy, precision, …)
    "test_metrics": None,   # dict after training finishes
    "error": None,
}
_lock = threading.Lock()


def _reset_state():
    _state.update(
        status="idle", epoch=0, total_epochs=0,
        train_loss=None, val_metrics=None, test_metrics=None, error=None,
    )


# ── Progress callback (called from train() each epoch) ───────────────────
def _progress(epoch, total_epochs, train_loss, val_metrics):
    with _lock:
        _state["epoch"] = epoch
        _state["total_epochs"] = total_epochs
        _state["train_loss"] = float(train_loss)
        if val_metrics is not None:
            _state["val_metrics"] = {
                k: v for k, v in val_metrics.items()
                if k in ("accuracy", "precision", "recall", "f1", "auc")
            }


# ── Background training thread ───────────────────────────────────────────
# Models that use sklearn instead of PyTorch
SKLEARN_MODELS = {"logistic_regression", "svm", "random_forest", "knn"}


def _train_worker(dataset_root, backbone, epochs, batch_size, max_samples, out_dir):
    try:
        with _lock:
            _state["status"] = "training"
            _state["total_epochs"] = epochs

        if backbone in SKLEARN_MODELS:
            from src.training.train_sklearn import train_sklearn
            test_metrics = train_sklearn(
                dataset_root=dataset_root,
                model_type=backbone,
                out_dir=out_dir,
                max_samples=max_samples,
                progress_callback=_progress,
            )
        else:
            # Import here so Flask startup is fast even if torch is slow to load
            from src.training.train import train as run_train
            test_metrics = run_train(
                dataset_root=dataset_root,
                model_type=backbone,
                epochs=epochs,
                batch_size=batch_size,
                lr=1e-4,
                out_dir=out_dir,
                max_samples=max_samples,
                progress_callback=_progress,
            )

        with _lock:
            _state["status"] = "complete"
            if test_metrics is not None:
                _state["test_metrics"] = {
                    k: v for k, v in test_metrics.items()
                    if k in ("accuracy", "precision", "recall", "f1", "auc",
                             "classification_report")
                }
                # confusion_matrix is ndarray → convert to nested list
                if "confusion_matrix" in test_metrics:
                    _state["test_metrics"]["confusion_matrix"] = (
                        test_metrics["confusion_matrix"].tolist()
                    )
    except Exception:
        with _lock:
            _state["status"] = "error"
            _state["error"] = traceback.format_exc()


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/download", methods=["POST"])
def api_download():
    try:
        subprocess.Popen(["bash", "scripts/download_script.sh"])
        return jsonify({"status": "started", "message": "Dataset download started in background."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/distort", methods=["POST"])
def api_distort():
    try:
        subprocess.Popen(["python", "scripts/apply_distortions.py", "dataset/raw", "3", "-t", "4"])
        return jsonify({"status": "started", "message": "Applying distortions started in background."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.get_json(force=True)
    env = data.get("env", "cpu")
    try:
        if env == "hpc":
            subprocess.Popen(["sbatch", "scripts/run_benchmark.sh"])
            return jsonify({"status": "started", "message": "Benchmark SLURM job submitted."})
        else:
            subprocess.Popen(["bash", "scripts/run_benchmark.sh"])
            return jsonify({"status": "started", "message": "Benchmark started locally (CPU)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    with _lock:
        if _state["status"] == "training":
            return jsonify({"error": "Training is already in progress."}), 409

    data = request.get_json(force=True)
    env = data.get("env", "cpu")
    max_samples = int(data.get("max_samples", 200))
    backbone = data.get("backbone", "efficientnet_b0")
    epochs = int(data.get("epochs", 5))
    batch_size = int(data.get("batch_size", 32))

    # Sanitise backbone choice
    allowed_backbones = {
        "efficientnet_b0", "mobilenet_v3_large", "resnet50",
        "logistic_regression", "svm", "random_forest", "knn",
    }
    if backbone not in allowed_backbones:
        return jsonify({"error": f"Invalid backbone. Choose from {allowed_backbones}"}), 400

    if env == "hpc":
        cmd = [
            "sbatch",
            "scripts/train.sh",
            "-d", "dataset",
            "-b", backbone,
            "-e", str(epochs),
            "-B", str(batch_size),
            "-o", "models",
            "-m", str(max_samples)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return jsonify({"status": "hpc_submitted", "message": res.stdout.strip()})
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Failed to submit HPC job: {e.stderr}"}), 500

    _reset_state()

    t = threading.Thread(
        target=_train_worker,
        kwargs=dict(
            dataset_root="dataset",
            backbone=backbone,
            epochs=epochs,
            batch_size=batch_size,
            max_samples=max_samples,
            out_dir="models",
        ),
        daemon=True,
    )
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def api_status():
    with _lock:
        payload = {
            "status": _state["status"],
            "epoch": _state["epoch"],
            "total_epochs": _state["total_epochs"],
            "train_loss": _state["train_loss"],
            "val_metrics": _state["val_metrics"],
            "test_metrics": _state["test_metrics"],
            "error": _state["error"],
        }
    return jsonify(payload)


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/api/eda", methods=["POST"])
def api_eda():
    try:
        from src.eda import run_eda
        result = run_eda("dataset")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Starting Flask UI → http://localhost:5001")
    app.run(debug=False, host="0.0.0.0", port=5001)

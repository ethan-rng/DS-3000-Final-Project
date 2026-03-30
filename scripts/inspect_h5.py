import h5py
import json

file_path = "models/deepfake_detection_model.h5"

with h5py.File(file_path, "r+") as f:
    model_config_raw = f.attrs.get("model_config")
    if model_config_raw is None:
        print("No model_config found")
    else:
        model_config = json.loads(model_config_raw)
        # Find InputLayer and fix batch_shape -> batch_input_shape
        modified = False
        if "config" in model_config:
            config = model_config.get("config", {})
            layers = config.get("layers", [])
            for layer in layers:
                if layer.get("class_name") == "InputLayer":
                    if "batch_shape" in layer["config"]:
                        print("Found batch_shape:", layer["config"]["batch_shape"])
                        layer["config"]["batch_input_shape"] = layer["config"].pop("batch_shape")
                        modified = True
            
            if modified:
                f.attrs["model_config"] = json.dumps(model_config).encode("utf-8")
                print("Successfully patched model_config in H5 file")
            else:
                print("No modification needed")
        elif "class_name" in model_config and model_config["class_name"] == "Sequential":
            config = model_config.get("config", {})
            if isinstance(config, list):
                layers = config
            else:
                layers = config.get("layers", [])
            for layer in layers:
                if layer.get("class_name") == "InputLayer":
                    if "batch_shape" in layer["config"]:
                        print("Found batch_shape:", layer["config"]["batch_shape"])
                        layer["config"]["batch_input_shape"] = layer["config"].pop("batch_shape")
                        modified = True
            if modified:
                f.attrs["model_config"] = json.dumps(model_config).encode("utf-8")
                print("Successfully patched model_config in H5 file")
            else:
                print("No modification needed")

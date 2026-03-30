# DS 3000 Final Project

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   ```bash
   bash download_script.sh
   ```

4. **Run the project:**
   ```bash
   python main.py
   ```

## Applying Image Distortions
We provide a specialized script (`scripts/apply_distortions.py`) to simulate real-world transmission and capture artifacts on the deepfake dataset. 

**What it does by default:**
- **Type 1 (Compression):** Simulates WhatsApp/Instagram image degradation by re-encoding images to a lower quality JPEG in-memory.
- **Type 2 (Moiré Effect):** Simulates a digital camera taking a picture of a screen by dynamically generating and blending a subtle, randomized Moiré interference pattern over the image.
- **Type 3 (Both):** Computes the Moiré effect first (capture simulation), followed by the compression (transmission simulation).
- **Multithreading:** The script automatically utilizes all available CPU cores to process images rapidly.

### Usage
**1. Testing on a Subset (Recommended First Step)**
Use the `-n` flag to limit the number of processed images (e.g., `-n 3` for 3 images). 
When testing, the script outputs to `dataset/tmp/` and creates two files side-by-side (`_original` and `_distorted`) so you can visually compare the effect:
```bash
python scripts/apply_distortions.py 'dataset/raw/Data Set 2' 2 -n 3
```

**2. Processing the Full Dataset**
Run without the `-n` flag to process the entire directory. The script will save the results to `dataset/cleaned/`, preserving the exact folder structure and original file names so your data loaders continue to work seamlessly:
```bash
python scripts/apply_distortions.py 'dataset/raw/Data Set 2' 2
```

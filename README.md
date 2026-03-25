# DS 3000 Final Project

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Download the dataset:**
   ```bash
   bash download_script.sh
   ```

4. **Run the project:**
   ```bash
   python main.py
   ```

## Project Structure

```
final-project/
├── main.py              # Main script
├── download_script.sh   # Downloads dataset (too large for GitHub)
├── data/                # Dataset directory (gitignored)
├── README.md
└── .gitignore
```

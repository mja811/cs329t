import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "out"
PROCESSED_COMMENTS_DIR = OUTPUT_DIR / "processed_comments"
OPPOSING_TEXTS = OUTPUT_DIR / "opposing_texts"
RUN_LOGS = OUTPUT_DIR / "run_logs"

def log_to_file(path, text):
    """Append debate text to transcript file."""
    with open(path, "a", encoding="utf-8") as f:
        query = textwrap.fill(text, width=80)
        f.write(query + "\n\n")
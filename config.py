from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "out"
PROCESSED_COMMENTS_DIR = OUTPUT_DIR / "processed_comments"
RUN_LOGS = OUTPUT_DIR / "run_logs"